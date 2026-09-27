"""Regenerate final predictions from raw test TSVs and saved CatBoost models.

The optional V3 cache path avoids repeating the identical legacy retrieval/features.
Use a fresh output directory. This entry point is Linux-tested.
"""
import argparse
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.gpu_runtime'), str(ROOT / 'src')]
import numpy as np
from catboost import CatBoostClassifier
import run_submission as engine


MODEL_HASHES = {
    'A': '1ea59adbbe61b158501508c032f6b4173f2b1da67ccfe84c01070a3e01170f3d',
    'B': 'bc55b94035f7b772a8e80684198b9f687a064be12703e22bd02026906dbd17bc',
}


class EqualBlend:
    """Fixed final experiment: arithmetic mean of A and B probabilities."""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, x, **kwargs):
        return .5 * self.models['A'].predict_proba(x, **kwargs) + .5 * self.models['B'].predict_proba(x, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, required=True, help='Official dataset/test directory')
    parser.add_argument('--output', type=Path, required=True, help='New resumable run directory')
    parser.add_argument('--database', type=Path, default=ROOT / 'data/processed/test_inference_verified.sqlite')
    parser.add_argument('--policy', choices=['hybrid', 'repaired', 'blend'], required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--cache-root', type=Path, help='Original v3_first_submission run directory')
    parser.add_argument('--max-new-batches', type=int, help='Bounded verification; does not assemble incomplete runs')
    args = parser.parse_args()
    assert args.test_dir.name == 'test', 'Pass the official directory named test'
    assert args.workers >= 1 and args.batch_size >= 1
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.database = args.database.resolve()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    engine.DB = args.database
    original_ro = engine.ro
    engine.ro = lambda path=None: original_ro(args.database if path is None else path)
    engine.DATASET_DIR = args.test_dir.resolve().parent
    original_atomic_json = engine.atomic_json
    engine.atomic_json = lambda path, data: original_atomic_json(
        args.output / 'input_database.json'
        if Path(path) == ROOT / 'reports/test_database_verified.json' else path, data)
    metadata = engine.build_db()
    models = {}
    paths = {'A': ROOT / 'artifacts/gpu_challenger/model.cbm', 'B': ROOT / 'artifacts/gpu_challenger_v2/model.cbm'}
    for label, path in paths.items():
        assert engine.digest(path) == MODEL_HASHES[label], 'Model hash differs: ' + label
        model = CatBoostClassifier(thread_count=2)
        model.load_model(str(path))
        assert list(model.classes_) == [0, 1] and len(model.feature_names_) == 22
        models[label] = model
    cache_manifest = None
    if args.cache_root:
        cache_manifest = json.loads((args.cache_root / 'manifest.json').read_text())
        assert cache_manifest['identity']['settings'] == engine.SETTINGS
        assert cache_manifest['features'] == engine.FEATURE_NAMES
        assert cache_manifest['identity']['inputs'] == metadata['input_hashes']
        assert cache_manifest['identity']['feature_code_sha256'] == engine.digest(ROOT / 'src/business_entity_resolution/inference_features.py')
    identity = {
        'inputs': metadata['input_hashes'], 'policy': args.policy,
        'models': MODEL_HASHES, 'thresholds': {'A': .362, 'B': .665},
        'settings': engine.SETTINGS, 'batch_size': args.batch_size,
        'feature_code_sha256': engine.digest(ROOT / 'src/business_entity_resolution/inference_features.py'),
        'retrieval_code_sha256': engine.digest(ROOT / 'scripts/run_submission.py'),
        'runner_sha256': engine.digest(Path(__file__)),
        'cache_run_id': cache_manifest['run_id'] if cache_manifest else None,
        'blend_policy': {'weight_b': .5, 'threshold': .6, 'France': 'B/.665'} if args.policy == 'blend' else None,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['run_id'] == run_id, 'Use a new output directory for changed settings'
    else:
        engine.atomic_json(manifest_path, {'run_id': run_id, 'identity': identity, 'features': engine.FEATURE_NAMES})
    new_batches = 0

    def selected(country):
        if args.policy == 'blend' and country != 'France':
            return EqualBlend(models), .6
        label = 'A' if args.policy == 'hybrid' and country == 'India' else 'B'
        return models[label], .362 if label == 'A' else .665

    if args.cache_root:
        conn = engine.ro()
        for batch_meta in sorted((args.cache_root / 'batches').glob('*/*/batch.json')):
            original = json.loads(batch_meta.read_text())
            assert original['run_id'] == cache_manifest['run_id']
            folder = args.output / 'batches' / batch_meta.parent.parent.name / batch_meta.parent.name
            if engine.valid_batch(folder, original['ids'], run_id):
                continue
            if args.max_new_batches is not None and new_batches >= args.max_new_batches:
                print('Bounded verification finished. Resume without --max-new-batches for full output.')
                return
            for name in ('features.npy', 'candidate_pairs.tsv'):
                assert engine.digest(batch_meta.parent / name) == original['files'][name]
            with (batch_meta.parent / 'candidate_pairs.tsv').open(newline='') as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            assert [r['source1_entity_id'] for r in rows] == original['ids']
            candidates = [r['candidate_entity_ids'].split(',') if r['candidate_entity_ids'] else [] for r in rows]
            refs = [dict(conn.execute('SELECT * FROM records WHERE entity_id=? AND source=1', (r,)).fetchone()) for r in original['ids']]
            assert len({r['country'] for r in refs}) == 1
            model, threshold = selected(refs[0]['country'])
            x = np.load(batch_meta.parent / 'features.npy', mmap_mode='r')
            assert x.shape == (sum(map(len, candidates)), 22)
            probabilities = model.predict_proba(x, thread_count=2)[:, 1] if len(x) else np.empty(0)
            predictions = []
            offset = 0
            for ids in candidates:
                predictions.append([t for t, p in zip(ids, probabilities[offset:offset + len(ids)]) if p >= threshold])
                offset += len(ids)
            folder.parent.mkdir(parents=True, exist_ok=True)
            engine.write_batch(folder, refs, candidates, predictions, x, probabilities, run_id, {'pairs': len(x), 'cache_reused': True})
            new_batches += 1
            print('Scored cached batch', batch_meta.parent.name, flush=True)
        conn.close()
    else:
        with engine.ro() as conn:
            countries = [r[0] for r in conn.execute('SELECT DISTINCT country FROM records WHERE source=1 ORDER BY country')]
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for number, country in enumerate(countries):
                index = None
                conn = engine.ro()
                cursor = conn.execute('SELECT * FROM records WHERE country=? AND source=1 ORDER BY entity_id', (country,))
                batch = 0
                model, threshold = selected(country)
                while True:
                    refs = [dict(r) for r in cursor.fetchmany(args.batch_size)]
                    if not refs:
                        break
                    folder = args.output / 'batches' / f'{number:02d}' / f'{batch:06d}'
                    batch += 1
                    if engine.valid_batch(folder, [r['entity_id'] for r in refs], run_id):
                        continue
                    if args.max_new_batches is not None and new_batches >= args.max_new_batches:
                        print('Bounded verification finished. Resume for full output.')
                        return
                    engine.memory_check()
                    if index is None:
                        index = engine.CountryIndex(args.database, country)
                    result = engine.infer_batch(refs, index, model, threshold, pool, path=args.database)
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    engine.write_batch(folder, refs, *result[:4], run_id, result[4])
                    new_batches += 1
                    print(country, 'batch', batch, result[4], flush=True)
                conn.close()
                del index
                gc.collect()
    output = engine.assemble(args.output, run_id, metadata['counts']['1'])
    # Validator source ships with this project; the test DB remains the external raw corpus.
    engine.DATASET_DIR = ROOT / '6ab10eb3b23ba_student_resource/student_resource/dataset'
    engine.validate_outputs(args.output, output, metadata['counts']['1'])
    print('Complete validated output:', output, flush=True)


if __name__ == '__main__':
    main()
