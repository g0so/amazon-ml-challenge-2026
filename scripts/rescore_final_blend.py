"""Apply the frozen, independently confirmed policy to existing V3 TEST features."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.gpu_runtime'), str(ROOT / 'src')]
import numpy as np
from catboost import CatBoostClassifier
from business_entity_resolution.inference_features import FEATURE_NAMES


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    result = json.loads((ROOT / 'reports/final_blend_experiment/aggregate_result.json').read_text())
    # This is an explicitly exploratory portal attempt, not a proven improvement.
    assert result['difference'] > 0 and result['selected']['weight_b'] == .5 and result['selected']['threshold'] == .6
    weight = result['selected']['weight_b']
    threshold = result['selected']['threshold']
    out = ROOT / 'reports/final_blend_submission'
    out.mkdir(exist_ok=True)
    final = out / 'matching_results.tsv'
    assert not final.exists(), 'Preserve previous completed result'
    cache = ROOT / 'data/processed/inference_runs/v3_first_submission'
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert manifest['features'] == FEATURE_NAMES
    assert manifest['identity']['feature_code_sha256'] == digest(ROOT / 'src/business_entity_resolution/inference_features.py')
    expected_settings = {'m_tokens': 5, 'max_df': 5000, 'max_df_ratio': .05, 'k': 50, 'feature_version': 'v3_exact_cached_v1'}
    assert manifest['identity']['settings'] == expected_settings
    conn = sqlite3.connect((ROOT / 'data/processed/test_inference_verified.sqlite').as_uri() + '?mode=ro', uri=True)
    required = dict(conn.execute('SELECT entity_id,country FROM records WHERE source=1'))
    conn.close()
    assert len(required) == 1732544
    models = {}
    hashes = {'A': '1ea59adbbe61b158501508c032f6b4173f2b1da67ccfe84c01070a3e01170f3d', 'B': 'bc55b94035f7b772a8e80684198b9f687a064be12703e22bd02026906dbd17bc'}
    for label, folder in [('A', 'gpu_challenger'), ('B', 'gpu_challenger_v2')]:
        path = ROOT / 'artifacts' / folder / 'model.cbm'
        assert digest(path) == hashes[label]
        models[label] = CatBoostClassifier().load_model(str(path))
        assert len(models[label].feature_names_) == 22
    policy = {'status': 'EXPERIMENTAL: confirmation interval includes zero; retain 0.787 unless portal score improves', 'weight_b': weight, 'threshold': threshold, 'France': {'weight_b': 1., 'threshold': .665}, 'model_hashes': hashes, 'cache_run_id': manifest['run_id'], 'confirmation': result}
    (out / 'policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    counts = {}; changed = {}; references = 0; pairs = 0
    batch_paths = sorted((cache / 'batches').glob('*/*/batch.json'))
    assert len(batch_paths) == 1734
    temp = out / 'matching_results.tsv.partial'
    with temp.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        for batch_number, meta_path in enumerate(batch_paths, 1):
            meta = json.loads(meta_path.read_text())
            assert meta['run_id'] == manifest['run_id']
            folder = meta_path.parent
            # Feature files are read sequentially, bounded to a single batch.
            for name in ('features.npy', 'candidate_pairs.tsv'):
                assert digest(folder / name) == meta['files'][name], str(folder / name)
            with (folder / 'candidate_pairs.tsv').open(newline='') as cf:
                reader = csv.reader(cf, delimiter='\t')
                assert next(reader) == ['source1_entity_id', 'candidate_entity_ids']
                rows = list(reader)
            assert [r[0] for r in rows] == meta['ids']
            countries = {required[r[0]] for r in rows}
            assert len(countries) == 1
            country = next(iter(countries))
            candidates = [r[1].split(',') if r[1] else [] for r in rows]
            x = np.load(folder / 'features.npy', mmap_mode='r')
            assert x.shape == (sum(map(len, candidates)), 22)
            b = models['B'].predict_proba(x, thread_count=2)[:, 1] if len(x) else np.empty(0)
            tau = .665 if country == 'France' else threshold
            p = b if country == 'France' or weight == 1 or not len(x) else weight * b + (1 - weight) * models['A'].predict_proba(x, thread_count=2)[:, 1]
            assert np.isfinite(p).all()
            offset = 0
            for row, ids in zip(rows, candidates):
                rid = row[0]
                assert len(ids) == len(set(ids))
                stop = offset + len(ids)
                predicted = [tid for tid, prob in zip(ids, p[offset:stop]) if prob >= tau]
                baseline = [tid for tid, prob in zip(ids, b[offset:stop]) if prob >= .665]
                changed[country] = changed.get(country, 0) + (predicted != baseline)
                assert set(predicted).issubset(ids)
                assert required.pop(rid) == country
                writer.writerow([rid, ','.join(predicted)])
                counts[country] = counts.get(country, 0) + 1
                references += 1
                offset = stop
            assert offset == len(x)
            pairs += len(x)
            if batch_number % 100 == 0:
                print(batch_number, '/', len(batch_paths), 'references', references, 'elapsed', round(time.monotonic() - started), flush=True)
    assert not required and references == 1732544 and pairs == 138530079
    assert changed.get('France', 0) == 0
    temp.replace(final)
    report = {'passed': True, 'references': references, 'pairs': pairs, 'countries': counts, 'changed_vs_B_0665': changed, 'sha256': digest(final), 'seconds': time.monotonic() - started, 'candidate_source': str(ROOT / 'reports/rescored_v3_submission/candidate_pairs.tsv'), 'checks': 'Exact verified cache feature/candidate hashes; model hashes; schema; unique complete test references; offsets; no duplicate candidate IDs; matches subset; France unchanged'}
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
