"""One bounded tune/untouched-confirmation experiment on legacy candidates."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.gpu_runtime')]
import numpy as np
from catboost import CatBoostClassifier
from business_entity_resolution.inference_features import feature_job, values, FEATURE_NAMES
from business_entity_resolution.retrieval_v5 import CountryIndexV5, RetrievalConfig

OUT = ROOT / 'reports/final_blend_experiment'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def ro(path):
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def score(part, probabilities, threshold):
    selected = probabilities >= threshold
    n = len(part['truth_counts'])
    count = np.bincount(part['pair_refs'], weights=selected, minlength=n)
    tp = np.bincount(part['pair_refs'], weights=selected & part['labels'], minlength=n)
    denom = count + .25 * part['truth_counts']
    scores = np.divide(1.25 * tp, denom, out=np.ones(n), where=denom != 0)
    return scores, {
        'macro_f05': float(np.mean(scores)),
        'countries': {c: float(np.mean(scores[part['countries'] == c])) for c in ('India', 'US')},
        'tp': int(tp.sum()), 'fp': int(count.sum() - tp.sum()),
        'fn': int(part['truth_counts'].sum() - tp.sum()),
        'singleton_errors': int(np.count_nonzero(count[part['truth_counts'] == 0])),
        'singletons': int(np.count_nonzero(part['truth_counts'] == 0)),
    }


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'selection.json').exists(), 'Do not reuse exposed confirmation for another search'
    exclusions = set()
    for parent in (ROOT / 'data/processed/gpu_feature_packet_v1', ROOT / 'reports/luna_export'):
        for path in parent.glob('*/reference_ids.json'):
            exclusions.update(json.loads(path.read_text()))
    for name in ('data/processed/phase4_train_sample_ids.json', 'data/processed/phase4_dev_tuning_ids.json', 'data/processed/phase3_pilot_sample_ids.json', 'reports/retrieval_v4_sample_ids.json'):
        exclusions.update(json.loads((ROOT / name).read_text()))
    split = ro(ROOT / 'data/processed/phase2_split.sqlite')
    eligible = [r[0] for r in split.execute("SELECT reference_id FROM reference_assignments WHERE split='dev' ORDER BY reference_id") if r[0] not in exclusions]
    sampled = random.Random(202609272314).sample(eligible, 6000)
    ids = {'tune': sampled[:3000], 'confirmation': sampled[3000:]}
    assert not set(ids['tune']) & set(ids['confirmation'])
    save(OUT / 'sample_manifest.json', {'ids': ids, 'excluded_count': len(exclusions), 'purpose': 'Previously uninspected DEV queries. Select on tune only; confirmation once.'})
    conn = ro(ROOT / 'data/processed/phase2_baseline_dev.sqlite')
    conn.execute('ATTACH DATABASE ? AS train_db', ((ROOT / 'data/processed/phase4_train_text.sqlite').resolve().as_uri() + '?mode=ro',))
    refs = {}
    truth = {}
    for off in range(0, len(sampled), 800):
        chunk = sampled[off:off + 800]
        placeholders = ','.join('?' for _ in chunk)
        for row in conn.execute('SELECT * FROM records WHERE entity_id IN (' + placeholders + ')', chunk):
            refs[row['entity_id']] = dict(row)
        for row in conn.execute('SELECT reference_id,target_id FROM truth_edges WHERE reference_id IN (' + placeholders + ')', chunk):
            truth.setdefault(row[0], set()).add(row[1])
    assert set(refs) == set(sampled)
    for rid in sampled:
        expected_count = split.execute('SELECT match_count FROM reference_assignments WHERE reference_id=?', (rid,)).fetchone()[0]
        truth.setdefault(rid, set())
        assert len(truth[rid]) == expected_count
    split.close()
    models = {}
    for key, name, expected in [('a', 'gpu_challenger', '1ea59adbbe61b158501508c032f6b4173f2b1da67ccfe84c01070a3e01170f3d'), ('b', 'gpu_challenger_v2', 'bc55b94035f7b772a8e80684198b9f687a064be12703e22bd02026906dbd17bc')]:
        path = ROOT / 'artifacts' / name / 'model.cbm'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        models[key] = CatBoostClassifier(thread_count=2)
        models[key].load_model(str(path))
    cfg = RetrievalConfig(max_df_abs=5000, top_k_cands=50, route_exact_name=False, route_exact_address=False, route_joint=False, union_baseline=False)
    save(OUT / 'configuration.json', {'retrieval': cfg.__dict__, 'feature_names': FEATURE_NAMES, 'weight_b_grid': [.25, .5, .75, 1.0], 'threshold_grid': sorted(set([round(x / 100, 3) for x in range(20, 86, 2)] + [.362, .665])), 'France': 'Unchanged Model B/.665'})
    with mp.get_context('spawn').Pool(4) as pool:
        for country in ('India', 'US'):
            print('Building original candidate index:', country, flush=True)
            index = CountryIndexV5(conn, country, cfg, include_train_db=True)
            for part, part_ids in ids.items():
                part_refs = [refs[r] for r in part_ids if refs[r]['country'] == country]
                for off in range(0, len(part_refs), 200):
                    chunk = part_refs[off:off + 200]
                    path = OUT / f'{part}_{country}_{off:05d}.npz'
                    if path.exists():
                        continue
                    candidate_lists = [index.retrieve(ref) for ref in chunk]
                    required = sorted({t for row in candidate_lists for t in row})
                    targets = {}
                    for k in range(0, len(required), 800):
                        target_ids = required[k:k + 800]
                        marks = ','.join('?' for _ in target_ids)
                        query = f'SELECT entity_id,name_raw,address_raw,name_norm,address_norm FROM main.records WHERE entity_id IN ({marks}) UNION ALL SELECT entity_id,name_raw,address_raw,name_norm,address_norm FROM train_db.records WHERE entity_id IN ({marks})'
                        for row in conn.execute(query, target_ids + target_ids):
                            assert row['entity_id'] not in targets
                            targets[row['entity_id']] = values(row)
                    assert set(targets) == set(required)
                    jobs = [(values(ref), [targets[t] for t in cands]) for ref, cands in zip(chunk, candidate_lists)]
                    features = pool.map(feature_job, jobs, chunksize=8)
                    x = np.asarray([f for batch in features for f in batch], dtype=np.float64).reshape(-1, 22)
                    lengths = np.asarray([len(c) for c in candidate_lists])
                    labels = np.asarray([t in truth[ref['entity_id']] for ref, cands in zip(chunk, candidate_lists) for t in cands], dtype=bool)
                    np.savez_compressed(path, ids=np.asarray([ref['entity_id'] for ref in chunk]), countries=np.asarray([country] * len(chunk)), lengths=lengths, labels=labels, truth_counts=np.asarray([len(truth[ref['entity_id']]) for ref in chunk]), a=models['a'].predict_proba(x, thread_count=2)[:, 1] if len(x) else np.empty(0), b=models['b'].predict_proba(x, thread_count=2)[:, 1] if len(x) else np.empty(0))
                    print(part, country, off + len(chunk), '/', len(part_refs), 'pairs', len(x), 'elapsed', round(time.monotonic() - started), flush=True)
            del index
            gc.collect()
    conn.close()

    def load(part):
        arrays = {}
        for path in sorted(OUT.glob(part + '_*.npz')):
            with np.load(path, allow_pickle=False) as d:
                for key in d.files:
                    arrays.setdefault(key, []).append(d[key])
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        assert set(arrays['ids']) == set(ids[part]) and len(arrays['ids']) == 3000
        arrays['pair_refs'] = np.repeat(np.arange(3000), arrays['lengths'])
        return arrays

    tune = load('tune')
    baseline_scores, baseline = score(tune, tune['b'], .665)
    search = []
    for weight in [.25, .5, .75, 1.]:
        probabilities = weight * tune['b'] + (1 - weight) * tune['a']
        for threshold in sorted(set([round(x / 100, 3) for x in range(20, 86, 2)] + [.362, .665])):
            _, metrics = score(tune, probabilities, threshold)
            search.append({'weight_b': weight, 'threshold': threshold, **metrics})
    # Require nonnegative observed gains in both labeled countries on tune.
    eligible = [r for r in search if all(r['countries'][c] >= baseline['countries'][c] for c in ('India', 'US'))]
    selected = max(eligible or search, key=lambda r: (r['macro_f05'], r['weight_b'], r['threshold']))
    save(OUT / 'selection.json', {'selected_on_tune_only': selected, 'baseline_tune': baseline, 'search': search})
    # Confirmation is first scored only after the selected setting is saved.
    confirmation = load('confirmation')
    base_scores, base = score(confirmation, confirmation['b'], .665)
    probabilities = selected['weight_b'] * confirmation['b'] + (1 - selected['weight_b']) * confirmation['a']
    candidate_scores, candidate = score(confirmation, probabilities, selected['threshold'])
    differences = candidate_scores - base_scores
    rng = np.random.default_rng(314159)
    bootstrap = np.asarray([differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(3000)])
    interval = np.quantile(bootstrap, [.025, .975])
    supported = bool(interval[0] > 0 and all(candidate['countries'][c] >= base['countries'][c] for c in ('India', 'US')))
    report = {'selected': selected, 'confirmation_baseline': base, 'confirmation_candidate': candidate, 'difference': float(differences.mean()), 'paired_bootstrap_95_percent': interval.tolist(), 'supported_on_fresh_confirmation': supported, 'elapsed_seconds': time.monotonic() - started, 'France': 'Keep exact current B/.665 predictions; no France accuracy claim.', 'note': 'One tune-selected challenger. No further tuning on confirmation.'}
    save(OUT / 'result.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
