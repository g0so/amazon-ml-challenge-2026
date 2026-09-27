"""Train one bounded CatBoost challenger from a verified GPU feature packet."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

FEATURE_NAMES = [
    'name_jaccard', 'address_jaccard', 'name_exact', 'address_exact',
    'name_missing', 'address_missing', 'name_char', 'address_char',
    'name_containment', 'address_containment', 'number_overlap',
    'number_disjoint', 'address_numbers_both', 'address_numbers_one',
    'address_numbers_none', 'address_numbers_equal', 'address_numbers_partial',
    'address_numbers_unshared_ref', 'address_numbers_unshared_target',
    'name_jaccard_x_address_missing', 'name_char_x_address_missing',
    'name_containment_x_address_missing',
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def packet_digest(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(root)).replace('\\', '/').encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def load_partition(root, name):
    folder = root / name
    ids = json.loads((folder / 'reference_ids.json').read_text())
    countries = json.loads((folder / 'countries.json').read_text())
    candidates = [json.loads(line) for line in (folder / 'candidates.jsonl').read_text().splitlines()]
    truth = [json.loads(line) for line in (folder / 'truth.jsonl').read_text().splitlines()]
    return {
        'name': name,
        'folder': folder,
        'X': np.load(folder / 'X.npy', allow_pickle=False),
        'y': np.load(folder / 'y.npy', allow_pickle=False),
        'offsets': np.load(folder / 'offsets.npy', allow_pickle=False),
        'ids': ids,
        'countries': countries,
        'candidates': candidates,
        'truth': truth,
        'v3': np.load(folder / 'v3_probabilities.npy', allow_pickle=False),
    }


def f05(tp, fp, fn):
    denominator = 1.25 * tp + fp + 0.25 * fn
    return 1.25 * tp / denominator if denominator else 1.0


def score(probabilities, part, threshold):
    scores = []
    tp = fp = fn = 0
    country_stats = {}
    singleton_fp = singleton_total = 0
    for i, ref_id in enumerate(part['ids']):
        start, end = part['offsets'][i:i + 2]
        candidate_ids = part['candidates'][i]['candidate_ids']
        truth_ids = set(part['truth'][i]['true_ids'])
        predicted = set() if threshold is None else {
            candidate for candidate, probability in zip(candidate_ids, probabilities[start:end])
            if probability >= threshold
        }
        local_tp = len(predicted & truth_ids)
        local_fp = len(predicted - truth_ids)
        local_fn = len(truth_ids - predicted)
        scores.append(f05(local_tp, local_fp, local_fn))
        tp += local_tp
        fp += local_fp
        fn += local_fn
        country = part['countries'][i]
        bucket = country_stats.setdefault(country, [0, 0, 0, 0])
        bucket[0] += local_tp
        bucket[1] += local_fp
        bucket[2] += local_fn
        bucket[3] += 1
        if not truth_ids:
            singleton_total += 1
            singleton_fp += local_fp
    macro = float(np.mean(scores))
    countries = {
        country: {
            'references': values[3],
            'macro_f05': f05(values[0], values[1], values[2]),
            'tp': values[0], 'fp': values[1], 'fn': values[2],
        }
        for country, values in sorted(country_stats.items())
    }
    return {
        'macro_f05': macro, 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'countries': countries,
        'singleton_references': singleton_total,
        'singleton_false_positive_rate': singleton_fp / singleton_total if singleton_total else 0.0,
    }


def threshold_search(probabilities, part):
    records = []
    for threshold in [round(i / 100, 2) for i in range(1, 100)]:
        metrics = score(probabilities, part, threshold)
        records.append({'threshold': threshold, **metrics})
    none_metrics = score(probabilities, part, None)
    records.append({'threshold': None, **none_metrics})
    best = max(records, key=lambda row: (row['macro_f05'], float('inf') if row['threshold'] is None else row['threshold']))
    center = 0.25 if best['threshold'] is None else best['threshold']
    for index in range(max(1, round((center - 0.02) * 500), 1), min(499, round((center + 0.02) * 500)) + 1):
        threshold = round(index / 500, 3)
        if all(row['threshold'] != threshold for row in records):
            metrics = score(probabilities, part, threshold)
            records.append({'threshold': threshold, **metrics})
    best = max(records, key=lambda row: (row['macro_f05'], float('inf') if row['threshold'] is None else row['threshold']))
    return best, sorted(records, key=lambda row: (float('inf') if row['threshold'] is None else row['threshold']))


def probability_column(model, X):
    return model.predict_proba(X, thread_count=1)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packet', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/gpu_challenger'))
    args = parser.parse_args()
    root = args.packet.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = load_partition(root, 'train')
    tune = load_partition(root, 'tune')
    started = time.perf_counter()
    model = CatBoostClassifier(
        loss_function='Logloss', task_type='GPU', devices='0', random_seed=2028,
        depth=6, learning_rate=0.05, iterations=800, l2_leaf_reg=3,
        od_type='Iter', od_wait=80, use_best_model=True, verbose=100,
        allow_writing_files=False,
    )
    model.fit(train['X'], train['y'], eval_set=(tune['X'], tune['y']))
    fit_seconds = time.perf_counter() - started
    tune_probabilities = probability_column(model, tune['X'])
    selected, search = threshold_search(tune_probabilities, tune)
    v3_metrics = score(tune['v3'], tune, 0.25)
    model.save_model(str(output / 'model.cbm'))

    parity_inputs = tune['X'][:2048].copy()
    parity_probabilities = tune_probabilities[:2048].copy()
    np.save(output / 'parity_inputs.npy', parity_inputs)
    np.save(output / 'parity_probabilities.npy', parity_probabilities)
    cpu_model = CatBoostClassifier()
    cpu_model.load_model(str(output / 'model.cbm'))
    cpu_probabilities = probability_column(cpu_model, parity_inputs)
    max_difference = float(np.max(np.abs(cpu_probabilities - parity_probabilities)))
    threshold_differences = int(np.count_nonzero((cpu_probabilities >= selected['threshold']) != (parity_probabilities >= selected['threshold']))) if selected['threshold'] is not None else 0

    predictions_path = output / 'tuning_predictions.jsonl'
    with predictions_path.open('w', encoding='utf-8') as stream:
        for i, ref_id in enumerate(tune['ids']):
            start, end = tune['offsets'][i:i + 2]
            predicted = [candidate for candidate, probability in zip(tune['candidates'][i]['candidate_ids'], tune_probabilities[start:end]) if selected['threshold'] is not None and probability >= selected['threshold']]
            stream.write(json.dumps({'reference_id': ref_id, 'predicted_ids': predicted}) + '\n')
    with (output / 'threshold_search.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['threshold', 'macro_f05', 'tp', 'fp', 'fn', 'precision', 'recall', 'singleton_false_positive_rate']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in search)

    tuning_results = {'challenger': selected, 'v3_baseline': v3_metrics, 'baseline_packet_score': 0.8839352147713383, 'improved_over_v3': selected['macro_f05'] > v3_metrics['macro_f05'], 'fit_seconds': fit_seconds, 'best_iteration': model.get_best_iteration(), 'parity': {'rows': len(parity_inputs), 'max_abs_probability_difference': max_difference, 'threshold_decision_differences': threshold_differences}}
    (output / 'tuning_results.json').write_text(json.dumps(tuning_results, indent=2) + '\n')
    (output / 'environment.txt').write_text('\n'.join([
        f'python={platform.python_version()}', f'numpy={np.__version__}',
        f'catboost={model.get_metadata().get("catboost_version", "1.2.10")}',
        'task_type=GPU', 'devices=0',
    ]) + '\n')
    metadata = {
        'packet_sha256': packet_digest(root), 'feature_names': FEATURE_NAMES,
        'selected_threshold': selected['threshold'], 'seed': 2028,
        'parameters': model.get_all_params(), 'best_iteration': model.get_best_iteration(),
        'fit_seconds': fit_seconds, 'device': 'GPU:0', 'positive_class': 1,
        'train_pairs': len(train['y']), 'train_positives': int(train['y'].sum()),
        'tune_pairs': len(tune['y']), 'tune_references': len(tune['ids']),
        'model_sha256': sha256(output / 'model.cbm'),
        'parity_max_abs_probability_difference': max_difference,
        'parity_threshold_decision_differences': threshold_differences,
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    (output / 'README.md').write_text('# GPU CatBoost challenger\n\nNative model uses the 22 packet features and positive class 1. CPU inference: load `model.cbm` with `CatBoostClassifier().load_model(...)` and call `predict_proba(X)[:, 1]`. Threshold selection is official per-reference macro F0.5 on tune only.\n')
    print(json.dumps(tuning_results, indent=2))


if __name__ == '__main__':
    main()
