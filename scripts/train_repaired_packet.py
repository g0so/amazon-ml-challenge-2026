"""Train and package a CatBoost matcher from a verified feature-packet ZIP."""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import psutil
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / 'src'
import sys
sys.path.insert(0, str(sys_path))
from business_entity_resolution.inference_features import FEATURE_NAMES

DEFAULT_BASELINE = ROOT / 'artifacts/frozen_handoff_restore_20260927/artifacts/gpu_challenger/model.cbm'
PARTITIONS = ('train', 'tune', 'confirmation')
PART_FILES = ('X.npy', 'y.npy', 'reference_ids.json', 'countries.json', 'offsets.npy', 'candidates.jsonl', 'truth.jsonl')
TARGET_ID = re.compile(r'S[23]-[^,\s]+\Z')
REFERENCE_ID = re.compile(r'S1-[^,\s]+\Z')
FEATURE_ALIASES = ('n_jac', 'a_jac', 'n_ex', 'a_ex', 'n_miss', 'a_miss', 'n_char', 'a_char',
                   'n_cont', 'a_cont', 'num_agr', 'num_con', 'a_num_both', 'a_num_one',
                   'a_num_none', 'a_num_exact', 'a_num_partial', 'a_num_unshared_ref',
                   'a_num_unshared_tgt', 'int_n_jac_a_miss', 'int_n_char_a_miss', 'int_n_cont_a_miss')


def expected_packet_files(metadata):
    file_hashes = metadata.get('files', metadata.get('file_hashes'))
    if not isinstance(file_hashes, dict):
        raise ValueError('metadata.json must map every payload path to SHA-256 in files or file_hashes')
    return file_hashes


def validate_feature_names(names):
    if names == list(FEATURE_NAMES):
        return 'canonical_v3_names'
    if names == list(FEATURE_ALIASES):
        return 'ordered_v3_aliases'
    raise ValueError('Packet feature names/order do not match the exact 22-column V3 schema')


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_process_mib = 0.0
        self.minimum_available_gib = psutil.virtual_memory().available / 2**30
        self.peak_gpu_mib = None
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            memory = self.process.memory_info()
            self.peak_process_mib = max(self.peak_process_mib, memory.rss / 2**20,
                                        getattr(memory, 'peak_wset', memory.rss) / 2**20)
            self.minimum_available_gib = min(self.minimum_available_gib,
                                             psutil.virtual_memory().available / 2**30)
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=3, check=True)
                values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
                if values:
                    self.peak_gpu_mib = max(self.peak_gpu_mib or 0, max(values))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(2)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return {'peak_process_working_set_mib': round(self.peak_process_mib, 1),
                'minimum_available_host_ram_gib': round(self.minimum_available_gib, 2),
                'peak_gpu_memory_used_mib': self.peak_gpu_mib}


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return sha256_stream(stream)


def digest_directory(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob('*') if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode('utf-8'))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def safe_member(name):
    if '\\' in name or ':' in name:
        raise ValueError(f'Unsafe packet path: {name}')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe packet path: {name}')
    return path


def validate_zip_and_extract(archive_path, staging_root=None):
    archive_path = Path(archive_path).resolve()
    packet_hash = sha256_file(archive_path)
    sidecar = archive_path.with_suffix(archive_path.suffix + '.sha256')
    if sidecar.exists():
        expected = sidecar.read_text(encoding='ascii').split()[0]
        if expected != packet_hash:
            raise ValueError('Packet ZIP SHA-256 sidecar mismatch')
    staging_base = Path(staging_root).resolve() if staging_root is not None else ROOT / 'data/processed/repaired_packets'
    staging = staging_base / packet_hash[:16]
    if staging.exists():
        validate_packet_directory(staging)
        return staging, packet_hash
    temporary = staging.with_name(staging.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError('Packet ZIP contains duplicate paths')
            for member in archive.infolist():
                safe_member(member.filename)
            if archive.testzip() is not None:
                raise ValueError('Packet ZIP CRC check failed')
            if 'metadata.json' not in archive.namelist():
                raise ValueError('Packet ZIP must contain root metadata.json')
            metadata = json.loads(archive.read('metadata.json'))
            expected_files = expected_packet_files(metadata)
            actual_files = {info.filename for info in archive.infolist() if not info.is_dir() and info.filename != 'metadata.json'}
            if actual_files != set(expected_files):
                raise ValueError('Packet payload paths do not exactly match metadata.files')
            for name, expected in expected_files.items():
                safe_member(name)
                with archive.open(name) as stream:
                    if sha256_stream(stream) != expected:
                        raise ValueError(f'Packet payload SHA-256 mismatch: {name}')
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path = safe_member(info.filename)
                destination = temporary.joinpath(*member_path.parts)
                if not destination.resolve().is_relative_to(temporary.resolve()):
                    raise ValueError(f'Packet path escapes extraction root: {info.filename}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info.filename) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        validate_packet_directory(temporary)
        staging.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, staging)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return staging, packet_hash


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f'Blank JSONL row at {path}:{line_number}')
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON at {path}:{line_number}: {error}') from error


def validate_packet_directory(root):
    root = Path(root).resolve()
    metadata_path = root / 'metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Missing packet metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    feature_schema = validate_feature_names(metadata.get('feature_names'))
    expected_files = expected_packet_files(metadata)
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != 'metadata.json'}
    if actual_files != set(expected_files):
        raise ValueError('Extracted packet payload paths do not exactly match metadata.files')
    for name, expected_hash in expected_files.items():
        path = root.joinpath(*safe_member(name).parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Packet path escapes root: {name}')
        if sha256_file(path) != expected_hash:
            raise ValueError(f'Packet payload SHA-256 mismatch: {name}')

    partition_ids = {}
    partition_stats = {}
    for partition in PARTITIONS:
        folder = root / partition
        missing = [name for name in PART_FILES if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{partition} is missing required packet files: {missing}')
        ids = json.loads((folder / 'reference_ids.json').read_text(encoding='utf-8'))
        countries = json.loads((folder / 'countries.json').read_text(encoding='utf-8'))
        if not isinstance(ids, list) or not all(isinstance(value, str) and REFERENCE_ID.fullmatch(value) for value in ids):
            raise ValueError(f'{partition}.reference_ids.json contains invalid reference IDs')
        if len(ids) != len(set(ids)) or len(countries) != len(ids) or not all(isinstance(value, str) for value in countries):
            raise ValueError(f'{partition} reference/country rows are not aligned unique lists')
        X = np.load(folder / 'X.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(folder / 'y.npy', mmap_mode='r', allow_pickle=False)
        offsets = np.load(folder / 'offsets.npy', allow_pickle=False)
        if X.ndim != 2 or X.shape[1] != 22 or X.dtype.kind != 'f' or not np.isfinite(X).all():
            raise ValueError(f'{partition}.X.npy must be a finite floating array with shape (pairs, 22)')
        if y.shape != (len(X),) or y.dtype.kind not in 'biu' or not np.isin(y, (0, 1)).all():
            raise ValueError(f'{partition}.y.npy must be pair-aligned binary labels')
        if offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in 'iu':
            raise ValueError(f'{partition}.offsets.npy must have one integer offset per reference plus endpoint')
        if offsets[0] != 0 or offsets[-1] != len(X) or np.any(np.diff(offsets) < 0):
            raise ValueError(f'{partition}.offsets.npy is not monotone or does not span all pairs')
        candidate_rows = read_jsonl(folder / 'candidates.jsonl')
        truth_rows = read_jsonl(folder / 'truth.jsonl')
        candidates_seen = 0
        positives = 0
        zero_queries = 0
        rows_seen = 0
        for index, (candidate_row, truth_row) in enumerate(zip(candidate_rows, truth_rows)):
            rows_seen += 1
            reference_id = ids[index] if index < len(ids) else None
            if candidate_row.get('reference_id') != reference_id or truth_row.get('reference_id') != reference_id:
                raise ValueError(f'{partition} JSONL order/reference ID mismatch at row {index}')
            candidate_ids = candidate_row.get('candidate_ids')
            true_ids = truth_row.get('true_ids')
            if not isinstance(candidate_ids, list) or not all(isinstance(value, str) and TARGET_ID.fullmatch(value) for value in candidate_ids):
                raise ValueError(f'{partition} candidate IDs malformed at row {index}')
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f'{partition} has duplicate candidate IDs at row {index}')
            if not isinstance(true_ids, list) or not all(isinstance(value, str) and TARGET_ID.fullmatch(value) for value in true_ids):
                raise ValueError(f'{partition} complete truth malformed at row {index}')
            if len(true_ids) != len(set(true_ids)):
                raise ValueError(f'{partition} has duplicate truth IDs at row {index}')
            truth_set = set(true_ids)
            start, end = int(offsets[index]), int(offsets[index + 1])
            if len(candidate_ids) != end - start:
                raise ValueError(f'{partition} candidate/offset alignment mismatch at row {index}')
            expected_labels = np.fromiter((int(target in truth_set) for target in candidate_ids), dtype=np.uint8, count=len(candidate_ids))
            if not np.array_equal(y[start:end], expected_labels):
                raise ValueError(f'{partition} label/candidate/truth alignment mismatch at row {index}')
            candidates_seen += len(candidate_ids)
            positives += int(expected_labels.sum())
            zero_queries += not candidate_ids
        if rows_seen != len(ids):
            raise ValueError(f'{partition} JSONL rows do not cover all {len(ids)} reference IDs')
        if next(candidate_rows, None) is not None or next(truth_rows, None) is not None:
            raise ValueError(f'{partition} has extra candidate or truth rows')
        partition_ids[partition] = set(ids)
        partition_stats[partition] = {'references': len(ids), 'pairs': len(X), 'positives': positives,
                                      'zero_candidate_queries': zero_queries, 'feature_dtype': str(X.dtype)}
        optional_v3 = folder / 'v3_probabilities.npy'
        if optional_v3.exists():
            v3 = np.load(optional_v3, mmap_mode='r', allow_pickle=False)
            if v3.shape != (len(X),) or not np.isfinite(v3).all() or np.any((v3 < 0) | (v3 > 1)):
                raise ValueError(f'{partition}.v3_probabilities.npy is not pair-aligned probabilities')
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1:]:
            overlap = partition_ids[left] & partition_ids[right]
            if overlap:
                raise ValueError(f'{left} and {right} IDs overlap ({len(overlap)} references)')
    metadata['_validated_feature_schema'] = feature_schema
    return metadata, partition_stats


def load_query_rows(root, partition, X, offsets):
    folder = Path(root) / partition
    ids = json.loads((folder / 'reference_ids.json').read_text(encoding='utf-8'))
    countries = json.loads((folder / 'countries.json').read_text(encoding='utf-8'))
    candidates = list(read_jsonl(folder / 'candidates.jsonl'))
    truth = list(read_jsonl(folder / 'truth.jsonl'))
    return [{'reference_id': reference_id, 'country': countries[index],
             'candidate_ids': candidates[index]['candidate_ids'],
             'truth_ids': truth[index]['true_ids'],
             'X': X[int(offsets[index]):int(offsets[index + 1])]}
            for index, reference_id in enumerate(ids)]


def predict(model, X):
    return model.predict_proba(X, thread_count=1)[:, 1] if len(X) else np.empty(0, dtype=np.float32)


def score_predictions(rows, predictions):
    scores = []
    counts = {'tp': 0, 'fp': 0, 'fn': 0, 'retrieval_fn': 0, 'classifier_fn': 0}
    countries = defaultdict(lambda: {'references': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'retrieval_fn': 0})
    singleton_total = singleton_errors = 0
    sources = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for row in rows:
        truth = set(row['truth_ids'])
        candidate_set = set(row['candidate_ids'])
        predicted = set(predictions.get(row['reference_id'], ()))
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        retrieval_fn = len(truth - candidate_set)
        classifier_fn = len((truth & candidate_set) - predicted)
        denominator = len(predicted) + 0.25 * len(truth)
        scores.append(1.25 * tp / denominator if denominator else 1.0)
        for key, value in (('tp', tp), ('fp', fp), ('fn', fn), ('retrieval_fn', retrieval_fn), ('classifier_fn', classifier_fn)):
            counts[key] += value
        country = countries[row['country']]
        country['references'] += 1
        country['tp'] += tp; country['fp'] += fp; country['fn'] += fn; country['retrieval_fn'] += retrieval_fn
        if not truth:
            singleton_total += 1
            singleton_errors += bool(predicted)
        for target in truth:
            sources[target[:2]]['fn'] += int(target not in predicted)
        for target in predicted & truth:
            sources[target[:2]]['tp'] += 1
        for target in predicted - truth:
            sources[target[:2]]['fp'] += 1
    country_scores = defaultdict(list)
    for row in rows:
        truth = set(row['truth_ids'])
        predicted = set(predictions.get(row['reference_id'], ()))
        denominator = len(predicted) + 0.25 * len(truth)
        country_scores[row['country']].append(1.25 * len(truth & predicted) / denominator if denominator else 1.0)
    return {'macro_f05': float(np.mean(scores)) if scores else 1.0, **counts,
            'singleton_references': singleton_total, 'singleton_errors': singleton_errors,
            'singleton_error_rate': singleton_errors / singleton_total if singleton_total else 0.0,
            'country_breakdown': {key: {**value, 'macro_f05': float(np.mean(country_scores[key]))}
                                  for key, value in sorted(countries.items())},
            'source_breakdown': dict(sources)}


def predictions_at_threshold(rows, flat_probabilities, threshold):
    output = {}
    position = 0
    for row in rows:
        end = position + len(row['candidate_ids'])
        output[row['reference_id']] = [target for target, probability in
                                       zip(row['candidate_ids'], flat_probabilities[position:end])
                                       if probability >= threshold]
        position = end
    if position != len(flat_probabilities):
        raise ValueError('Probability rows do not align with query candidates')
    return output


def tune_threshold(rows, flat_probabilities):
    best = None
    results = []
    grid = [0.0] + [round(value / 1000, 3) for value in range(10, 991, 5)] + [1.0]
    for threshold in grid:
        predictions = predictions_at_threshold(rows, flat_probabilities, threshold)
        result = {'threshold': threshold, **score_predictions(rows, predictions)}
        results.append(result)
        if best is None or (result['macro_f05'], threshold) > (best['macro_f05'], best['threshold']):
            best = result
    low = max(1, int((best['threshold'] - 0.01) * 1000))
    high = min(1000, int((best['threshold'] + 0.01) * 1000))
    seen = {row['threshold'] for row in results}
    for tick in range(low, high + 1):
        threshold = round(tick / 1000, 3)
        if threshold in seen:
            continue
        predictions = predictions_at_threshold(rows, flat_probabilities, threshold)
        result = {'threshold': threshold, **score_predictions(rows, predictions)}
        results.append(result)
        if (result['macro_f05'], threshold) > (best['macro_f05'], best['threshold']):
            best = result
    return best, results


def save_predictions(path, rows, predictions):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps({'reference_id': row['reference_id'],
                'predicted_ids': predictions.get(row['reference_id'], [])}, separators=(',', ':')) + '\n')


def oracle_predictions(rows):
    return {row['reference_id']: sorted(set(row['candidate_ids']) & set(row['truth_ids'])) for row in rows}


def write_threshold_csv(path, results):
    fields = ('threshold', 'macro_f05', 'tp', 'fp', 'fn', 'retrieval_fn', 'classifier_fn', 'singleton_error_rate')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packet', type=Path)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    if args.packet.is_dir():
        packet_root = args.packet.resolve()
        packet_hash = digest_directory(packet_root)
        metadata, partition_stats = validate_packet_directory(packet_root)
    else:
        packet_root, packet_hash = validate_zip_and_extract(args.packet)
        metadata, partition_stats = validate_packet_directory(packet_root)
    packet_config = metadata.get('identity', {})
    retrieval_config = metadata.get('retrieval_config', packet_config.get('retrieval_config', packet_config.get('settings', {})))
    config_hash = hashlib.sha256(json.dumps(retrieval_config, sort_keys=True).encode()).hexdigest()
    summary = {'packet_sha256': packet_hash, 'retrieval_config_sha256': config_hash,
               'feature_names': list(FEATURE_NAMES), 'partition_stats': partition_stats}
    if args.verify_only:
        print(json.dumps({'verified': True, **summary}, indent=2), flush=True)
        return
    if not args.baseline.is_file():
        raise FileNotFoundError(f'Frozen baseline model not found: {args.baseline}')
    output = args.output_root.resolve() / f'repaired_packet_model_{packet_hash[:12]}_{time.strftime("%Y%m%d_%H%M%S")}'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing model package: {output}')
    output.mkdir(parents=True, exist_ok=False)

    train_X = np.load(packet_root / 'train/X.npy', mmap_mode='r', allow_pickle=False)
    train_y = np.load(packet_root / 'train/y.npy', mmap_mode='r', allow_pickle=False)
    tune_X = np.load(packet_root / 'tune/X.npy', mmap_mode='r', allow_pickle=False)
    tune_y = np.load(packet_root / 'tune/y.npy', mmap_mode='r', allow_pickle=False)
    confirmation_X = np.load(packet_root / 'confirmation/X.npy', mmap_mode='r', allow_pickle=False)
    tune_offsets = np.load(packet_root / 'tune/offsets.npy', allow_pickle=False)
    confirmation_offsets = np.load(packet_root / 'confirmation/offsets.npy', allow_pickle=False)
    tune_rows = load_query_rows(packet_root, 'tune', tune_X, tune_offsets)
    confirmation_rows = load_query_rows(packet_root, 'confirmation', confirmation_X, confirmation_offsets)
    if len(train_y) == 0 or len(np.unique(train_y)) != 2:
        raise ValueError('TRAIN must contain both positive and negative candidate pairs')
    if len(tune_y) == 0:
        raise ValueError('TUNE must contain candidate pairs for early stopping and calibration')

    params = {'task_type': 'GPU', 'devices': '0', 'loss_function': 'Logloss', 'depth': 6,
              'learning_rate': 0.05, 'iterations': 1500, 'l2_leaf_reg': 3,
              'border_count': 128, 'gpu_ram_part': 0.75, 'random_seed': 2028,
              'od_type': 'Iter', 'od_wait': 100, 'use_best_model': True,
              'verbose': 100, 'allow_writing_files': False}
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    model = CatBoostClassifier(**params)
    fit_start = time.monotonic()
    model.fit(Pool(train_X, label=train_y), eval_set=Pool(tune_X, label=tune_y))
    fit_seconds = time.monotonic() - fit_start
    validation_start = time.monotonic()

    tune_probabilities = predict(model, tune_X)
    selected, threshold_results = tune_threshold(tune_rows, tune_probabilities)
    tune_predictions = predictions_at_threshold(tune_rows, tune_probabilities, selected['threshold'])
    confirmation_probabilities = predict(model, confirmation_X)
    confirmation_predictions = predictions_at_threshold(confirmation_rows, confirmation_probabilities, selected['threshold'])
    confirmation_metrics = score_predictions(confirmation_rows, confirmation_predictions)
    candidate_oracle = score_predictions(confirmation_rows, oracle_predictions(confirmation_rows))

    baseline = CatBoostClassifier()
    baseline.load_model(str(args.baseline.resolve()))
    baseline_tune_probabilities = predict(baseline, tune_X)
    baseline_threshold, baseline_threshold_results = tune_threshold(tune_rows, baseline_tune_probabilities)
    baseline_confirmation_probabilities = predict(baseline, confirmation_X)
    baseline_confirmation_predictions = predictions_at_threshold(
        confirmation_rows, baseline_confirmation_probabilities, baseline_threshold['threshold'])
    baseline_confirmation_metrics = score_predictions(confirmation_rows, baseline_confirmation_predictions)
    baseline_tune_predictions = predictions_at_threshold(tune_rows, baseline_tune_probabilities, baseline_threshold['threshold'])
    validation_seconds = time.monotonic() - validation_start

    model_path = output / 'model.cbm'
    model.save_model(str(model_path))
    parity_rows = min(2048, len(tune_X))
    parity_inputs = np.asarray(tune_X[:parity_rows], dtype=np.float32).copy()
    gpu_parity_probabilities = np.asarray(predict(model, parity_inputs), dtype=np.float32)
    cpu_model = CatBoostClassifier()
    cpu_model.load_model(str(model_path))
    cpu_probabilities = np.asarray(predict(cpu_model, parity_inputs), dtype=np.float32)
    parity_max_difference = float(np.max(np.abs(gpu_parity_probabilities - cpu_probabilities))) if parity_rows else 0.0
    parity_decisions = (cpu_probabilities >= selected['threshold']).astype(np.uint8)
    np.save(output / 'parity_inputs.npy', parity_inputs, allow_pickle=False)
    np.save(output / 'expected_cpu_probabilities.npy', cpu_probabilities, allow_pickle=False)
    np.save(output / 'expected_threshold_decisions.npy', parity_decisions, allow_pickle=False)
    save_predictions(output / 'tuning_predictions.jsonl', tune_rows, tune_predictions)
    save_predictions(output / 'baseline_tuning_predictions.jsonl', tune_rows, baseline_tune_predictions)
    save_predictions(output / 'confirmation_predictions.jsonl', confirmation_rows, confirmation_predictions)
    save_predictions(output / 'baseline_confirmation_predictions.jsonl', confirmation_rows, baseline_confirmation_predictions)
    write_threshold_csv(output / 'threshold_search.csv', threshold_results)
    write_threshold_csv(output / 'baseline_threshold_search.csv', baseline_threshold_results)
    atomic_results = {'packet_sha256': packet_hash, 'retrieval_config_sha256': config_hash,
        'train': partition_stats['train'], 'tune': partition_stats['tune'],
        'confirmation': partition_stats['confirmation'], 'selected_threshold': selected['threshold'],
        'tune_metrics': selected, 'baseline_tune_threshold_and_metrics': baseline_threshold,
        'confirmation_metrics': confirmation_metrics, 'confirmation_candidate_oracle': candidate_oracle,
        'baseline_confirmation_metrics_same_candidates': baseline_confirmation_metrics,
        'fit_seconds': fit_seconds, 'validation_seconds': validation_seconds,
        'best_iteration': model.get_best_iteration(), 'parameters': model.get_all_params(),
        'cpu_model_parity': {'rows': parity_rows, 'max_abs_probability_difference': parity_max_difference,
            'threshold_decision_differences': int(np.count_nonzero((gpu_parity_probabilities >= selected['threshold']) != parity_decisions))}}
    resource_usage = resource_monitor.stop()
    atomic_results['resource_usage'] = resource_usage
    atomic_results['elapsed_seconds'] = time.monotonic() - started
    (output / 'evaluation_report.json').write_text(json.dumps(atomic_results, indent=2) + '\n', encoding='utf-8')
    output_metadata = {'packet_sha256': packet_hash, 'packet_metadata_sha256': sha256_file(packet_root / 'metadata.json'),
        'packet_identity': metadata.get('identity', {}), 'packet_schema_version': metadata.get('schema_version'),
        'packet_feature_names': metadata.get('feature_names'), 'packet_payload_sha256': expected_packet_files(metadata),
        'packet_original_code_hash': metadata.get('original_code_hash'), 'retrieval_config': retrieval_config,
        'retrieval_config_sha256': config_hash, 'feature_names': list(FEATURE_NAMES),
        'feature_version': 'v3_exact_cached_v1_22_columns', 'model_sha256': sha256_file(model_path),
        'baseline_model_sha256': sha256_file(args.baseline), 'train_tune_confirmation_ids_disjoint': True,
        'threshold_selection': 'Official per-query macro F0.5 on complete TUNE truth; confirmation scored once at this fixed threshold.',
        'threshold': selected['threshold'], 'parameters': model.get_all_params(),
        'resource_usage': resource_usage}
    (output / 'metadata.json').write_text(json.dumps(output_metadata, indent=2) + '\n', encoding='utf-8')
    (output / 'requirements.txt').write_text('catboost==1.2.10\nnumpy==2.5.3\npsutil==7.2.2\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# Repaired-retrieval CatBoost model\n\n'
        'The model uses the exact ordered 22-feature V3 schema. TRAIN alone fits the model; TUNE drives early stopping and threshold calibration; CONFIRMATION is evaluated once at the locked threshold. The frozen baseline is calibrated on TUNE and compared on the identical CONFIRMATION candidates.\n\n'
        'Reproduce from the project root with `python scripts/train_repaired_packet.py path/to/repaired_feature_packet.zip`. The ZIP is hash-checked, extracted to a packet-SHA-addressed directory, and all partition offsets, candidate rows, truth rows, labels, hashes and split disjointness are verified before fitting.\n', encoding='utf-8')
    package = output.with_suffix('.zip')
    with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output))
        archive.write(Path(__file__), 'source/train_repaired_packet.py')
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise ValueError('Model package ZIP integrity check failed')
    package.with_suffix(package.suffix + '.sha256').write_text(
        f'{sha256_file(package)}  {package.name}\n', encoding='ascii')
    print(json.dumps(atomic_results, indent=2), flush=True)


if __name__ == '__main__':
    main()