"""Shared Phase 1 metric, with explicit evaluation input checks."""
import re
from statistics import mean


def parse_match_ids(text):
    if not isinstance(text, str):
        raise TypeError('ID list must be a string')
    if not text.strip():
        return set()
    ids = [value.strip() for value in text.split(',')]
    if any(not re.fullmatch(r'S[23]-[^,\s]+', value) for value in ids):
        raise ValueError('Malformed target ID')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate target ID')
    return set(ids)


def score_entity(true_ids, predicted_ids):
    if isinstance(true_ids, str) or isinstance(predicted_ids, str):
        raise TypeError('Pass collections of IDs, not comma-separated strings')
    truth, prediction = set(true_ids), set(predicted_ids)
    tp = len(truth & prediction)
    denominator = len(prediction) + 0.25 * len(truth)
    return 1.0 if denominator == 0 else 1.25 * tp / denominator


def macro_score(required_ids, truth_by_reference, predictions_by_reference):
    required = list(required_ids)
    ids = set(required)
    if not required or len(required) != len(ids):
        raise ValueError('Required IDs must be nonempty and unique')
    if ids != predictions_by_reference.keys():
        raise KeyError('Predictions must cover exactly the required references')
    if not ids.issubset(truth_by_reference):
        raise KeyError('Missing ground truth')
    return mean(score_entity(truth_by_reference[r], predictions_by_reference[r]) for r in required)
