"""Independent final checks; never rewrites any submission."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path, column):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream, delimiter='\t')
        assert next(reader) == ['source1_entity_id', column], str(path)
        for line, row in enumerate(reader, 2):
            assert len(row) == 2, (str(path), line)
            yield row


def parsed(value):
    ids = value.split(',') if value else []
    assert len(ids) == len(set(ids)), 'Repeated target'
    assert all(x.startswith(('S2-', 'S3-')) and not any(c.isspace() for c in x) for x in ids)
    return set(ids)


def main():
    start = time.monotonic()
    a = ROOT / 'data/processed/inference_runs/catboost_first_submission/output/matching_results.tsv'
    b = ROOT / 'reports/rescored_v3_submission/matching_results.tsv'
    h = ROOT / 'reports/hybrid_submission/matching_results.tsv'
    ca = a.with_name('candidate_pairs.tsv')
    cb = b.with_name('candidate_pairs.tsv')
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (a, b, h, ca, cb)}
    assert hashes[str(a.relative_to(ROOT))] == 'aaf4659d58b131c2f0746e65a4329d750bdb292d510323cb1ebd13eecab5b9c1'
    assert hashes[str(b.relative_to(ROOT))] == '5d885b5e7c030634d5fd30b7d5d444b8477a7bc425fa0778b211890aa5d413a4'
    assert hashes[str(h.relative_to(ROOT))] == '5f4239110991859cd98355aa4beeef7ee25a72f5edfe0077669160dde2c41ac4'
    conn = sqlite3.connect((ROOT / 'data/processed/test_inference_verified.sqlite').as_uri() + '?mode=ro', uri=True)
    codes = {'India': 0, 'US': 1, 'France': 2}
    countries = {eid: codes[c] for eid, c in conn.execute('SELECT entity_id,country FROM records WHERE source=1')}
    assert len(countries) == 1732544
    expected = {}
    for rid, prediction in rows(b, 'matched_entity_ids'):
        assert rid in countries and rid not in expected
        parsed(prediction)
        expected[rid] = prediction
    assert len(expected) == len(countries)
    seen = set()
    changed = Counter()
    for rid, prediction in rows(a, 'matched_entity_ids'):
        assert rid in countries and rid not in seen
        seen.add(rid)
        parsed(prediction)
        if countries[rid] == 0:
            changed['India_changed_vs_B'] += parsed(expected[rid]) != parsed(prediction)
            expected[rid] = prediction
    assert len(seen) == len(countries)
    seen.clear()
    by_country = Counter()
    for rid, prediction in rows(h, 'matched_entity_ids'):
        assert rid in expected and rid not in seen
        seen.add(rid)
        assert prediction == expected[rid], 'Wrong source policy for ' + rid
        by_country[countries[rid]] += 1
    assert len(seen) == len(countries)
    seen.clear()
    print('PASS: original hashes, complete unique coverage and exact country-policy mapping.', flush=True)
    targets = {row[0] for row in conn.execute('SELECT entity_id FROM records WHERE source IN (2,3)')}
    assert len(targets) == 9969589
    candidate_digests = {}
    pairs = 0
    for rid, candidates in rows(ca, 'candidate_entity_ids'):
        assert rid in countries and rid not in candidate_digests
        ids = parsed(candidates)
        assert ids <= targets, 'Nonexistent candidate'
        assert parsed(expected[rid]) <= ids, 'Hybrid match missing from candidates'
        candidate_digests[rid] = hashlib.sha256(candidates.encode()).digest()
        pairs += len(ids)
    assert len(candidate_digests) == len(countries)
    del targets
    for rid, candidates in rows(cb, 'candidate_entity_ids'):
        assert rid in candidate_digests, 'Extra or duplicate candidate reference'
        assert candidate_digests.pop(rid) == hashlib.sha256(candidates.encode()).digest(), 'Candidate list differs'
    assert not candidate_digests
    print('PASS: all target IDs, candidate uniqueness, exact candidate-list identity and hybrid subsets.', flush=True)
    report = {
        'passed': True, 'references': len(countries), 'candidate_pairs': pairs,
        'country_counts': {c: by_country[i] for c, i in codes.items()},
        'policy': {'India': {'model': 'A', 'threshold': .362}, 'US': {'model': 'B', 'threshold': .665}, 'France': {'model': 'B', 'threshold': .665}},
        'changes_vs_current_0787': dict(changed), 'files_sha256': hashes,
        'checks': ['Exact known source and hybrid hashes', 'Headers and exactly two TSV columns', 'Complete unique reference coverage against verified test DB', 'No duplicate targets', 'All 9,969,589 target IDs sourced from verified test DB; every candidate checked', 'Hybrid equals A for India and B for US/France', 'All hybrid matches are subsets of original candidates', 'Original and rescored candidate lists identical per reference despite row-order differences'],
        'limitations': 'Structural and source-policy verification, not test accuracy. Country selection used inspected DEV evidence; no fresh holdout result was supplied.',
        'elapsed_seconds': time.monotonic() - start,
    }
    output = ROOT / 'reports/hybrid_submission/independent_validation.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
