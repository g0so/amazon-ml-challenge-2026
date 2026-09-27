"""Audit training identities with SQLite, keeping memory bounded.

Run from any directory with the project Python. Inputs are read-only. Each run
builds a fresh temporary database, so rerunning never appends duplicate rows.
Only a completed run replaces the previous database/report.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from business_entity_resolution.workspace import DATASET_DIR, iter_tsv

BATCH = 10_000


def parse_targets(text):
    """Preserve duplicate links for auditing; reject malformed ID tokens."""
    ids = [value.strip() for value in text.split(',')] if text.strip() else []
    if any(not re.fullmatch(r'S[23]-[^,\s]+', value) for value in ids):
        raise ValueError(f'Malformed target list: {text!r}')
    return ids


def insert_batches(db, query, rows, label):
    batch = []
    count = 0
    for row in rows:
        batch.append(row)
        if len(batch) == BATCH:
            db.executemany(query, batch)
            db.commit()
            count += len(batch)
            batch.clear()
            if count % 1_000_000 == 0:
                print(f'{label}: {count:,} rows', flush=True)
    if batch:
        db.executemany(query, batch)
        db.commit()
        count += len(batch)
    return count


def file_hash(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def audit(train_dir, database):
    started = time.monotonic()
    paths = [train_dir / 'train_ground_truth.tsv'] + [train_dir / f'train_source{s}.tsv' for s in (1, 2, 3)]
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_name(database.name + '.' + uuid.uuid4().hex + '.building')
    db = sqlite3.connect(temporary)
    db.execute('PRAGMA cache_size=-32768')  # About 32 MiB SQLite page cache.
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA synchronous=NORMAL')
    db.executescript('''
        CREATE TABLE reference_rows (reference_id TEXT, match_count INTEGER);
        CREATE TABLE truth_links (reference_id TEXT, target_id TEXT);
        CREATE TABLE source_records (entity_id TEXT, source INTEGER, country TEXT);
    ''')
    try:
        print('Loading ground truth (singletons retained separately)...', flush=True)
        references = []
        links = []
        counts = Counter()
        duplicate_examples = []
        for row in iter_tsv(paths[0]):
            ref = row['source1_entity_id']
            if not re.fullmatch(r'S1-[^,\s]+', ref):
                raise ValueError(f'Malformed reference ID: {ref!r}')
            ids = parse_targets(row['matched_entity_ids'])
            counts['reference_rows'] += 1
            counts['positive_link_rows'] += len(ids)
            counts['singleton_rows'] += not ids
            if len(ids) != len(set(ids)):
                counts['truth_rows_with_duplicate_targets'] += 1
                if len(duplicate_examples) < 5:
                    duplicate_examples.append(ref)
            references.append((ref, len(ids)))
            links.extend((ref, target) for target in ids)
            if len(references) >= BATCH or len(links) >= BATCH:
                db.executemany('INSERT INTO reference_rows VALUES (?, ?)', references)
                db.executemany('INSERT INTO truth_links VALUES (?, ?)', links)
                db.commit()
                references.clear()
                links.clear()
        db.executemany('INSERT INTO reference_rows VALUES (?, ?)', references)
        db.executemany('INSERT INTO truth_links VALUES (?, ?)', links)
        db.commit()
        print(f'Truth: {counts["reference_rows"]:,} references; {counts["positive_link_rows"]:,} links', flush=True)

        def source_rows(path, source):
            for row in iter_tsv(path):
                if not re.fullmatch(fr'S{source}-[^,\s]+', row['entity_id']):
                    raise ValueError(f'{path.name}: malformed ID {row["entity_id"]!r}')
                yield row['entity_id'], source, row['country']

        source_counts = {}
        for source, path in enumerate(paths[1:], 1):
            print(f'Loading {path.name} (IDs and country only)...', flush=True)
            source_counts[f'S{source}'] = insert_batches(db, 'INSERT INTO source_records VALUES (?, ?, ?)',
                                                       source_rows(path, source), path.name)

        print('Building disk indexes and counting target owners...', flush=True)
        db.executescript('''
            CREATE INDEX reference_lookup ON reference_rows(reference_id);
            CREATE INDEX target_lookup ON truth_links(target_id, reference_id);
            CREATE INDEX source_lookup ON source_records(entity_id);
            CREATE TABLE target_owners AS
              SELECT target_id, COUNT(DISTINCT reference_id) AS reference_count
              FROM truth_links GROUP BY target_id;
            CREATE UNIQUE INDEX owner_lookup ON target_owners(target_id);
        ''')
        checks = {
            'duplicate_reference_ids': '''SELECT reference_id, COUNT(*) AS count FROM reference_rows
                GROUP BY reference_id HAVING COUNT(*) > 1''',
            'duplicate_source_ids': '''SELECT entity_id, COUNT(*) AS count FROM source_records
                GROUP BY entity_id HAVING COUNT(*) > 1''',
            'targets_with_multiple_references': '''SELECT target_id, reference_count FROM target_owners
                WHERE reference_count > 1''',
            'truth_references_missing_from_source1': '''SELECT DISTINCT r.reference_id FROM reference_rows r
                WHERE NOT EXISTS (SELECT 1 FROM source_records s
                                  WHERE s.entity_id=r.reference_id AND s.source=1)''',
            'source1_records_missing_truth': '''SELECT DISTINCT s.entity_id FROM source_records s
                WHERE s.source=1 AND NOT EXISTS
                    (SELECT 1 FROM reference_rows r WHERE r.reference_id=s.entity_id)''',
            'truth_targets_missing_from_sources': '''SELECT t.target_id FROM target_owners t
                WHERE NOT EXISTS (SELECT 1 FROM source_records s
                                  WHERE s.entity_id=t.target_id AND s.source IN (2,3))''',
        }
        findings = {}
        for label, query in checks.items():
            print(f'Checking {label}...', flush=True)
            number = db.execute(f'SELECT COUNT(*) FROM ({query})').fetchone()[0]
            columns = [c[0] for c in db.execute(query + ' LIMIT 0').description]
            examples = [dict(zip(columns, row)) for row in db.execute(query + ' LIMIT 5')]
            findings[label] = {'count': number, 'examples': examples}
            print(f'  {number:,}', flush=True)
        overlap_examples = []
        for row in findings['targets_with_multiple_references']['examples']:
            owners = [r[0] for r in db.execute('SELECT DISTINCT reference_id FROM truth_links WHERE target_id=? LIMIT 10',
                                              (row['target_id'],))]
            overlap_examples.append({'target_id': row['target_id'], 'references': owners})
        distinct_targets = db.execute('SELECT COUNT(*) FROM target_owners').fetchone()[0]
        distractors = db.execute('''SELECT COUNT(*) FROM source_records s WHERE source IN (2,3)
            AND NOT EXISTS (SELECT 1 FROM target_owners t WHERE t.target_id=s.entity_id)''').fetchone()[0]
        print('Recording input checksums...', flush=True)
        inputs = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in paths]
        if any(before[p.name] != (p.stat().st_size, p.stat().st_mtime_ns) for p in paths):
            raise RuntimeError('Input changed during audit; results not published.')
        errors = [key for key, value in findings.items()
                  if key != 'targets_with_multiple_references' and value['count']]
        if counts['truth_rows_with_duplicate_targets']:
            errors.append('truth_rows_with_duplicate_targets')
        report = {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Full training ground truth and all three training source ID columns',
            'reference_rows': counts['reference_rows'], 'positive_link_rows': counts['positive_link_rows'],
            'singleton_rows': counts['singleton_rows'], 'source_rows': source_counts,
            'truth_rows_with_duplicate_targets': counts['truth_rows_with_duplicate_targets'],
            'duplicate_target_list_examples': duplicate_examples,
            'distinct_labeled_targets': distinct_targets,
            'unmatched_target_record_rows': distractors,
            'checks': findings, 'shared_target_examples': overlap_examples,
            'structural_integrity_passed': not errors, 'blocking_integrity_issues': errors,
            'grouping_requires_shared_target_components': findings['targets_with_multiple_references']['count'] > 0,
            'inputs': inputs, 'elapsed_seconds': round(time.monotonic() - started, 2),
            'limitations': ['Does not establish semantic identity correctness or completeness of labels.',
                            'Distinct IDs can still have identical or related business text; content leakage is not checked.',
                            'No split manifest, normalization, matcher, or model was created.'],
        }
        db.execute('CREATE TABLE audit_metadata (report_json TEXT)')
        db.execute('INSERT INTO audit_metadata VALUES (?)', (json.dumps(report),))
        db.commit()
        db.close()
        os.replace(temporary, database)
        return report
    except BaseException:
        db.close()
        # Only this run's generated temporary database is removed; raw inputs and
        # any completed prior database are untouched.
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + '-journal').unlink(missing_ok=True)
        raise


def write_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'phase2_relationship_audit.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)
    lines = ['# Phase 2 relationship audit', '', report['created_at_utc'], '',
             '**Scope:** ' + report['scope'], '', '| Measure | Result |', '|---|---:|']
    for key in ['reference_rows', 'positive_link_rows', 'singleton_rows', 'distinct_labeled_targets',
                'unmatched_target_record_rows', 'truth_rows_with_duplicate_targets']:
        lines.append(f'| {key} | {report[key]:,} |')
    for key, value in report['checks'].items():
        lines.append(f'| {key} | {value["count"]:,} |')
    lines += ['', '## Interpretation', '']
    if not report['structural_integrity_passed']:
        lines += ['Resolve the listed integrity issues before creating a split.']
    elif report['grouping_requires_shared_target_components']:
        lines += ['Build connected components for references that share targets, then split complete components.']
    else:
        lines += ['Each labeled reference and its target variants forms a separate labeled group: no targets are shared across references.',
                  'A connected-component algorithm is unnecessary for the supplied labeled edges. Preserve each complete reference group when splitting.']
    lines += ['', 'Unmatched target rows are potential distractors, not records to discard or attach to invented references.',
              '', '## Next step', '', 'Create and verify a saved group split manifest. It has not been created by this audit.',
              '', '## Limits', ''] + [f'- {x}' for x in report['limitations']]
    lines += ['', f'Elapsed: {report["elapsed_seconds"]:.2f} seconds.',
              'The JSON report and SQLite audit_metadata table include input SHA-256 checksums.', '']
    path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=DATASET_DIR / 'train')
    parser.add_argument('--database', type=Path, default=ROOT / 'data/processed/training_relationships.sqlite')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    report = audit(args.train_dir, args.database)
    write_report(report, args.report_dir)
    print(f'Complete: {args.report_dir / "phase2_relationship_audit.md"}', flush=True)
    if report['blocking_integrity_issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
