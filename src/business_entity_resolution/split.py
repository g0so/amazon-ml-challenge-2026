"""Verified disk-backed partitions; import and preserve an existing JSON split.

Labeled targets inherit their reference's partition. Unmatched targets have
explicit assignments. Ground-truth labels are used for splitting/evaluation,
never to remove false positives from baseline predictions.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import uuid

from .workspace import CONFIG, DATASET_DIR, ROOT

DATABASE_PATH = ROOT / 'data/processed/training_relationships.sqlite'
LEGACY_PATH = ROOT / 'data/processed/split_manifest.json'
SPLIT_DB = ROOT / 'data/processed/phase2_split.sqlite'
MANIFEST_PATH = ROOT / 'reports/phase2_split_summary.json'
SEED = CONFIG['seed']
SPLIT_PROPORTIONS = {'train': .70, 'dev': .15, 'holdout': .15}
BATCH = 10_000


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bucket(count):
    return '0' if count == 0 else '1' if count == 1 else '2' if count == 2 else '3-4' if count <= 4 else '5-7' if count <= 7 else '8+'


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def stratified_split(groups, seed, proportions):
    """Preserve the original algorithm, with audited input ordering and validation."""
    if set(proportions) != set(SPLIT_PROPORTIONS) or abs(sum(proportions.values()) - 1) > 1e-12:
        raise ValueError('Expected train/dev/holdout proportions summing to one')
    if any(x < 0 for x in proportions.values()):
        raise ValueError('Negative split proportion')
    strata = defaultdict(list)
    for g in groups:
        strata[g['country'], bucket(g['match_count'])].append(g['reference_id'])
    rng = random.Random(seed)
    assignments = {}
    for refs in strata.values():
        rng.shuffle(refs)
        a = int(len(refs) * proportions['train'])
        b = a + int(len(refs) * proportions['dev'])
        for i, ref in enumerate(refs):
            if ref in assignments:
                raise ValueError(f'Duplicate reference: {ref}')
            assignments[ref] = 'train' if i < a else 'dev' if i < b else 'holdout'
    return assignments


def reference_rows(conn):
    return conn.execute('''SELECT r.reference_id, r.match_count, s.country FROM reference_rows r
        JOIN source_records s ON s.entity_id=r.reference_id AND s.source=1 ORDER BY r.rowid''')


def distractor_rows(conn):
    return conn.execute('''SELECT s.entity_id, s.source, s.country FROM source_records s
        WHERE s.source IN (2,3) AND NOT EXISTS
        (SELECT 1 FROM target_owners t WHERE t.target_id=s.entity_id) ORDER BY s.rowid''')


def input_hashes(train_dir):
    print('Verifying training file checksums...', flush=True)
    return {name: file_hash(Path(train_dir) / name) for name in
            ['train_ground_truth.tsv', 'train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']}


def create_manifest(audit_path=DATABASE_PATH, legacy_path=LEGACY_PATH, output_path=SPLIT_DB,
                    summary_path=MANIFEST_PATH, train_dir=DATASET_DIR / 'train'):
    """Import legacy assignments or reproduce the original seeded split recipe."""
    output_path, summary_path, legacy_path = map(Path, (output_path, summary_path, legacy_path))
    hashes = input_hashes(train_dir)
    audit = readonly(audit_path)
    audit_report = json.loads(audit.execute('SELECT report_json FROM audit_metadata').fetchone()[0])
    expected_hashes = {x['name']: x['sha256'] for x in audit_report['inputs']}
    if hashes != expected_hashes:
        audit.close()
        raise ValueError('Training files changed since the relationship audit; rebuild that audit first')
    if not audit_report['structural_integrity_passed'] or audit_report['grouping_requires_shared_target_components']:
        audit.close()
        raise ValueError('This splitter requires clean labels and no shared targets')
    if output_path.exists():
        with readonly(output_path) as cached:
            metadata = json.loads(cached.execute('SELECT report_json FROM split_metadata').fetchone()[0])
            if metadata['input_hashes'] != hashes:
                raise ValueError('Saved split belongs to different inputs')
            if legacy_path.exists() and metadata['provenance'].get('legacy_sha256', file_hash(legacy_path)) != file_hash(legacy_path):
                raise ValueError('Legacy assignments changed after import; inspect before rebuilding')
            verify_manifest(cached, audit, audit_report)
        audit.close()
        print('Reusing independently verified saved assignments.', flush=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata

    if legacy_path.exists():
        print('Importing your existing split; assignments will not change.', flush=True)
        legacy = json.loads(legacy_path.read_text())
        if legacy['input_hashes'] != hashes:
            raise ValueError('Existing JSON split hashes do not match the supplied training data')
        ref_assign = legacy['reference_assignments']
        dist_assign = legacy['distractor_assignments']
        seed = legacy['seed']
        provenance = {'mode': 'preserved_existing_json', 'legacy_sha256': file_hash(legacy_path)}
    else:
        print('No legacy JSON found; reproducing the seeded reference/distractor recipe.', flush=True)
        seed = SEED
        ref_assign = stratified_split(({'reference_id': r, 'match_count': m, 'country': c}
                                       for r, m, c in reference_rows(audit)), seed, SPLIT_PROPORTIONS)
        country_counts = defaultdict(Counter)
        for ref, mc, country in reference_rows(audit):
            country_counts[country][ref_assign[ref]] += 1
        rng = random.Random(seed + 1)
        dist_assign = {}
        for target, source, country in distractor_rows(audit):
            counts = country_counts[country]
            total = sum(counts.values())
            train = counts['train'] / total if total else .7
            dev = counts['dev'] / total if total else .15
            draw = rng.random()
            dist_assign[target] = 'train' if draw <= train else 'dev' if draw <= train + dev else 'holdout'
        provenance = {'mode': 'seeded_original_recipe', 'seed': seed}
    if len(ref_assign) != audit_report['reference_rows'] or len(dist_assign) != audit_report['unmatched_target_record_rows']:
        raise ValueError('Assignment counts do not match independently audited counts')
    if any(s not in SPLIT_PROPORTIONS for s in ref_assign.values()) or any(s not in SPLIT_PROPORTIONS for s in dist_assign.values()):
        raise ValueError('Invalid split name')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + '.' + uuid.uuid4().hex + '.building')
    db = sqlite3.connect(temp)
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA temp_store=FILE')
    db.executescript('''
        CREATE TABLE reference_assignments (
            reference_id TEXT PRIMARY KEY, split TEXT NOT NULL, country TEXT NOT NULL,
            match_count INTEGER NOT NULL, bucket TEXT NOT NULL);
        CREATE TABLE target_assignments (
            target_id TEXT PRIMARY KEY, split TEXT NOT NULL, reference_id TEXT, source INTEGER NOT NULL);
    ''')
    try:
        logical = hashlib.sha256()
        def insert(query, rows):
            batch = []
            for row in rows:
                logical.update((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                batch.append(row)
                if len(batch) == BATCH:
                    db.executemany(query, batch); db.commit(); batch.clear()
            if batch:
                db.executemany(query, batch); db.commit()
        def references():
            for ref, mc, country in reference_rows(audit):
                if ref not in ref_assign:
                    raise ValueError(f'Unassigned reference: {ref}')
                yield ref, ref_assign[ref], country, mc, bucket(mc)
        print('Writing reference assignments...', flush=True)
        insert('INSERT INTO reference_assignments VALUES (?,?,?,?,?)', references())
        print('Assigning every labeled target to its reference partition...', flush=True)
        def labeled():
            for ref, target in audit.execute('SELECT reference_id, target_id FROM truth_links ORDER BY target_id'):
                yield target, ref_assign[ref], ref, int(target[1])
        insert('INSERT INTO target_assignments VALUES (?,?,?,?)', labeled())
        print('Writing and validating distractor assignments...', flush=True)
        def distractors():
            for target, source, country in distractor_rows(audit):
                if target not in dist_assign:
                    raise ValueError(f'Unassigned distractor: {target}')
                yield target, dist_assign[target], None, source
        insert('INSERT INTO target_assignments VALUES (?,?,?,?)', distractors())
        # Release the large legacy JSON objects before SQL verification.
        del ref_assign, dist_assign
        if 'legacy' in locals():
            del legacy
        db.executescript('''CREATE INDEX ref_split ON reference_assignments(split, reference_id);
            CREATE INDEX target_split ON target_assignments(split, target_id);''')
        verify_manifest(db, audit, audit_report)
        strata = [dict(zip(['country', 'bucket', 'split', 'references'], r)) for r in db.execute(
            'SELECT country,bucket,split,COUNT(*) FROM reference_assignments GROUP BY country,bucket,split')]
        metadata = {
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'format_version': 2,
            'manifest_id': logical.hexdigest(), 'seed': seed, 'proportions': SPLIT_PROPORTIONS,
            'provenance': provenance, 'input_hashes': hashes, 'verified': True,
            'reference_counts': dict(db.execute('SELECT split,COUNT(*) FROM reference_assignments GROUP BY split')),
            'target_counts': dict(db.execute('SELECT split,COUNT(*) FROM target_assignments GROUP BY split')),
            'distractor_counts': dict(db.execute('SELECT split,COUNT(*) FROM target_assignments WHERE reference_id IS NULL GROUP BY split')),
            'strata': strata,
            'checks': ['All audited reference and target IDs assigned exactly once',
                       'Every labeled target inherits its true reference partition',
                       'No missing, additional, or mislabeled distractor assignments',
                       'Allowed partition names and reference country/match-count strata verified'],
            'limitations': ['This prevents labeled-identity overlap; text aliases and related real businesses can still span partitions.',
                            'Holdout assignments were checked structurally, but no holdout predictions were scored.'],
        }
        db.execute('CREATE TABLE split_metadata (report_json TEXT)')
        db.execute('INSERT INTO split_metadata VALUES (?)', (json.dumps(metadata),))
        db.commit(); db.close(); audit.close()
        temp.replace(output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata
    except BaseException:
        db.close(); audit.close()
        temp.unlink(missing_ok=True)
        Path(str(temp) + '-journal').unlink(missing_ok=True)
        raise


def verify_manifest(db, audit, audit_report):
    """Verify against independently audited IDs, not counts copied from a manifest."""
    # The caller's audit DB path is obtained from the open connection (fixtures supported).
    audit_file = next(row[2] for row in audit.execute('PRAGMA database_list') if row[1] == 'main')
    db.execute('ATTACH DATABASE ? AS audit_check', (Path(audit_file).as_uri() + '?mode=ro',))
    try:
        tests = {
            'reference coverage': '''SELECT COUNT(*) FROM audit_check.reference_rows a
                LEFT JOIN reference_assignments r ON r.reference_id=a.reference_id WHERE r.reference_id IS NULL''',
            'target coverage': '''SELECT COUNT(*) FROM audit_check.source_records s
                LEFT JOIN target_assignments t ON t.target_id=s.entity_id WHERE s.source IN (2,3) AND t.target_id IS NULL''',
            'labeled target ownership/partition': '''SELECT COUNT(*) FROM target_assignments t
                LEFT JOIN reference_assignments r ON r.reference_id=t.reference_id
                WHERE t.reference_id IS NOT NULL AND (r.reference_id IS NULL OR t.split<>r.split)''',
            'truth owner correctness': '''SELECT COUNT(*) FROM audit_check.truth_links l
                JOIN target_assignments t ON t.target_id=l.target_id
                WHERE t.reference_id IS NULL OR t.reference_id<>l.reference_id''',
            'reference metadata': '''SELECT COUNT(*) FROM reference_assignments r
                JOIN audit_check.reference_rows a ON a.reference_id=r.reference_id
                JOIN audit_check.source_records s ON s.entity_id=r.reference_id
                WHERE r.match_count<>a.match_count OR r.country<>s.country''',
        }
        for label, query in tests.items():
            print('Verifying ' + label + '...', flush=True)
            if db.execute(query).fetchone()[0]:
                raise ValueError('Split failed: ' + label)
        expected = audit_report['source_rows']['S2'] + audit_report['source_rows']['S3']
        if db.execute('SELECT COUNT(*) FROM target_assignments').fetchone()[0] != expected:
            raise ValueError('Extra/missing target assignments')
        if db.execute('SELECT COUNT(*) FROM reference_assignments').fetchone()[0] != audit_report['reference_rows']:
            raise ValueError('Extra/missing reference assignments')
        if db.execute('SELECT COUNT(*) FROM target_assignments WHERE reference_id IS NULL').fetchone()[0] != audit_report['unmatched_target_record_rows']:
            raise ValueError('Distractor classification mismatch')
        for table in ['reference_assignments', 'target_assignments']:
            if db.execute(f"SELECT COUNT(*) FROM {table} WHERE split NOT IN ('train','dev','holdout')").fetchone()[0]:
                raise ValueError('Unknown split name')
        strata = defaultdict(Counter)
        for country, b, split, count in db.execute('SELECT country,bucket,split,COUNT(*) FROM reference_assignments GROUP BY country,bucket,split'):
            strata[country, b][split] = count
        for key, counts in strata.items():
            n = sum(counts.values())
            for split, proportion in SPLIT_PROPORTIONS.items():
                if abs(counts[split] - n * proportion) > max(2, .02 * n):
                    raise ValueError(f'Poor stratification in {key}: {counts}')
    finally:
        db.execute('DETACH DATABASE audit_check')


def load_manifest():
    with readonly(SPLIT_DB) as conn:
        return json.loads(conn.execute('SELECT report_json FROM split_metadata').fetchone()[0])


def get_split_references(split_name, manifest=None):
    if split_name not in SPLIT_PROPORTIONS:
        raise ValueError('Unknown partition')
    with readonly(SPLIT_DB) as conn:
        return [r[0] for r in conn.execute('SELECT reference_id FROM reference_assignments WHERE split=? ORDER BY reference_id', (split_name,))]


def get_split_distractors(split_name, manifest=None):
    if split_name not in SPLIT_PROPORTIONS:
        raise ValueError('Unknown partition')
    with readonly(SPLIT_DB) as conn:
        return [r[0] for r in conn.execute('SELECT target_id FROM target_assignments WHERE split=? AND reference_id IS NULL ORDER BY target_id', (split_name,))]
