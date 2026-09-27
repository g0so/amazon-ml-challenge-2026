"""Honest Phase 2 development baseline: raw and normalized exact matching.

Predictions depend only on country, name, and address in the full dev target pool,
including distractors. Ground truth is attached only AFTER predictions are made.
The holdout is never scored. SQLite keeps the text index and predictions on disk.
"""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .workspace import DATASET_DIR, ROOT, iter_tsv
from .normalize import normalize_text, NORMALIZATION_VERSION
from .split import SPLIT_DB, DATABASE_PATH, readonly

BASELINE_DB = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
REPORT_PATH = ROOT / 'reports/phase2_baseline.json'
RULES = ('raw_exact', 'normalized_exact')
BATCH = 5_000


def create_record_table(db):
    db.executescript('''CREATE TABLE records (
        entity_id TEXT PRIMARY KEY, source INTEGER NOT NULL, country TEXT NOT NULL,
        name_raw TEXT NOT NULL, address_raw TEXT NOT NULL, country_norm TEXT NOT NULL,
        name_norm TEXT NOT NULL, address_norm TEXT NOT NULL, eligible INTEGER NOT NULL);''')


def record_values(row, source):
    country, name, address = row['country'], row['business_name'], row['business_address']
    cn, nn, an = map(normalize_text, (country, name, address))
    return row['entity_id'], source, country, name, address, cn, nn, an, int(bool(cn and nn and an))


def generate_predictions(db):
    """No label table or truth mapping is accepted or read by this function."""
    db.executescript('''
        CREATE INDEX target_raw_key ON records(country, name_raw, address_raw)
            WHERE source IN (2,3) AND eligible=1;
        CREATE INDEX target_normalized_key ON records(country_norm, name_norm, address_norm)
            WHERE source IN (2,3) AND eligible=1;
        CREATE TABLE predictions (
            rule TEXT, reference_id TEXT, target_id TEXT,
            PRIMARY KEY(rule, reference_id, target_id));
    ''')
    for rule, fields in [('raw_exact', ('country', 'name_raw', 'address_raw')),
                         ('normalized_exact', ('country_norm', 'name_norm', 'address_norm'))]:
        print('Generating label-blind ' + rule + ' predictions...', flush=True)
        conditions = ' AND '.join(f'r.{field}=t.{field}' for field in fields)
        db.execute(f'''INSERT INTO predictions SELECT ?, r.entity_id, t.entity_id
            FROM records r JOIN records t ON {conditions}
            WHERE r.source=1 AND r.eligible=1 AND t.source IN (2,3) AND t.eligible=1''', (rule,))
        db.commit()


def evaluate_predictions(db):
    """Truth is used here, after candidate/prediction generation has finished."""
    db.executescript('''
        CREATE TABLE truth_counts AS SELECT reference_id, COUNT(*) AS n FROM truth_edges GROUP BY reference_id;
        CREATE UNIQUE INDEX truth_count_lookup ON truth_counts(reference_id);
        CREATE TABLE prediction_counts AS
            SELECT p.rule, p.reference_id, COUNT(*) AS n,
                   SUM(CASE WHEN t.target_id IS NOT NULL THEN 1 ELSE 0 END) AS tp
            FROM predictions p LEFT JOIN truth_edges t
                ON t.reference_id=p.reference_id AND t.target_id=p.target_id
            GROUP BY p.rule, p.reference_id;
        CREATE UNIQUE INDEX prediction_count_lookup ON prediction_counts(rule,reference_id);
        CREATE TABLE evaluation (
            rule TEXT, reference_id TEXT, country TEXT, true_count INTEGER,
            predicted_count INTEGER, tp INTEGER, fp INTEGER, fn INTEGER, score REAL,
            PRIMARY KEY(rule, reference_id));
    ''')
    for rule in RULES:
        db.execute('''INSERT INTO evaluation
            SELECT ?, r.entity_id, r.country, COALESCE(t.n,0), COALESCE(p.n,0),
                   COALESCE(p.tp,0), COALESCE(p.n,0)-COALESCE(p.tp,0),
                   COALESCE(t.n,0)-COALESCE(p.tp,0),
                   CASE WHEN COALESCE(p.n,0)+COALESCE(t.n,0)=0 THEN 1.0
                        ELSE 1.25*COALESCE(p.tp,0)/(COALESCE(p.n,0)+0.25*COALESCE(t.n,0)) END
            FROM records r LEFT JOIN truth_counts t ON t.reference_id=r.entity_id
            LEFT JOIN prediction_counts p ON p.rule=? AND p.reference_id=r.entity_id
            WHERE r.source=1''', (rule, rule))
    db.commit()
    return {rule: summarize(db, rule) for rule in RULES}


def summarize(db, rule, country=None):
    where, args = 'rule=?', [rule]
    if country is not None:
        where += ' AND country=?'; args.append(country)
    row = db.execute(f'''SELECT COUNT(*), AVG(score), SUM(tp), SUM(fp), SUM(fn),
        SUM(CASE WHEN true_count=0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN true_count=0 AND predicted_count>0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN predicted_count>0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN fp=0 AND fn=0 THEN 1 ELSE 0 END),
        AVG(CASE WHEN true_count=0 THEN 1.0 ELSE 1.25*tp/(tp+0.25*true_count) END)
        FROM evaluation WHERE {where}''', args).fetchone()
    n, macro, tp, fp, fn, singles, false_singles, predicted_refs, perfect, oracle = row
    if not n:
        raise ValueError('Empty evaluation partition')
    return {'references': n, 'macro_f05': macro, 'tp': tp, 'fp': fp, 'fn': fn,
            'micro_precision': tp/(tp+fp) if tp+fp else None,
            'micro_recall': tp/(tp+fn) if tp+fn else None,
            'singletons': singles, 'singleton_false_matches': false_singles,
            'singleton_false_positive_rate': false_singles/singles if singles else None,
            'references_with_predictions': predicted_refs, 'exact_set_accuracy': perfect/n,
            'empty_baseline': singles/n, 'candidate_oracle_macro_f05': oracle}


def error_examples(db, rule, limit=10):
    fields = '''r.entity_id AS reference_id, r.country, r.name_raw AS reference_name,
        r.address_raw AS reference_address, t.entity_id AS target_id,
        t.name_raw AS target_name, t.address_raw AS target_address'''
    queries = {
        'false_positive': f'''SELECT {fields} FROM predictions p
            LEFT JOIN truth_edges truth ON truth.reference_id=p.reference_id AND truth.target_id=p.target_id
            JOIN records r ON r.entity_id=p.reference_id JOIN records t ON t.entity_id=p.target_id
            WHERE p.rule=? AND truth.target_id IS NULL ORDER BY p.reference_id,p.target_id LIMIT ?''',
        'false_negative': f'''SELECT {fields} FROM truth_edges truth
            LEFT JOIN predictions p ON p.rule=? AND p.reference_id=truth.reference_id AND p.target_id=truth.target_id
            JOIN records r ON r.entity_id=truth.reference_id JOIN records t ON t.entity_id=truth.target_id
            WHERE p.target_id IS NULL ORDER BY truth.reference_id,truth.target_id LIMIT ?''',
    }
    result = {}
    for kind, query in queries.items():
        cur = db.execute(query, (rule, limit))
        names = [c[0] for c in cur.description]
        result[kind] = [dict(zip(names, row)) for row in cur]
    return result


def export_predictions(db, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for rule in RULES:
        path = output_dir / f'dev_{rule}_predictions.tsv'
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['source1_entity_id', 'matched_entity_ids'])
            previous, targets = None, []
            for ref, target in db.execute('''SELECT r.entity_id,p.target_id FROM records r
                LEFT JOIN predictions p ON p.reference_id=r.entity_id AND p.rule=?
                WHERE r.source=1 ORDER BY r.entity_id,p.target_id''', (rule,)):
                if ref != previous:
                    if previous is not None:
                        writer.writerow([previous, ','.join(targets)])
                    previous, targets = ref, []
                if target is not None:
                    targets.append(target)
            if previous is not None:
                writer.writerow([previous, ','.join(targets)])


def run_baseline(split_path=SPLIT_DB, audit_path=DATABASE_PATH, train_dir=DATASET_DIR/'train',
                 database_path=BASELINE_DB, report_path=REPORT_PATH,
                 predictions_dir=ROOT/'data/processed/phase2'):
    started = time.monotonic()
    split = readonly(split_path)
    metadata = json.loads(split.execute('SELECT report_json FROM split_metadata').fetchone()[0])
    ref_ids = {r[0] for r in split.execute("SELECT reference_id FROM reference_assignments WHERE split='dev'")}
    target_ids = {r[0] for r in split.execute("SELECT target_id FROM target_assignments WHERE split='dev'")}
    if not ref_ids:
        raise ValueError('Development partition is empty')
    dbpath, report_path = Path(database_path), Path(report_path)
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    temp = dbpath.with_name(dbpath.name + '.' + uuid.uuid4().hex + '.building')
    db = sqlite3.connect(temp)
    db.execute('PRAGMA cache_size=-65536'); db.execute('PRAGMA temp_store=FILE')
    create_record_table(db)
    counts = defaultdict(int)
    try:
        print(f'Full dev pool: {len(ref_ids):,} references and {len(target_ids):,} targets (including distractors).', flush=True)
        for source in (1,2,3):
            print(f'Streaming training source {source}; storing only dev records...', flush=True)
            wanted = ref_ids if source == 1 else target_ids
            batch = []
            for row in iter_tsv(Path(train_dir)/f'train_source{source}.tsv'):
                if row['entity_id'] in wanted:
                    batch.append(record_values(row, source)); counts[source] += 1
                    if len(batch) == BATCH:
                        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)', batch)
                        db.commit(); batch.clear()
            if batch:
                db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)', batch); db.commit()
        if counts[1] != len(ref_ids) or counts[2]+counts[3] != len(target_ids):
            raise ValueError('Development data coverage incomplete')
        del ref_ids, target_ids
        # Inference happens before any truth table is created in this database.
        generate_predictions(db)
        print('Predictions frozen. Attaching development labels for evaluation only...', flush=True)
        db.execute('ATTACH DATABASE ? AS audit', (Path(audit_path).resolve().as_uri()+'?mode=ro',))
        db.execute('CREATE TABLE truth_edges (reference_id TEXT, target_id TEXT, PRIMARY KEY(reference_id,target_id))')
        db.execute('''INSERT INTO truth_edges SELECT l.reference_id,l.target_id FROM audit.truth_links l
            JOIN records r ON r.entity_id=l.reference_id AND r.source=1''')
        db.commit()
        missing = db.execute('''SELECT COUNT(*) FROM truth_edges e LEFT JOIN records t ON t.entity_id=e.target_id
            WHERE t.entity_id IS NULL OR t.source=1''').fetchone()[0]
        expected_links = split.execute("SELECT SUM(match_count) FROM reference_assignments WHERE split='dev'").fetchone()[0]
        if missing or db.execute('SELECT COUNT(*) FROM truth_edges').fetchone()[0] != expected_links:
            raise ValueError('True target variants lost from development partition')
        db.execute('DETACH DATABASE audit')
        metrics = evaluate_predictions(db)
        countries = [r[0] for r in db.execute('SELECT DISTINCT country FROM records WHERE source=1 ORDER BY country')]
        for rule in RULES:
            metrics[rule]['by_country'] = {c: summarize(db, rule, c) for c in countries}
        errors = {rule: error_examples(db, rule) for rule in RULES}
        print('Exporting complete development prediction files...', flush=True)
        export_predictions(db, Path(predictions_dir))
        report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'evaluation_partition': 'dev', 'holdout_scored': False,
                  'manifest_id': metadata['manifest_id'], 'normalization_version': NORMALIZATION_VERSION,
                  'input_hashes': metadata['input_hashes'],
                  'reference_count': counts[1], 'target_count': counts[2]+counts[3],
                  'distractor_count': metadata['distractor_counts']['dev'],
                  'target_sources': {'S2': counts[2], 'S3': counts[3]},
                  'blank_target_addresses': db.execute('SELECT COUNT(*) FROM records WHERE source IN (2,3) AND address_norm=\'\'').fetchone()[0],
                  'metrics': metrics, 'elapsed_seconds': round(time.monotonic()-started,2),
                  'validation_checks': ['Predictions generated before attaching labels',
                      'Entire dev target pool includes unmatched distractors and other references\' variants',
                      'Every dev reference loaded and scored, including singletons',
                      'Every dev true target remains in the development pool',
                      'Country included in both match keys; blank name/address/country keys rejected'],
                  'limitations': ['Development-only result, not a leaderboard or France-performance estimate.',
                      'Exact keys have a low recall ceiling; missing/typo/reordered/transliterated fields are often missed.',
                      'Unrelated IDs with related business text may still cross partitions despite labeled-ID isolation.',
                      'Text normalization is intentionally minimal; no model, tuning, or test predictions yet.']}
        db.execute('CREATE TABLE baseline_metadata (report_json TEXT)')
        db.execute('INSERT INTO baseline_metadata VALUES (?)', (json.dumps(report),))
        db.commit(); db.close(); split.close()
        temp.replace(dbpath)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        report_path.with_name('phase2_error_examples.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False)+'\n')
        return report
    except BaseException:
        db.close(); split.close(); temp.unlink(missing_ok=True)
        Path(str(temp)+'-journal').unlink(missing_ok=True)
        raise
