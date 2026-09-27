import sqlite3
import re
import math
import array
import json
import time
import sys
import gc
import pickle
import difflib
import resource
import os
import csv
import hashlib
import argparse
from collections import defaultdict
from pathlib import Path

# --- Configuration & Paths ---
cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

from business_entity_resolution.workspace import DATASET_DIR
from business_entity_resolution.normalize import normalize_record

TEST_DIR = DATASET_DIR / 'test'
DB_TEST = ROOT / 'data' / 'processed' / 'phase6_test_targets.sqlite'
DB_TEST_TMP = ROOT / 'data' / 'processed' / 'phase6_test_targets.sqlite.tmp'
V3_MODEL_PATH = ROOT / 'models' / 'experiment_v3_challenger.pkl'
MANIFEST_PATH = ROOT / 'reports' / 'test_inference_manifest.json'

# Attempt to read official validator to determine the exact required TSV output names
# ER competitions often require specific names like `predictions.tsv` or dual files
README_PATH = TEST_DIR / 'README.md'
VALIDATOR_PATH = TEST_DIR / 'validator.py'
OUTPUT_TSV_1 = ROOT / 'reports' / 'submission_1.tsv' 
OUTPUT_TSV_2 = ROOT / 'reports' / 'submission_2.tsv'

M_TOKENS = 5
MAX_DF_RATIO = 0.05
MAX_DF_ABS = 5000
V3_THRESHOLD = 0.25
BATCH_SIZE = 1000

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def get_file_hash(filepath):
    if not filepath.exists(): return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def tokenize(text):
    if not text: return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

# --- Exact V3 Feature Extractor (22 Features) ---
def extract_all_features(ref_row, tgt_row):
    rn_raw, ra_raw = str(ref_row['name_raw'] or ''), str(ref_row['address_raw'] or '')
    tn_raw, ta_raw = str(tgt_row['name_raw'] or ''), str(tgt_row['address_raw'] or '')
    rn_norm, ra_norm = str(ref_row['name_norm'] or ''), str(ref_row['address_norm'] or '')
    tn_norm, ta_norm = str(tgt_row['name_norm'] or ''), str(tgt_row['address_norm'] or '')
    
    n_miss = 1.0 if not rn_raw.strip() or not tn_raw.strip() else 0.0
    a_miss = 1.0 if not ra_raw.strip() or not ta_raw.strip() else 0.0
    
    rt_n, tt_n = tokenize(rn_raw), tokenize(tn_raw)
    rt_a, tt_a = tokenize(ra_raw), tokenize(ta_raw)
    
    # 0-5: V1 Features
    n_jac = len(rt_n & tt_n) / len(rt_n | tt_n) if (rt_n or tt_n) and not n_miss else 0.0
    a_jac = len(rt_a & tt_a) / len(rt_a | tt_a) if (rt_a or tt_a) and not a_miss else 0.0
    n_ex = 1.0 if rn_norm and tn_norm and rn_norm == tn_norm and not n_miss else 0.0
    a_ex = 1.0 if ra_norm and ta_norm and ra_norm == ta_norm and not a_miss else 0.0
    
    # 6-11: V2 Features
    n_char = difflib.SequenceMatcher(None, rn_norm, tn_norm).ratio() if not n_miss else 0.0
    a_char = difflib.SequenceMatcher(None, ra_norm, ta_norm).ratio() if not a_miss else 0.0
    n_cont = 1.0 if not n_miss and rt_n and tt_n and (rt_n.issubset(tt_n) or tt_n.issubset(rt_n)) else 0.0
    a_cont = 1.0 if not a_miss and rt_a and tt_a and (rt_a.issubset(tt_a) or tt_a.issubset(rt_a)) else 0.0
    r_nums = set(re.findall(r'\d+', rn_raw + " " + ra_raw))
    t_nums = set(re.findall(r'\d+', tn_raw + " " + ta_raw))
    num_agr = 1.0 if r_nums and t_nums and (r_nums & t_nums) else 0.0
    num_con = 1.0 if r_nums and t_nums and not (r_nums & t_nums) else 0.0
    
    # 12-18: V3 Numeric
    ra_nums = set(re.findall(r'\d+', ra_raw))
    ta_nums = set(re.findall(r'\d+', ta_raw))
    a_num_both = 1.0 if ra_nums and ta_nums else 0.0
    a_num_one = 1.0 if (bool(ra_nums) ^ bool(ta_nums)) else 0.0
    a_num_none = 1.0 if not ra_nums and not ta_nums else 0.0
    a_num_exact = 1.0 if a_num_both and (ra_nums == ta_nums) else 0.0
    a_num_partial = 1.0 if a_num_both and (ra_nums & ta_nums) and (ra_nums != ta_nums) else 0.0
    a_num_unshared_ref = 1.0 if a_num_both and (ra_nums - ta_nums) else 0.0
    a_num_unshared_tgt = 1.0 if a_num_both and (ta_nums - ra_nums) else 0.0
    
    # 19-21: V3 Interactions
    int_n_jac_a_miss = n_jac * a_miss
    int_n_char_a_miss = n_char * a_miss
    int_n_cont_a_miss = n_cont * a_miss
    
    return [
        n_jac, a_jac, n_ex, a_ex, n_miss, a_miss,
        n_char, a_char, n_cont, a_cont, num_agr, num_con,
        a_num_both, a_num_one, a_num_none, a_num_exact, a_num_partial, a_num_unshared_ref, a_num_unshared_tgt,
        int_n_jac_a_miss, int_n_char_a_miss, int_n_cont_a_miss
    ]

# --- 1. Atomic Database Construction ---
def build_atomic_db():
    print(f"[{get_peak_ram_mb():.1f} MB] Checking Database Requirements...")
    src_files = [TEST_DIR / f'test_source{i}.tsv' for i in [1, 2, 3]]
    for f in src_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing required test input: {f}")

    total_expected = 0
    for f in src_files:
        total_expected += sum(1 for _ in open(f, 'rb')) - 1

    if DB_TEST.exists():
        conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
        count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        conn.close()
        if count == total_expected:
            print(f"Valid test database already exists with {count} records. Skipping build.")
            return
        else:
            print(f"Existing DB count mismatch ({count} vs {total_expected}). Rebuilding...")
            DB_TEST.unlink()

    if DB_TEST_TMP.exists(): DB_TEST_TMP.unlink()

    print(f"[{get_peak_ram_mb():.1f} MB] Building temporary DB atomically...")
    conn = sqlite3.connect(DB_TEST_TMP)
    conn.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY, source INTEGER, country TEXT, 
            name_raw TEXT, address_raw TEXT, name_norm TEXT, address_norm TEXT
        )
    """)

    t0 = time.time()
    for src in [1, 2, 3]:
        with open(TEST_DIR / f'test_source{src}.tsv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            batch = []
            for r in reader:
                norm = normalize_record(r.get('business_name'), r.get('business_address'))
                batch.append((
                    r['entity_id'], src, r['country'], 
                    r.get('business_name', ''), r.get('business_address', ''), 
                    norm['name_raw'], norm['address_raw']
                ))
                if len(batch) >= 20000:
                    try:
                        conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", batch)
                    except sqlite3.IntegrityError:
                        raise ValueError("Duplicate entity_id detected! Database rejected.")
                    batch = []
            if batch: conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", batch)

    conn.execute("CREATE INDEX idx_country_source ON records(country, source)")
    conn.commit()
    conn.close()

    os.rename(DB_TEST_TMP, DB_TEST)
    print(f"[{get_peak_ram_mb():.1f} MB] Database finalized successfully in {time.time()-t0:.1f}s.")

# --- 2. Per-Country Indexing ---
def build_country_index(country):
    t0 = time.time()
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM records WHERE source IN (2,3) AND country=? ORDER BY entity_id", (country,))
    target_str_to_int, target_int_to_str = {}, []
    name_df, addr_df = defaultdict(int), defaultdict(int)
    
    for idx, row in enumerate(cur):
        target_int_to_str.append(row['entity_id'])
        target_str_to_int[row['entity_id']] = idx
        for t in tokenize(row['name_raw']): name_df[t] += 1
        for t in tokenize(row['address_raw']): addr_df[t] += 1
            
    limit = min(MAX_DF_ABS, len(target_int_to_str) * MAX_DF_RATIO)
    name_idf = {t: math.log(len(target_int_to_str) / df) for t, df in name_df.items() if df <= limit}
    addr_idf = {t: math.log(len(target_int_to_str) / df) for t, df in addr_df.items() if df <= limit}
    
    name_index, addr_index = defaultdict(lambda: array.array('I')), defaultdict(lambda: array.array('I'))
    exact_n_index, exact_a_index = defaultdict(list), defaultdict(list)
    
    cur.execute("SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM records WHERE source IN (2,3) AND country=?", (country,))
    for row in cur:
        tgt_int = target_str_to_int[row['entity_id']]
        for t in tokenize(row['name_raw']):
            if t in name_idf: name_index[t].append(tgt_int)
        for t in tokenize(row['address_raw']):
            if t in addr_idf: addr_index[t].append(tgt_int)
        if row['name_norm']: exact_n_index[row['name_norm']].append(row['entity_id'])
        if row['address_norm']: exact_a_index[row['address_norm']].append(row['entity_id'])

    idx_time = time.time() - t0
    conn.close()
    return name_index, name_idf, addr_index, addr_idf, exact_n_index, exact_a_index, target_int_to_str, idx_time

def process_batch(refs, indexes, clf):
    t_start = time.time()
    name_index, name_idf, addr_index, addr_idf, exact_n_index, exact_a_index, target_int_to_str, _ = indexes
    
    # Pre-tokenize refs
    ref_tokens = {}
    for r in refs:
        ref_tokens[r['entity_id']] = {
            'n': [(name_idf[t], t) for t in tokenize(r['name_raw']) if t in name_idf],
            'a': [(addr_idf[t], t) for t in tokenize(r['address_raw']) if t in addr_idf]
        }
        ref_tokens[r['entity_id']]['n'].sort(key=lambda x: (-x[0], x[1]))
        ref_tokens[r['entity_id']]['a'].sort(key=lambda x: (-x[0], x[1]))

    t_ret = time.time()
    cands_dict = {}
    for r in refs:
        e_id = r['entity_id']
        route_a = set(exact_n_index.get(r['name_norm'], [])) & set(exact_a_index.get(r['address_norm'], [])) if (r['name_norm'] and r['address_norm']) else set()
        
        def get_top_k(tok_list, idx_dict):
            scores = defaultdict(float)
            for idf, t in tok_list[:M_TOKENS]:
                for tgt_int in idx_dict[t]: scores[tgt_int] += idf
            return [target_int_to_str[tgt] for tgt, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:50]]
            
        cands_dict[e_id] = list(route_a.union(get_top_k(ref_tokens[e_id]['n'], name_index)).union(get_top_k(ref_tokens[e_id]['a'], addr_index)))
    
    ret_time = time.time() - t_ret
    
    t_fetch = time.time()
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    all_tgt_ids = list(set().union(*cands_dict.values()))
    targets = {}
    for i in range(0, len(all_tgt_ids), 999):
        batch = all_tgt_ids[i:i+999]
        cur = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for row in cur: targets[row['entity_id']] = dict(row)
    conn.close()
    fetch_time = time.time() - t_fetch
    
    t_feat = time.time()
    X_batch, lookup = [], []
    for r in refs:
        for cand in cands_dict[r['entity_id']]:
            X_batch.append(extract_all_features(r, targets[cand]))
            lookup.append((r['entity_id'], cand))
    feat_time = time.time() - t_feat
    
    t_pred = time.time()
    predictions = defaultdict(list)
    if X_batch:
        probs = clf.predict_proba(X_batch)[:, 1]
        for i, p in enumerate(probs):
            if p >= V3_THRESHOLD:
                predictions[lookup[i][0]].append(lookup[i][1])
    pred_time = time.time() - t_pred
    
    return predictions, (time.time()-t_start, ret_time, fetch_time, feat_time, pred_time)

def run_benchmark():
    print(f"\n=== BENCHMARK MODE (100 Refs / Country) ===")
    build_atomic_db()
    
    with open(V3_MODEL_PATH, 'rb') as f: clf = pickle.load(f)
    assert clf.n_features_in_ == 22, f"Model feature count mismatch! Expected 22, got {clf.n_features_in_}"
    
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM records WHERE source=1").fetchall()]
    
    total_refs_count = 0
    est_total_time = 0
    
    for country in countries:
        print(f"\n--- Profiling {country} ---")
        cur = conn.execute("SELECT COUNT(*) FROM records WHERE source=1 AND country=?", (country,))
        c_count = cur.fetchone()[0]
        total_refs_count += c_count
        
        print(f"[{get_peak_ram_mb():.1f} MB] Building Full Index for {country} targets...")
        indexes = build_country_index(country)
        print(f"[{get_peak_ram_mb():.1f} MB] Index built in {indexes[-1]:.1f}s.")
        
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM records WHERE source=1 AND country=? LIMIT 100", (country,))
        refs = [dict(r) for r in cur.fetchall()]
        
        if not refs: continue
        preds, (tot, ret, ftc, feat, pred) = process_batch(refs, indexes, clf)
        
        time_per_1k = (tot / len(refs)) * 1000
        country_est = (time_per_1k * (c_count / 1000)) / 60
        est_total_time += country_est
        
        print(f"Timings (per 100 refs): Total={tot:.2f}s | Ret={ret:.2f}s | Fetch={ftc:.2f}s | Feat={feat:.2f}s | Pred={pred:.2f}s")
        print(f"Est. full run for {c_count} {country} refs: ~{country_est:.1f} minutes")
        
        del indexes
        gc.collect()

    print(f"\n=== BENCHMARK COMPLETE ===")
    print(f"Absolute Peak RAM Measured: {get_peak_ram_mb():.1f} MB")
    print(f"Estimated Total Runtime: ~{est_total_time/60:.1f} HOURS")
    print(f"\nTo start/resume the full production run, execute:\n  python3 scripts/run_phase6_test_inference.py --resume")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.resume:
        print("Resume logic is staged. Please verify benchmark boundaries first.")
    else:
        print("Please specify --benchmark to profile, or --resume to run/resume.")