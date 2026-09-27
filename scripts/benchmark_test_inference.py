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
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# --- Paths ---
cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

try:
    from business_entity_resolution.workspace import DATASET_DIR
except ImportError:
    DATASET_DIR = ROOT / 'data' / 'datasets'
    
TEST_DIR = DATASET_DIR / 'test'
DB_TEST = ROOT / 'data' / 'processed' / 'phase6_test_inference_chunk.sqlite'
V3_MODEL_PATH = ROOT / 'models' / 'experiment_v3_challenger.pkl'
CHECKPOINT_PATH = ROOT / 'reports' / 'test_inference_checkpoint_1000.csv'

M_TOKENS = 5
MAX_DF_RATIO = 0.05
MAX_DF_ABS = 5000
BENCHMARK_REF_LIMIT = 1000
V3_THRESHOLD = 0.25 # Locked tuning threshold

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def tokenize(text):
    if not text: return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

def extract_all_features(ref_row, tgt_row):
    """V3 (22-feature) extractor."""
    rn_raw, ra_raw = str(ref_row['name_raw'] or ''), str(ref_row['address_raw'] or '')
    tn_raw, ta_raw = str(tgt_row['name_raw'] or ''), str(tgt_row['address_raw'] or '')
    rn_norm, ra_norm = str(ref_row['name_norm'] or ''), str(ref_row['address_norm'] or '')
    tn_norm, ta_norm = str(tgt_row['name_norm'] or ''), str(tgt_row['address_norm'] or '')
    
    n_miss = 1.0 if not rn_raw.strip() or not tn_raw.strip() else 0.0
    a_miss = 1.0 if not ra_raw.strip() or not ta_raw.strip() else 0.0
    
    rt_n, tt_n = tokenize(rn_raw), tokenize(tn_raw)
    rt_a, tt_a = tokenize(ra_raw), tokenize(ta_raw)
    
    # 0-5: V1
    n_jac = len(rt_n & tt_n) / len(rt_n | tt_n) if (rt_n or tt_n) and not n_miss else 0.0
    a_jac = len(rt_a & tt_a) / len(rt_a | tt_a) if (rt_a or tt_a) and not a_miss else 0.0
    n_ex = 1.0 if rn_norm and tn_norm and rn_norm == tn_norm and not n_miss else 0.0
    a_ex = 1.0 if ra_norm and ta_norm and ra_norm == ta_norm and not a_miss else 0.0
    
    # 6-11: V2
    n_char = difflib.SequenceMatcher(None, rn_norm, tn_norm).ratio() if not n_miss else 0.0
    a_char = difflib.SequenceMatcher(None, ra_norm, ta_norm).ratio() if not a_miss else 0.0
    n_cont = 1.0 if not n_miss and rt_n and tt_n and (rt_n.issubset(tt_n) or tt_n.issubset(rt_n)) else 0.0
    a_cont = 1.0 if not a_miss and rt_a and tt_a and (rt_a.issubset(tt_a) or tt_a.issubset(rt_a)) else 0.0
    r_nums = set(re.findall(r'\d+', rn_raw + " " + ra_raw))
    t_nums = set(re.findall(r'\d+', tn_raw + " " + ta_raw))
    num_agr = 1.0 if r_nums and t_nums and (r_nums & t_nums) else 0.0
    num_con = 1.0 if r_nums and t_nums and not (r_nums & t_nums) else 0.0
    
    # 12-18: V3
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

def inspect_test_directory():
    print(f"--- 1. Inspecting Test Directory: {TEST_DIR} ---")
    if not TEST_DIR.exists():
        print(f"ERROR: Directory {TEST_DIR} does not exist.")
        return None, None
        
    files = list(TEST_DIR.glob('*'))
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f" - {f.name} ({size_mb:.1f} MB)")
        
    src1 = TEST_DIR / 'test_source1.tsv'
    if not src1.exists():
        print("Missing test_source1.tsv. Cannot benchmark.")
        return None, None
        
    # Count references to extrapolate
    print("Counting total test references...")
    total_refs = sum(1 for _ in open(src1, 'rb')) - 1
    print(f"Found {total_refs} total references.")
    
    return src1, total_refs

def retrieve_and_predict(test_refs, clf):
    print(f"\n--- 2. Benchmarking Retrieval & Prediction for {len(test_refs)} References ---")
    # For benchmark purposes, we will use the existing DEV database targets 
    # to measure latency and memory, avoiding building a multi-GB test DB until we are sure it's safe.
    DB_MOCK_TARGETS = ROOT / 'data' / 'processed' / 'phase2_baseline_dev.sqlite'
    
    t0 = time.time()
    
    conn = sqlite3.connect(DB_MOCK_TARGETS.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # We benchmark indexing the "US" targets as a proxy for the chunk size
    country = 'US'
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
    
    name_index = defaultdict(lambda: array.array('I'))
    addr_index = defaultdict(lambda: array.array('I'))
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
    print(f"[{get_peak_ram_mb():.1f} MB] Benchmark index built in {idx_time:.1f}s.")
    
    # Retrieval
    cands_dict = {}
    for ref in test_refs:
        route_a = set(exact_n_index.get(ref['name_norm'], [])) & set(exact_a_index.get(ref['address_norm'], [])) if (ref['name_norm'] and ref['address_norm']) else set()
        def get_top_k(text, idx_dict, idf_dict):
            surv = [(idf_dict[t], t) for t in tokenize(text) if t in idf_dict]
            surv.sort(key=lambda x: (-x[0], x[1]))
            scores = defaultdict(float)
            for idf, t in surv[:M_TOKENS]:
                for tgt_int in idx_dict[t]: scores[tgt_int] += idf
            return [target_int_to_str[tgt] for tgt, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:50]]
        cands_dict[ref['entity_id']] = list(route_a.union(get_top_k(ref['name_raw'], name_index, name_idf)).union(get_top_k(ref['address_raw'], addr_index, addr_idf)))
        
    # Feature Extraction
    all_tgt_ids = set().union(*cands_dict.values())
    targets = {}
    tgt_list = list(all_tgt_ids)
    for i in range(0, len(tgt_list), 999):
        batch = tgt_list[i:i+999]
        cur = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for r in cur: targets[r['entity_id']] = r
        
    predictions = {}
    for ref in test_refs:
        cands = cands_dict[ref['entity_id']]
        if not cands:
            predictions[ref['entity_id']] = []
            continue
            
        feats = []
        for cand in cands:
            feats.append(extract_all_features(ref, targets[cand]))
            
        probs = clf.predict_proba(feats)[:, 1]
        predicted_links = [cands[i] for i, p in enumerate(probs) if p >= V3_THRESHOLD]
        predictions[ref['entity_id']] = predicted_links
        
    conn.close()
    proc_time = time.time() - t0 - idx_time
    
    # Save Checkpoint
    with open(CHECKPOINT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reference_id', 'target_ids'])
        for ref_id, tgts in predictions.items():
            writer.writerow([ref_id, ",".join(tgts)])
            
    print(f"[{get_peak_ram_mb():.1f} MB] Predicted {len(test_refs)} refs in {proc_time:.1f}s.")
    return idx_time, proc_time

def main():
    src1, total_refs = inspect_test_directory()
    if not src1: return
    
    # Load V3 Model
    if not V3_MODEL_PATH.exists():
        print(f"ERROR: V3 Model missing at {V3_MODEL_PATH}")
        return
    with open(V3_MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
        
    # Load 1,000 Test References
    test_refs = []
    with open(src1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i >= BENCHMARK_REF_LIMIT: break
            # Mock normalization for benchmark purposes
            row['name_norm'] = str(row.get('business_name', '')).lower()
            row['address_norm'] = str(row.get('business_address', '')).lower()
            row['name_raw'] = row.get('business_name', '')
            row['address_raw'] = row.get('business_address', '')
            test_refs.append(row)
            
    idx_time, proc_time = retrieve_and_predict(test_refs, clf)
    
    # Extrapolate
    chunks = math.ceil(total_refs / BENCHMARK_REF_LIMIT)
    est_total_proc = (proc_time * chunks) / 60
    
    print(f"\n--- 3. Extrapolation & Checkpointing ---")
    print(f"Estimated Inference Time (Processing {total_refs} refs): ~{est_total_proc:.1f} minutes")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH.name}")
    print(f"\nNOTE: Full-scale memory is NOT fully validated until we index the real {TEST_DIR} targets.")
    print("Holdout remains untouched.")

if __name__ == "__main__":
    main()