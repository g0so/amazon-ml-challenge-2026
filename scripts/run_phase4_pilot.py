import sqlite3
import re
import math
import array
import json
import random
import psutil
import time
import sys
import gc
import pickle
import csv
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# --- Paths & Setup ---
cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))
from business_entity_resolution.normalize import normalize_record
from business_entity_resolution.workspace import DATASET_DIR

M_TOKENS = 5
MAX_DF_RATIO = 0.05
MAX_DF_ABS = 5000
TRAIN_SAMPLE_SIZE = 5000
TRAIN_SEED = 2028
TUNING_SEED = 2029

DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
DB_SPLIT = ROOT / 'data/processed/phase2_split.sqlite'
TRUTH_PATH = DATASET_DIR / 'train' / 'train_ground_truth.tsv'

# The reviewer explicitly noted the existing JSON is Seed 2027
DEV_2027_IDS = ROOT / 'data/processed/phase3_pilot_sample_ids.json' 
DEV_TUNING_IDS = ROOT / 'data/processed/phase4_dev_tuning_ids.json'
TRAIN_IDS = ROOT / 'data/processed/phase4_train_sample_ids.json'

MODEL_PATH = ROOT / 'models' / 'phase4_logreg.pkl'
FEAT_DEFS = ROOT / 'models' / 'phase4_features.json'
RESULTS_PATH = ROOT / 'reports' / 'phase4_pilot_results.json'

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_ram_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def check_memory_budget(limit_mb=12000):
    if get_ram_mb() > limit_mb:
        raise MemoryError(f"Graceful Abort: Process exceeded {limit_mb} MB limit to prevent system swap crash.")

def tokenize(text):
    if not text: return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

# --- Embedded Tests ---
# --- Embedded Tests ---
def run_preflight_tests():
    print("Running Preflight Tests...")
    
    # 1. Missing Field Logic using dictionary mocks
    ref = {'name_raw': 'Acme', 'address_raw': '', 'name_norm': 'acme', 'address_norm': ''}
    tgt = {'name_raw': 'Acme', 'address_raw': '', 'name_norm': 'acme', 'address_norm': ''}
    
    feats = extract_features(ref, tgt)
    assert feats[0] == 1.0 and feats[2] == 1.0, "Valid name match failed"
    assert feats[1] == 0.0 and feats[3] == 0.0, "Missing address must not count as matching evidence"
    assert feats[5] == 1.0, "Missing address flag failed"
    
    # 2. Oracle vs Actual Evaluation
    truth = {"ref1": {"A", "B"}}
    preds = {"ref1": {"A", "C"}}
    cand_pool = {"ref1": {"A", "C", "D"}}
    
    res_actual = score_macro_f05(preds, truth, ["ref1"])
    # TP=1, FP=1, FN=1. F0.5 = 1.25(1) / (1.25(1) + 1 + 0.25(1)) = 1.25 / 2.5 = 0.5
    assert math.isclose(res_actual['macro_f05'], 0.5), "Actual macro math wrong"
    
    # Oracle perfectly isolates true targets from the pool (A) and discards C, D
    oracle_pool = {r: c & truth[r] for r, c in cand_pool.items()}
    res_oracle = score_macro_f05(oracle_pool, truth, ["ref1"])
    # Oracle TP=1, FP=0, FN=1. F0.5 = 1.25(1) / (1.25(1) + 0 + 0.25(1)) = 1.25 / 1.5 = 0.8333
    assert math.isclose(res_oracle['macro_f05'], 0.8333333333), "Oracle macro math wrong"
    
    print("Preflight Tests Passed.")

# --- Core Logic ---
def extract_features(ref_row, tgt_row):
    rn_raw, ra_raw = str(ref_row['name_raw'] or ''), str(ref_row['address_raw'] or '')
    tn_raw, ta_raw = str(tgt_row['name_raw'] or ''), str(tgt_row['address_raw'] or '')
    rn_norm, ra_norm = str(ref_row['name_norm'] or ''), str(ref_row['address_norm'] or '')
    tn_norm, ta_norm = str(tgt_row['name_norm'] or ''), str(tgt_row['address_norm'] or '')
    
    n_miss = 1.0 if not rn_raw.strip() or not tn_raw.strip() else 0.0
    a_miss = 1.0 if not ra_raw.strip() or not ta_raw.strip() else 0.0
    
    rt_n, tt_n = tokenize(rn_raw), tokenize(tn_raw)
    rt_a, tt_a = tokenize(ra_raw), tokenize(ta_raw)
    
    n_jaccard = len(rt_n & tt_n) / len(rt_n | tt_n) if (rt_n or tt_n) and not n_miss else 0.0
    a_jaccard = len(rt_a & tt_a) / len(rt_a | tt_a) if (rt_a or tt_a) and not a_miss else 0.0
    
    n_exact = 1.0 if rn_norm and tn_norm and rn_norm == tn_norm and not n_miss else 0.0
    a_exact = 1.0 if ra_norm and ta_norm and ra_norm == ta_norm and not a_miss else 0.0
    
    return [n_jaccard, a_jaccard, n_exact, a_exact, n_miss, a_miss]

def score_macro_f05(preds_dict, truth_dict, req_refs):
    scores = []
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    zero_retrieved_ns, total_ns = 0, 0
    
    for ref in req_refs:
        if ref not in truth_dict:
            raise ValueError(f"Missing explicit ground truth for {ref}.")
        g_set = truth_dict[ref]
        p_set = preds_dict.get(ref, set())
        
        g = len(g_set)
        t = len(p_set & g_set)
        fp = len(p_set - g_set)
        fn = g - t
        
        tp_sum += t; fp_sum += fp; fn_sum += fn
        
        if g == 0:
            scores.append(1.0 if fp == 0 else 0.0)
        else:
            total_ns += 1
            if t == 0:
                scores.append(0.0)
                zero_retrieved_ns += 1
            else:
                scores.append((1.25 * t) / (1.25 * t + fp + 0.25 * fn))
                
    macro = sum(scores) / len(scores) if scores else 0.0
    up = tp_sum / (tp_sum + fp_sum) if tp_sum+fp_sum > 0 else 0.0
    ur = tp_sum / (tp_sum + fn_sum) if tp_sum+fn_sum > 0 else 0.0
    return {
        'macro_f05': macro, 'micro_p': up, 'micro_r': ur,
        'tp': tp_sum, 'fp': fp_sum, 'fn': fn_sum,
        'zero_retrieved_rate': zero_retrieved_ns / total_ns if total_ns > 0 else 0.0
    }

def get_oracle_macro_f05(cand_pool_dict, truth_dict, req_refs):
    """Calculates strict oracle ceiling assuming FP=0. Singletons (g=0) score 1.0."""
    scores = []
    for ref in req_refs:
        g_set = truth_dict.get(ref, set())
        g = len(g_set)
        if g == 0:
            scores.append(1.0)
        else:
            # FIX: Ensure p_set is explicitly cast to a set
            p_set = set(cand_pool_dict.get(ref, set()))
            t = len(p_set & g_set)
            if t == 0:
                scores.append(0.0)
            else:
                scores.append((1.25 * t) / (t + 0.25 * g))
    return sum(scores) / len(scores) if scores else 0.0

def load_truth_for_refs(ref_list):
    ref_set = set(ref_list)
    truth = {}
    with open(TRUTH_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split('\t')
            if parts[0] in ref_set:
                truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
    
    missing = ref_set - set(truth.keys())
    if missing:
        raise ValueError(f"Missing explicit truth labels for {len(missing)} required references.")
    return truth

def build_train_db_if_needed():
    if DB_TRAIN.exists(): return
    print(f"[{get_ram_mb():.1f} MB] Building disk-backed Train text database...")
    
    conn_split = sqlite3.connect(DB_SPLIT.as_uri() + '?mode=ro', uri=True)
    
    print("Loading Train Target IDs from split assignment...")
    cur = conn_split.execute("SELECT target_id FROM target_assignments WHERE split='train'")
    train_targets = {r[0] for r in cur}
    
    print("Sampling 5,000 Train References from split assignment...")
    cur = conn_split.execute("SELECT reference_id FROM reference_assignments WHERE split='train' ORDER BY reference_id")
    all_train_refs = [r[0] for r in cur.fetchall()]
    random.seed(TRAIN_SEED)
    sample_train_refs = set(random.sample(all_train_refs, TRAIN_SAMPLE_SIZE))
    with open(TRAIN_IDS, 'w') as f: json.dump(list(sample_train_refs), f)
    
    conn_split.close()
    
    conn = sqlite3.connect(DB_TRAIN)
    conn.execute("CREATE TABLE records (entity_id TEXT PRIMARY KEY, source INTEGER, country TEXT, name_raw TEXT, address_raw TEXT, name_norm TEXT, address_norm TEXT)")
    
    print(f"[{get_ram_mb():.1f} MB] Streaming TSVs for Train targets...")
    for src in [2, 3]:
        with open(DATASET_DIR / 'train' / f'train_source{src}.tsv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            batch = []
            for r in reader:
                if r['entity_id'] in train_targets:
                    norm = normalize_record(r['business_name'], r['business_address'])
                    batch.append((r['entity_id'], src, r['country'], norm['name_raw'], norm['address_raw'], norm['name_norm'], norm['address_norm']))
                    if len(batch) >= 10000:
                        conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", batch)
                        batch = []
            if batch: conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", batch)

    print(f"[{get_ram_mb():.1f} MB] Streaming TSV for sampled Train references...")
    with open(DATASET_DIR / 'train' / 'train_source1.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        batch = []
        for r in reader:
            if r['entity_id'] in sample_train_refs:
                norm = normalize_record(r['business_name'], r['business_address'])
                batch.append((r['entity_id'], 1, r['country'], norm['name_raw'], norm['address_raw'], norm['name_norm'], norm['address_norm']))
                if len(batch) >= 1000:
                    conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", batch)
                    batch = []
        if batch: conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", batch)
        
    conn.commit()
    conn.close()
    print(f"[{get_ram_mb():.1f} MB] Train DB built successfully.")

def retrieve_for_country(country, db_path, ref_ids):
    check_memory_budget()
    conn = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print(f"[{get_ram_mb():.1f} MB] Indexing targets for {country} in {db_path.name}...")
    cur.execute("SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM records WHERE source IN (2,3) AND country=? ORDER BY entity_id", (country,))
    
    target_str_to_int = {}
    target_int_to_str = []
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

    print(f"[{get_ram_mb():.1f} MB] Querying {len(ref_ids)} {country} references...")
    cur.execute(f"SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM records WHERE source=1 AND country=? AND entity_id IN ({','.join(['?']*len(ref_ids))})", [country] + list(ref_ids))
    
    candidates_dict = {}
    for ref in cur:
        cands_n = set(exact_n_index.get(ref['name_norm'], [])) if ref['name_norm'] else set()
        cands_a = set(exact_a_index.get(ref['address_norm'], [])) if ref['address_norm'] else set()
        route_a = cands_n & cands_a
        
        def get_top_k(text, idx_dict, idf_dict):
            surv = [(idf_dict[t], t) for t in tokenize(text) if t in idf_dict]
            surv.sort(key=lambda x: (-x[0], x[1])) # Deterministic: -IDF, Token ASC
            scores = defaultdict(float)
            for idf, t in surv[:M_TOKENS]:
                for tgt_int in idx_dict[t]: scores[tgt_int] += idf
            ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0])) # -Score, ID ASC
            return [target_int_to_str[tgt] for tgt, _ in ranked[:50]]

        cands = list(route_a.union(get_top_k(ref['name_raw'], name_index, name_idf)).union(get_top_k(ref['address_raw'], addr_index, addr_idf)))
        candidates_dict[ref['entity_id']] = cands
        
    del name_index, addr_index, exact_n_index, exact_a_index, target_str_to_int, target_int_to_str
    gc.collect()
    return candidates_dict

def get_features_for_candidates(db_path, candidates_dict):
    """Fetches candidates in bounded batches from disk to avoid RAM bloat."""
    conn = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    
    # 1. Fetch Refs
    refs = {}
    ref_list = list(candidates_dict.keys())
    for i in range(0, len(ref_list), 999):
        batch = ref_list[i:i+999]
        cur = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for r in cur: refs[r['entity_id']] = r
        
    # 2. Invert Dictionary to fetch targets by batch
    cand_to_refs = defaultdict(list)
    for ref_id, cands in candidates_dict.items():
        for cand_id in cands:
            cand_to_refs[cand_id].append(ref_id)
            
    X_dict, pair_ids = defaultdict(list), defaultdict(list)
    all_tgt_ids = list(cand_to_refs.keys())
    
    for i in range(0, len(all_tgt_ids), 999):
        batch_tgts = all_tgt_ids[i:i+999]
        cur = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch_tgts))})", batch_tgts)
        for tgt_row in cur:
            tgt_id = tgt_row['entity_id']
            for ref_id in cand_to_refs[tgt_id]:
                X_dict[ref_id].append(extract_features(refs[ref_id], tgt_row))
                pair_ids[ref_id].append(tgt_id)
                
    conn.close()
    return X_dict, pair_ids

def main():
    t0 = time.time()
    run_preflight_tests()
    
    build_train_db_if_needed()
    train_ids = json.loads(TRAIN_IDS.read_text())
    
    # Dev Sample Setup
    if not DEV_2027_IDS.exists():
        raise FileNotFoundError(f"Missing {DEV_2027_IDS}. Aborting to preserve valid comparisons.")
    dev_2027_ids = set(json.loads(DEV_2027_IDS.read_text()))
    
    if not DEV_TUNING_IDS.exists():
        conn = sqlite3.connect(DB_DEV.as_uri() + '?mode=ro', uri=True)
        all_dev = {row[0] for row in conn.execute("SELECT entity_id FROM records WHERE source=1")}
        eligible = sorted(list(all_dev - dev_2027_ids))
        random.seed(TUNING_SEED)
        tuning_ids = random.sample(eligible, 2000)
        with open(DEV_TUNING_IDS, 'w') as f: json.dump(tuning_ids, f)
        conn.close()
    tuning_ids = json.loads(DEV_TUNING_IDS.read_text())
    
    # Strict validation of disjointness
    assert len(set(tuning_ids) & dev_2027_ids) == 0, "Tuning and Eval samples must be strictly disjoint."
    
    # --- 1. Retrieval ---
    train_cands, tune_cands, eval_cands = {}, {}, {}
    for country in ['US', 'India']:
        train_cands.update(retrieve_for_country(country, DB_TRAIN, train_ids))
        tune_cands.update(retrieve_for_country(country, DB_DEV, tuning_ids))
        eval_cands.update(retrieve_for_country(country, DB_DEV, dev_2027_ids))
        
    # --- 2. Feature Generation ---
    print(f"[{get_ram_mb():.1f} MB] Generating Features from Disk...")
    X_train_dict, pairs_train = get_features_for_candidates(DB_TRAIN, train_cands)
    X_tune_dict, pairs_tune = get_features_for_candidates(DB_DEV, tune_cands)
    X_eval_dict, pairs_eval = get_features_for_candidates(DB_DEV, eval_cands)
    
    # --- 3. Model Fitting (TRAIN) ---
    truth_train = load_truth_for_refs(train_ids)
    X_train_flat, y_train_flat = [], []
    for ref, feats in X_train_dict.items():
        g_set = truth_train[ref]
        for idx, cand in enumerate(pairs_train[ref]):
            X_train_flat.append(feats[idx])
            y_train_flat.append(1 if cand in g_set else 0)
            
    print(f"Fitting LR on {len(X_train_flat)} Train pairs ({sum(y_train_flat)} positives)...")
    clf = LogisticRegression(class_weight=None, max_iter=1000, random_state=42)
    clf.fit(X_train_flat, y_train_flat)
    
    with open(MODEL_PATH, 'wb') as f: pickle.dump(clf, f)
    with open(FEAT_DEFS, 'w') as f: json.dump(["name_jaccard", "addr_jaccard", "name_exact", "addr_exact", "name_missing", "addr_missing"], f)

    # --- 4. Tuning Threshold (SEED 2029) ---
    print(f"[{get_ram_mb():.1f} MB] Tuning Threshold on Disjoint Sample (Seed 2029)...")
    truth_tune = load_truth_for_refs(tuning_ids)
    tune_probs = {}
    for ref, feats in X_tune_dict.items():
        tune_probs[ref] = clf.predict_proba(feats)[:, 1] if feats else []
            
    best_t, best_f05 = 0.0, 0.0
    for t in np.arange(-0.1, 1.2, 0.1): # Covers predict-all (<0) and predict-none (>1)
        preds = {r: {pairs_tune[r][i] for i, p in enumerate(tune_probs[r]) if p >= t} for r in tuning_ids}
        macro = score_macro_f05(preds, truth_tune, tuning_ids)['macro_f05']
        if macro > best_f05: best_t, best_f05 = t, macro
            
    print(f"Best Tested Threshold: {best_t:.2f} (Tuning Macro F0.5: {best_f05:.4f})")
    
    # --- 5. Evaluation (SEED 2027) ---
    truth_eval = load_truth_for_refs(dev_2027_ids)
    eval_preds, eval_probs_dict = {}, {}
    
    for ref, feats in X_eval_dict.items():
        if feats:
            probs = clf.predict_proba(feats)[:, 1]
            eval_preds[ref] = {pairs_eval[ref][i] for i, p in enumerate(probs) if p >= best_t}
            eval_probs_dict[ref] = {pairs_eval[ref][i]: float(p) for i, p in enumerate(probs)}
        else:
            eval_preds[ref] = set()
            eval_probs_dict[ref] = {}
            
    # Calculate Per-Sample Baselines
    res_model = score_macro_f05(eval_preds, truth_eval, dev_2027_ids)
    res_oracle = get_oracle_macro_f05(eval_cands, truth_eval, dev_2027_ids)
    
    conn = sqlite3.connect(DB_DEV.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    exact_n, exact_a = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for row in conn.execute("SELECT entity_id, country, name_norm, address_norm FROM records WHERE source IN (2,3)"):
        if row['name_norm']: exact_n[row['country']][row['name_norm']].append(row['entity_id'])
        if row['address_norm']: exact_a[row['country']][row['address_norm']].append(row['entity_id'])
    
    exact_baseline = {}
    for row in conn.execute(f"SELECT entity_id, country, name_norm, address_norm FROM records WHERE entity_id IN ({','.join(['?']*len(dev_2027_ids))})", list(dev_2027_ids)):
        if row['name_norm'] and row['address_norm']:
            cand_n = set(exact_n[row['country']].get(row['name_norm'], []))
            cand_a = set(exact_a[row['country']].get(row['address_norm'], []))
            exact_baseline[row['entity_id']] = cand_n & cand_a
        else: exact_baseline[row['entity_id']] = set()
    res_exact = score_macro_f05(exact_baseline, truth_eval, dev_2027_ids)
    
    print("\n| Metric | Seed-2027 Target Results |")
    print(f"| :--- | :--- |")
    print(f"| Per-Sample Exact-Match Baseline | {res_exact['macro_f05']:.4f} |")
    print(f"| Per-Sample Oracle F0.5 Ceiling | {res_oracle:.4f} |")
    print(f"| Classifier Actual Macro F0.5 | {res_model['macro_f05']:.4f} |")
    print(f"| Classifier TP / FP / FN | {res_model['tp']} / {res_model['fp']} / {res_model['fn']} |")
    
    # Error Output matching strict request requirements
    errors = []
    for r in dev_2027_ids:
        if len(errors) >= 10: break
        g_set = truth_eval[r]
        p_set = eval_preds[r]
        
        for fp_cand in (p_set - g_set):
            errors.append({"type": "FP", "ref": r, "target": fp_cand, "prob": eval_probs_dict[r].get(fp_cand, None), "retrieved": True})
        for fn_cand in (g_set - p_set):
            prob = eval_probs_dict[r].get(fn_cand, None)
            errors.append({"type": "FN", "ref": r, "target": fn_cand, "prob": prob, "retrieved": (prob is not None)})
            
    with open(ROOT / 'reports' / 'phase4_pilot_errors.json', 'w') as f: json.dump(errors[:10], f, indent=2)
    with open(RESULTS_PATH, 'w') as f:
        json.dump({'threshold': best_t, 'res_model': res_model, 'res_exact': res_exact, 'oracle_f05': res_oracle}, f, indent=2)
        
    print(f"\n[{get_ram_mb():.1f} MB] Pilot completed cleanly in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()