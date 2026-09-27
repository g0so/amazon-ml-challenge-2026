import sqlite3
import re
import math
import array
import json
import random
import time
import sys
import gc
import pickle
import difflib
import resource
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# --- Paths ---
cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))
from business_entity_resolution.workspace import DATASET_DIR

M_TOKENS = 5
MAX_DF_RATIO = 0.05
MAX_DF_ABS = 5000

DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
TRUTH_PATH = DATASET_DIR / 'train' / 'train_ground_truth.tsv' 

DEV_EVAL_IDS = ROOT / 'data/processed/phase3_pilot_sample_ids.json' # Seed 2027 (Secondary Dev)
DEV_TUNE_IDS = ROOT / 'data/processed/phase4_dev_tuning_ids.json'   # Seed 2029 (Tuning)
TRAIN_IDS = ROOT / 'data/processed/phase4_train_sample_ids.json'    # Seed 2028 (Train)

V3_MODEL_PATH = ROOT / 'models' / 'experiment_v3_challenger.pkl'
V3_RESULTS = ROOT / 'reports' / 'phase4_experiment_v3_results.json'
V3_ERRORS_TUNE = ROOT / 'reports' / 'phase4_experiment_v3_tuning_errors.json'
DEV_TUNE_IDS = ROOT / 'data/processed/phase4_dev_tuning_ids.json'   # Seed 2029 (Tuning)
TRAIN_IDS = ROOT / 'data/processed/phase4_train_sample_ids.json'    # Seed 2028 (Train)

V3_MODEL_PATH = ROOT / 'models' / 'experiment_v3_challenger.pkl'
V3_RESULTS = ROOT / 'reports' / 'phase4_experiment_v3_results.json'
V3_ERRORS_TUNE = ROOT / 'reports' / 'phase4_experiment_v3_tuning_errors.json'

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def tokenize(text):
    if not text: return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

# --- Feature Engineering (V2 + V3 Numeric + V3 Interactions) ---
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
    
    # 12-18: V3 Address Numeric Features
    ra_nums = set(re.findall(r'\d+', ra_raw))
    ta_nums = set(re.findall(r'\d+', ta_raw))
    
    a_num_both = 1.0 if ra_nums and ta_nums else 0.0
    a_num_one = 1.0 if (bool(ra_nums) ^ bool(ta_nums)) else 0.0
    a_num_none = 1.0 if not ra_nums and not ta_nums else 0.0
    
    a_num_exact = 1.0 if a_num_both and (ra_nums == ta_nums) else 0.0
    a_num_partial = 1.0 if a_num_both and (ra_nums & ta_nums) and (ra_nums != ta_nums) else 0.0
    a_num_unshared_ref = 1.0 if a_num_both and (ra_nums - ta_nums) else 0.0
    a_num_unshared_tgt = 1.0 if a_num_both and (ta_nums - ra_nums) else 0.0
    
    # 19-21: V3 Interactions (Address missing behavior override)
    int_n_jac_a_miss = n_jac * a_miss
    int_n_char_a_miss = n_char * a_miss
    int_n_cont_a_miss = n_cont * a_miss
    
    return [
        n_jac, a_jac, n_ex, a_ex, n_miss, a_miss, # 0-5
        n_char, a_char, n_cont, a_cont, num_agr, num_con, # 6-11
        a_num_both, a_num_one, a_num_none, a_num_exact, a_num_partial, a_num_unshared_ref, a_num_unshared_tgt, # 12-18
        int_n_jac_a_miss, int_n_char_a_miss, int_n_cont_a_miss # 19-21
    ]

# --- Core Evaluation ---
def score_predictions(preds_dict, truth_dict, req_refs, country_dict=None):
    if any(r not in truth_dict for r in req_refs):
        raise ValueError("Missing truth for explicit references.")
        
    scores, country_scores = [], defaultdict(list)
    tp_sum, fp_sum, fn_sum = 0, 0, 0
    singletons, singleton_fps = 0, 0
    
    for ref in req_refs:
        g_set = truth_dict[ref]
        p_set = preds_dict.get(ref, set())
        
        g, t, fp = len(g_set), len(p_set & g_set), len(p_set - g_set)
        fn = g - t
        tp_sum += t; fp_sum += fp; fn_sum += fn
        c = country_dict.get(ref, 'Unknown') if country_dict else 'Unknown'
        
        if g == 0:
            singletons += 1
            if fp > 0:
                singleton_fps += 1
                scores.append(0.0); country_scores[c].append(0.0)
            else:
                scores.append(1.0); country_scores[c].append(1.0)
        else:
            if t == 0:
                scores.append(0.0); country_scores[c].append(0.0)
            else:
                s = (1.25 * t) / (1.25 * t + fp + 0.25 * fn)
                scores.append(s); country_scores[c].append(s)
                
    macro = sum(scores) / len(scores) if scores else 0.0
    c_macro = {c: (sum(s)/len(s) if s else 0.0) for c, s in country_scores.items()}
    return {
        'macro_f05': macro, 'country_macro_f05': c_macro,
        'tp': tp_sum, 'fp': fp_sum, 'fn': fn_sum,
        'singleton_fp_rate': singleton_fps / singletons if singletons else 0.0
    }

def load_truth(ref_list):
    ref_set = set(ref_list)
    truth = {}
    with open(TRUTH_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split('\t')
            if parts[0] in ref_set:
                truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
    return truth

# --- Retrieval & Extraction Logic ---
def retrieve_for_country(country, db_path, ref_ids):
    conn = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
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

    cur.execute(f"SELECT entity_id, name_raw, address_raw, name_norm, address_norm FROM records WHERE source=1 AND country=? AND entity_id IN ({','.join(['?']*len(ref_ids))})", [country] + list(ref_ids))
    candidates_dict = {}
    for ref in cur:
        route_a = set(exact_n_index.get(ref['name_norm'], [])) & set(exact_a_index.get(ref['address_norm'], [])) if (ref['name_norm'] and ref['address_norm']) else set()
        
        def get_top_k(text, idx_dict, idf_dict):
            surv = [(idf_dict[t], t) for t in tokenize(text) if t in idf_dict]
            surv.sort(key=lambda x: (-x[0], x[1]))
            scores = defaultdict(float)
            for idf, t in surv[:M_TOKENS]:
                for tgt_int in idx_dict[t]: scores[tgt_int] += idf
            return [target_int_to_str[tgt] for tgt, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:50]]

        cands = list(route_a.union(get_top_k(ref['name_raw'], name_index, name_idf)).union(get_top_k(ref['address_raw'], addr_index, addr_idf)))
        candidates_dict[ref['entity_id']] = cands
        
    del name_index, addr_index, exact_n_index, exact_a_index
    gc.collect()
    return candidates_dict

def get_features_for_candidates(db_path, candidates_dict):
    conn = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    
    refs = {}
    ref_list = list(candidates_dict.keys())
    for i in range(0, len(ref_list), 999):
        batch = ref_list[i:i+999]
        cur = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for r in cur: refs[r['entity_id']] = dict(r)
        
    cand_to_refs = defaultdict(list)
    for ref_id, cands in candidates_dict.items():
        for cand_id in cands: cand_to_refs[cand_id].append(ref_id)
            
    X_dict, pair_ids = defaultdict(list), defaultdict(list)
    targets = {}
    all_tgt_ids = list(cand_to_refs.keys())
    
    for i in range(0, len(all_tgt_ids), 999):
        batch_tgts = all_tgt_ids[i:i+999]
        cur = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch_tgts))})", batch_tgts)
        for tgt_row in cur:
            tgt_id = tgt_row['entity_id']
            targets[tgt_id] = dict(tgt_row)
            for ref_id in cand_to_refs[tgt_id]:
                X_dict[ref_id].append(extract_all_features(refs[ref_id], tgt_row))
                pair_ids[ref_id].append(tgt_id)
                
    conn.close()
    return X_dict, pair_ids, refs, targets

def tune_threshold(clf, X_dict, pairs_dict, truth, ref_ids, country_dict, feature_slice):
    best_t, best_f05 = 0.0, 0.0
    probs = {}
    for r in ref_ids:
        if not X_dict[r]:
            probs[r] = []
        else:
            feats = [x[:feature_slice] for x in X_dict[r]]
            probs[r] = clf.predict_proba(feats)[:, 1]
            
    for t in np.arange(0.0, 1.05, 0.05):
        preds = {r: {pairs_dict[r][i] for i, p in enumerate(probs[r]) if p >= t} for r in ref_ids}
        res = score_predictions(preds, truth, ref_ids, country_dict)
        if res['macro_f05'] > best_f05:
            best_t, best_f05 = t, res['macro_f05']
            
    return round(best_t, 2), best_f05, probs

def main():
    t0 = time.time()
    train_ids = json.loads(TRAIN_IDS.read_text())
    tune_ids = json.loads(DEV_TUNE_IDS.read_text())
    eval_ids = json.loads(DEV_EVAL_IDS.read_text())
    
    conn_dev = sqlite3.connect(DB_DEV.as_uri() + '?mode=ro', uri=True)
    country_map = {r[0]: r[1] for r in conn_dev.execute("SELECT entity_id, country FROM records WHERE source=1")}
    conn_dev.close()
    
    print("Retrieving candidates...")
    train_cands, tune_cands, eval_cands = {}, {}, {}
    for country in ['US', 'India']:
        train_cands.update(retrieve_for_country(country, DB_TRAIN, train_ids))
        tune_cands.update(retrieve_for_country(country, DB_DEV, tune_ids))
        eval_cands.update(retrieve_for_country(country, DB_DEV, eval_ids))
        
    print("Generating Features...")
    fg_t0 = time.time()
    X_train, pairs_train, _, _ = get_features_for_candidates(DB_TRAIN, train_cands)
    X_tune, pairs_tune, tune_refs, tune_tgts = get_features_for_candidates(DB_DEV, tune_cands)
    X_eval, pairs_eval, eval_refs, eval_tgts = get_features_for_candidates(DB_DEV, eval_cands)
    fg_time = time.time() - fg_t0
    print(f"Feature generation completed in {fg_time:.1f}s.")
    
    truth_train = load_truth(train_ids)
    
    X_train_v2, X_train_v3n, X_train_v3f, y_train = [], [], [], []
    for ref, feats in X_train.items():
        g_set = truth_train[ref]
        for idx, cand in enumerate(pairs_train[ref]):
            X_train_v2.append(feats[idx][:12])
            X_train_v3n.append(feats[idx][:19])
            X_train_v3f.append(feats[idx][:22])
            y_train.append(1 if cand in g_set else 0)
            
    print("Fitting Logistic Regressions...")
    clf_v2 = LogisticRegression(class_weight=None, max_iter=1000, random_state=42).fit(X_train_v2, y_train)
    clf_v3n = LogisticRegression(class_weight=None, max_iter=1000, random_state=42).fit(X_train_v3n, y_train)
    clf_v3f = LogisticRegression(class_weight=None, max_iter=1000, random_state=42).fit(X_train_v3f, y_train)
    
    print("Tuning Thresholds on TUNING sample...")
    truth_tune = load_truth(tune_ids)
    t_v2, f05_v2, _ = tune_threshold(clf_v2, X_tune, pairs_tune, truth_tune, tune_ids, country_map, 12)
    t_v3n, f05_v3n, _ = tune_threshold(clf_v3n, X_tune, pairs_tune, truth_tune, tune_ids, country_map, 19)
    t_v3f, f05_v3f, probs_v3f = tune_threshold(clf_v3f, X_tune, pairs_tune, truth_tune, tune_ids, country_map, 22)
    
    print(f"V2 (12 feats): {f05_v2:.4f} (t={t_v2})")
    print(f"V3 Numeric (19 feats): {f05_v3n:.4f} (t={t_v3n})")
    print(f"V3 Full (22 feats): {f05_v3f:.4f} (t={t_v3f})")
    
    # Model Selection
    if f05_v3f > f05_v2 + 0.002: # Requires a convincing improvement to adopt
        best_clf, best_t, best_slice, best_name = clf_v3f, t_v3f, 22, "V3 Full"
    elif f05_v3n > f05_v2 + 0.002:
        best_clf, best_t, best_slice, best_name = clf_v3n, t_v3n, 19, "V3 Numeric"
    else:
        best_clf, best_t, best_slice, best_name = clf_v2, t_v2, 12, "V2 Baseline"
        
    print(f"\nSelected Model: {best_name}")
    
    # --- Tuning Error Analysis (Missing Addr & Numeric Conflict) ---
    tune_errors_out = {'missing_address': [], 'numeric_disagreement': []}
    tune_preds = {r: {pairs_tune[r][i] for i, p in enumerate(probs_v3f[r]) if p >= t_v3f} for r in tune_ids}
    
    for r in tune_ids:
        g_set, p_set = truth_tune[r], tune_preds[r]
        for fp in (p_set - g_set):
            feats = extract_all_features(tune_refs[r], tune_tgts[fp])
            err_dict = {"type": "FP", "ref": r, "target": fp, "ref_name": tune_refs[r]['name_raw'], "tgt_name": tune_tgts[fp]['name_raw'], "ref_addr": tune_refs[r]['address_raw'], "tgt_addr": tune_tgts[fp]['address_raw']}
            if feats[5] == 1.0 and len(tune_errors_out['missing_address']) < 5:
                tune_errors_out['missing_address'].append(err_dict)
            if feats[12] == 1.0 and feats[15] == 0.0 and len(tune_errors_out['numeric_disagreement']) < 5:
                tune_errors_out['numeric_disagreement'].append(err_dict)
                
        for fn in (g_set - p_set):
            if fn in tune_tgts:
                feats = extract_all_features(tune_refs[r], tune_tgts[fn])
                err_dict = {"type": "FN", "ref": r, "target": fn, "ref_name": tune_refs[r]['name_raw'], "tgt_name": tune_tgts[fn]['name_raw'], "ref_addr": tune_refs[r]['address_raw'], "tgt_addr": tune_tgts[fn]['address_raw']}
                if feats[5] == 1.0 and len(tune_errors_out['missing_address']) < 5:
                    tune_errors_out['missing_address'].append(err_dict)
                if feats[12] == 1.0 and feats[15] == 0.0 and len(tune_errors_out['numeric_disagreement']) < 5:
                    tune_errors_out['numeric_disagreement'].append(err_dict)

    with open(V3_ERRORS_TUNE, 'w') as f: json.dump(tune_errors_out, f, indent=2)

    # --- Secondary Dev Evaluation ---
    truth_eval = load_truth(eval_ids)
    preds, exp_preds = {}, {}
    for r in eval_ids:
        if not X_eval[r]:
            preds[r], exp_preds[r] = set(), []
        else:
            feats = [x[:best_slice] for x in X_eval[r]]
            probs = best_clf.predict_proba(feats)[:, 1]
            preds[r] = {pairs_eval[r][i] for i, p in enumerate(probs) if p >= best_t}
            exp_preds[r] = list(preds[r])
            
    res_eval = score_predictions(preds, truth_eval, eval_ids, country_map)
    
    with open(V3_MODEL_PATH, 'wb') as f: pickle.dump(best_clf, f)
    with open(V3_RESULTS, 'w') as f:
        json.dump({'model': best_name, 'threshold': best_t, 'results': res_eval, 'predictions': exp_preds}, f, indent=2)

    print("\n| Metric | Secondary Dev Evaluation |")
    print(f"| :--- | :--- |")
    print(f"| Macro F0.5 | {res_eval['macro_f05']:.4f} |")
    print(f"| US Macro F0.5 | {res_eval['country_macro_f05'].get('US', 0):.4f} |")
    print(f"| India Macro F0.5 | {res_eval['country_macro_f05'].get('India', 0):.4f} |")
    print(f"| TP / FP / FN | {res_eval['tp']} / {res_eval['fp']} / {res_eval['fn']} |")
    print(f"| Singleton FP Rate | {res_eval['singleton_fp_rate']:.4f} |")
    
    print(f"\n[Peak RAM: {get_peak_ram_mb():.1f} MB] Completed in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()