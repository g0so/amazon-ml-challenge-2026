import sqlite3, json, time, sys, random
import numpy as np
from pathlib import Path
from collections import defaultdict

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

from business_entity_resolution.inference_features import values, features
sys.path.insert(0, str(ROOT / '.gpu_runtime'))
from catboost import CatBoostClassifier
from business_entity_resolution.retrieval_v5 import CountryIndexV5, RetrievalConfig

def compute_query_f05(preds_set, truth_set):
    if len(truth_set) == 0:
        return (1.0, 1.0, 1.0, 0, len(preds_set), 0) if len(preds_set) == 0 else (0.0, 0.0, 0.0, 0, len(preds_set), 0)
    tp = len(preds_set & truth_set)
    fp = len(preds_set - truth_set)
    fn = len(truth_set - preds_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
    return (f, p, r, tp, fp, fn)

def main():
    print("=== COUNTRY POLICY HYBRID EVALUATION ===")
    t_start = time.time()
    
    # Safely Load Truth from available sources
    truth = {}
    for p in [ROOT / 'dataset/train/train_ground_truth.tsv']:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if len(parts) > 0:
                        truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
                        
    conf_truth = ROOT / 'reports/luna_export/confirmation/truth.jsonl'
    if conf_truth.exists():
        with open(conf_truth, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                truth[d['reference_id']] = set(d['true_ids'])
                
    # Identify previously inspected IDs to build TUNE set and avoid leakage
    inspected_ids = set()
    for p in (ROOT / 'artifacts').rglob('*.jsonl'):
        with open(p, 'r') as f:
            for line in f:
                try: inspected_ids.add(json.loads(line)['reference_id'])
                except: pass
    for p in (ROOT / 'reports').rglob('*.json'):
        if 'reference_ids.json' in p.name:
            try: inspected_ids.update(json.loads(p.read_text()))
            except: pass
            
    # Split Sets
    tune_ids = [rid for rid in truth if rid in inspected_ids][:1500] 
    available_fresh = [rid for rid in truth if rid not in inspected_ids]
    
    random.seed(42)
    fresh_dev_ids = random.sample(available_fresh, min(2000, len(available_fresh)))
    
    print(f"Loaded {len(tune_ids)} existing inspected queries for TUNE.")
    print(f"Sampled {len(fresh_dev_ids)} disjoint, unseen queries for FRESH DEV.")
    
    DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
    DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
    conn = sqlite3.connect(f"file:{DB_DEV}?mode=ro", uri=True)
    conn.execute(f"ATTACH DATABASE '{DB_TRAIN}' AS train_db")
    conn.row_factory = sqlite3.Row
    
    # Models
    orig_path = next((p for p in list(ROOT.rglob('*.cbm')) if 'gpu_challenger_v2' not in str(p)), None)
    repaired_path = ROOT / 'artifacts/gpu_challenger_v2/model.cbm'
    clf_a = CatBoostClassifier().load_model(str(orig_path))
    clf_b = CatBoostClassifier().load_model(str(repaired_path))
    
    # Legacy Config
    kwargs = {'max_df_abs': 5000, 'top_k_cands': 50, 'route_exact_name': False, 'route_exact_address': False, 'route_joint': False, 'union_baseline': True}
    legacy_cfg = RetrievalConfig(**kwargs)

    target_cache = {}
    def get_target(eid):
        if eid not in target_cache:
            row = conn.execute("SELECT * FROM main.records WHERE entity_id=?", (eid,)).fetchone()
            if not row: row = conn.execute("SELECT * FROM train_db.records WHERE entity_id=?", (eid,)).fetchone()
            target_cache[eid] = dict(row) if row else None
        return target_cache[eid]
        
    def process_subset(subset_ids, name):
        print(f"\n--- Retrieving {name} Set ---")
        countries = defaultdict(list)
        for rid in subset_ids:
            row = conn.execute("SELECT * FROM main.records WHERE entity_id=?", (rid,)).fetchone()
            if row: countries[dict(row)['country']].append(dict(row))
            
        results = {}
        for country, c_queries in countries.items():
            if country not in ['US', 'India']: continue # Focus on US/India (No France labels)
            idx = CountryIndexV5(conn, country, legacy_cfg, include_train_db=True)
            results[country] = []
            
            for q in c_queries:
                eid = q['entity_id']
                cands = idx.retrieve(q)
                X, cand_ids = [], []
                for c in cands:
                    tgt = get_target(c)
                    if tgt:
                        X.append(features(values(q), values(tgt)))
                        cand_ids.append(c)
                
                if not X:
                    results[country].append({'eid': eid, 'true': truth[eid], 'probs_a': [], 'probs_b': [], 'cands': []})
                    continue
                    
                probs_a = clf_a.predict_proba(np.array(X, dtype=np.float32))[:, 1]
                probs_b = clf_b.predict_proba(np.array(X, dtype=np.float32))[:, 1]
                results[country].append({'eid': eid, 'true': truth[eid], 'probs_a': probs_a, 'probs_b': probs_b, 'cands': cand_ids})
        return results

    tune_res = process_subset(tune_ids, "TUNE")
    dev_res = process_subset(fresh_dev_ids, "FRESH DEV")
    
    print("\n--- TUNING GRID SEARCH ---")
    thresholds = [0.3, 0.362, 0.4, 0.5, 0.6, 0.665, 0.7, 0.8]
    best_policy = {}
    for country in ['India', 'US']:
        best_f05 = -1
        best_cfg = None
        for m_name, prob_key in [('A', 'probs_a'), ('B', 'probs_b')]:
            for th in thresholds:
                f_list = []
                for q in tune_res.get(country, []):
                    preds = set(q['cands'][i] for i, p in enumerate(q[prob_key]) if p >= th)
                    f_list.append(compute_query_f05(preds, q['true'])[0])
                macro = np.mean(f_list) if f_list else 0
                if macro > best_f05:
                    best_f05 = macro
                    best_cfg = (m_name, th)
        best_policy[country] = best_cfg
        print(f"TUNE: Best for {country} -> Model {best_cfg[0]} @ {best_cfg[1]} (Macro F0.5: {best_f05:.4f})")
        
    print("\n--- FRESH DEV EVALUATION (Disjoint Holdout) ---")
    policies = {
        '1_Current_Submission': {'India': ('B', 0.665), 'US': ('B', 0.665)},
        '2_Simple_Hybrid': {'India': ('A', 0.362), 'US': ('B', 0.665)},
        '3_Tuned_Hybrid': best_policy
    }
    
    for p_name, policy in policies.items():
        print(f"\nPolicy: {p_name} | {policy}")
        t_tp, t_fp, t_fn, t_sing_err, t_sing_true = 0, 0, 0, 0, 0
        f05_all = []
        
        for country in ['India', 'US']:
            if country not in dev_res: continue
            m_name, th = policy[country]
            prob_key = 'probs_a' if m_name == 'A' else 'probs_b'
            c_f05 = []
            for q in dev_res[country]:
                preds = set(q['cands'][i] for i, p in enumerate(q[prob_key]) if p >= th)
                f, p, r, tp, fp, fn = compute_query_f05(preds, q['true'])
                c_f05.append(f)
                f05_all.append(f)
                t_tp += tp; t_fp += fp; t_fn += fn
                if len(q['true']) == 0:
                    t_sing_true += 1
                    if len(preds) > 0: t_sing_err += 1
            print(f"  {country}: Macro F0.5 = {np.mean(c_f05):.4f}")
            
        print(f"  OVERALL Macro F0.5 = {np.mean(f05_all):.4f}")
        print(f"  Totals -> TP: {t_tp} | FP: {t_fp} | FN: {t_fn}")
        print(f"  Singleton Errors: {t_sing_err} / {t_sing_true}")

    print(f"\nElapsed: {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    main()
