import sqlite3, json, time, sys, hashlib, inspect
import numpy as np
from pathlib import Path
from collections import defaultdict

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

from business_entity_resolution.inference_features import values, features
sys.path.insert(0, str(ROOT / '.gpu_runtime'))
from catboost import CatBoostClassifier
import business_entity_resolution.retrieval_v5 as ret_module
from business_entity_resolution.retrieval_v5 import CountryIndexV5, RetrievalConfig

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def compute_query_f05(preds_set, truth_set):
    # Standard Entity Resolution Query-Level Metric
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
    print("=== 2K BOUNDED LEGACY CANDIDATE VALIDATION ===")
    t_start = time.time()
    
    # 1. VERIFY INPUTS & CONFIRMATION SPLIT
    CONF_DIR = ROOT / 'reports/luna_export/confirmation'
    ref_ids = json.loads((CONF_DIR / 'reference_ids.json').read_text())
    assert len(ref_ids) == 2000, f"Expected 2000 ref IDs, got {len(ref_ids)}"
    assert len(set(ref_ids)) == 2000, "Duplicate references found in confirmation set!"
    
    truth = {}
    with open(CONF_DIR / 'truth.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            truth[d['reference_id']] = set(d['true_ids'])
            
    assert len(truth) == 2000, f"Expected 2000 truth records, got {len(truth)}"
    
    DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
    DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
    conn = sqlite3.connect(f"file:{DB_DEV}?mode=ro", uri=True)
    conn.execute(f"ATTACH DATABASE '{DB_TRAIN}' AS train_db")
    conn.row_factory = sqlite3.Row
    
    queries = []
    countries = defaultdict(list)
    singletons_count = 0
    for rid in ref_ids:
        if rid not in truth:
            raise ValueError(f"FATAL: Missing explicit truth record for {rid}!")
        row = conn.execute("SELECT * FROM main.records WHERE entity_id=?", (rid,)).fetchone()
        if not row:
            raise ValueError(f"FATAL: Missing reference text for DEV query {rid} in main.records!")
        q = dict(row)
        queries.append(q)
        countries[q['country']].append(q)
        if len(truth[rid]) == 0:
            singletons_count += 1
            
    print(f"Verified 2,000 unique DEV queries ({singletons_count} true singletons) across {len(countries)} countries.")

    # 2. LOCATE & VERIFY MODELS
    cbm_candidates = list(ROOT.rglob('*.cbm'))
    orig_path = next((p for p in cbm_candidates if 'gpu_challenger_v2' not in str(p)), None)
    repaired_path = ROOT / 'artifacts/gpu_challenger_v2/model.cbm'
    
    assert orig_path and orig_path.exists(), f"Could not find baseline model in artifacts!"
    assert repaired_path.exists(), f"Could not find repaired model at {repaired_path}!"
    
    hash_orig = get_sha256(orig_path)
    hash_repaired = get_sha256(repaired_path)
    print(f"Model A (Original Preserved): {orig_path} (SHA-256: {hash_orig[:16]}...)")
    print(f"Model B (Repaired Luna):      {repaired_path} (SHA-256: {hash_repaired[:16]}...)")
    
    clf_a = CatBoostClassifier().load_model(str(orig_path))
    clf_b = CatBoostClassifier().load_model(str(repaired_path))

    # 3. CONFIGURE EXACT LEGACY RETRIEVAL (DF5000, top50 union, no joint)
    cfg_params = inspect.signature(RetrievalConfig).parameters
    kwargs = {'max_df_abs': 5000, 'top_k_cands': 50}
    if 'route_exact_name' in cfg_params: kwargs['route_exact_name'] = False
    if 'route_exact_address' in cfg_params: kwargs['route_exact_address'] = False
    if 'route_joint' in cfg_params: kwargs['route_joint'] = False
    if 'union_baseline' in cfg_params: kwargs['union_baseline'] = True
    
    legacy_cfg = RetrievalConfig(**kwargs)
    print(f"Retrieval Configuration: {legacy_cfg.__dict__}")

    # 4. RETRIEVE & SCORE
    per_query_scores = {'A': {}, 'B': {}}
    country_metrics = defaultdict(lambda: {'A': [], 'B': [], 'oracle': []})
    totals = {
        'A': {'tp': 0, 'fp': 0, 'fn': 0, 'singleton_err': 0},
        'B': {'tp': 0, 'fp': 0, 'fn': 0, 'singleton_err': 0}
    }
    
    oracle_retrieved, oracle_targets = 0, 0
    saved_cands = {}
    
    target_cache = {}
    def get_target(eid):
        if eid not in target_cache:
            row = conn.execute("SELECT * FROM main.records WHERE entity_id=?", (eid,)).fetchone()
            if not row: row = conn.execute("SELECT * FROM train_db.records WHERE entity_id=?", (eid,)).fetchone()
            target_cache[eid] = dict(row)
        return target_cache[eid]

    for country, c_queries in countries.items():
        print(f"\nProcessing Country: {country} ({len(c_queries)} queries)...")
        idx = CountryIndexV5(conn, country, legacy_cfg, include_train_db=True)
        
        for q in c_queries:
            eid = q['entity_id']
            true_set = truth[eid]
            cands = idx.retrieve(q)
            saved_cands[eid] = cands
            
            # Oracle Check
            if len(true_set) > 0:
                oracle_targets += len(true_set)
                hits = len(set(cands) & true_set)
                oracle_retrieved += hits
                country_metrics[country]['oracle'].append(hits / len(true_set))
            else:
                country_metrics[country]['oracle'].append(1.0)
                
            if not cands:
                f_a, p_a, r_a, tp_a, fp_a, fn_a = compute_query_f05(set(), true_set)
                f_b, p_b, r_b, tp_b, fp_b, fn_b = compute_query_f05(set(), true_set)
                preds_a, preds_b = set(), set()
            else:
                X = [features(values(q), values(get_target(c))) for c in cands]
                X_mat = np.array(X, dtype=np.float32)
                probs_a = clf_a.predict_proba(X_mat)[:, 1]
                probs_b = clf_b.predict_proba(X_mat)[:, 1]
                
                preds_a = set(cands[i] for i, p in enumerate(probs_a) if p >= 0.362)
                preds_b = set(cands[i] for i, p in enumerate(probs_b) if p >= 0.665)
                
                f_a, p_a, r_a, tp_a, fp_a, fn_a = compute_query_f05(preds_a, true_set)
                f_b, p_b, r_b, tp_b, fp_b, fn_b = compute_query_f05(preds_b, true_set)
                
            per_query_scores['A'][eid] = f_a
            per_query_scores['B'][eid] = f_b
            country_metrics[country]['A'].append(f_a)
            country_metrics[country]['B'].append(f_b)
            
            totals['A']['tp'] += tp_a; totals['A']['fp'] += fp_a; totals['A']['fn'] += fn_a
            totals['B']['tp'] += tp_b; totals['B']['fp'] += fp_b; totals['B']['fn'] += fn_b
            
            if len(true_set) == 0:
                if len(preds_a) > 0: totals['A']['singleton_err'] += 1
                if len(preds_b) > 0: totals['B']['singleton_err'] += 1

    conn.close()

    # 5. AGGREGATE RESULTS
    macro_a = np.mean(list(per_query_scores['A'].values()))
    macro_b = np.mean(list(per_query_scores['B'].values()))
    oracle_macro = oracle_retrieved / oracle_targets if oracle_targets > 0 else 1.0
    
    diffs = [per_query_scores['B'][eid] - per_query_scores['A'][eid] for eid in ref_ids]
    wins_b = sum(1 for d in diffs if d > 1e-5)
    wins_a = sum(1 for d in diffs if d < -1e-5)
    ties = sum(1 for d in diffs if abs(d) <= 1e-5)
    mean_diff = np.mean(diffs)

    print("\n" + "="*50)
    print("FINAL EVALUATION ON IDENTICAL LEGACY CANDIDATES")
    print("="*50)
    print(f"Candidate Oracle Recall: {oracle_macro:.6f} ({oracle_retrieved}/{oracle_targets} targets)")
    print(f"\nModel A (Original @ 0.362): Actual Macro F0.5 = {macro_a:.6f}")
    print(f"  Totals -> TP: {totals['A']['tp']} | FP: {totals['A']['fp']} | FN: {totals['A']['fn']}")
    print(f"  Singleton Errors: {totals['A']['singleton_err']} / {singletons_count}")
    
    print(f"\nModel B (Repaired @ 0.665): Actual Macro F0.5 = {macro_b:.6f}")
    print(f"  Totals -> TP: {totals['B']['tp']} | FP: {totals['B']['fp']} | FN: {totals['B']['fn']}")
    print(f"  Singleton Errors: {totals['B']['singleton_err']} / {singletons_count}")
    
    print(f"\nPer-Query Comparison (Model B minus Model A):")
    print(f"  Mean Delta: {mean_diff:+.6f}")
    print(f"  Wins (Repaired B > Orig A): {wins_b} queries")
    print(f"  Losses (Orig A > Repaired B): {wins_a} queries")
    print(f"  Ties: {ties} queries")
    
    print(f"\nCountry Breakdown (Actual Macro F0.5):")
    for c in sorted(country_metrics.keys()):
        c_a = np.mean(country_metrics[c]['A'])
        c_b = np.mean(country_metrics[c]['B'])
        print(f"  {c:<10} -> Orig A: {c_a:.6f} | Repaired B: {c_b:.6f} | Delta: {c_b - c_a:+.6f}")
        
    print(f"\nTotal Elapsed Time: {time.time()-t_start:.1f}s")
    
    report = {
        'candidate_oracle': float(oracle_macro),
        'model_a': {'threshold': 0.362, 'macro_f05': float(macro_a), 'totals': totals['A']},
        'model_b': {'threshold': 0.665, 'macro_f05': float(macro_b), 'totals': totals['B']},
        'delta': {'mean': float(mean_diff), 'wins_b': wins_b, 'wins_a': wins_a, 'ties': ties},
        'country_breakdown': {c: {'A': float(np.mean(country_metrics[c]['A'])), 'B': float(np.mean(country_metrics[c]['B']))} for c in country_metrics}
    }
    report_path = ROOT / 'reports/legacy_2k_evaluation_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved evaluation report to: {report_path}")

if __name__ == '__main__':
    main()
