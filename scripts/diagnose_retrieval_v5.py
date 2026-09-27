import sqlite3, json, time, sys, gc, statistics, hashlib, pickle
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

try:
    import resource
    def get_peak_ram_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    import psutil
    def get_peak_ram_mb(): return psutil.Process().memory_info().rss / (1024 * 1024)

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

from business_entity_resolution.retrieval_v5 import RetrievalConfig, CountryIndexV5
from business_entity_resolution.workspace import DATASET_DIR
from business_entity_resolution.inference_features import values, features

# Attempt CatBoost, but do not fail if missing. We will export the eval packet regardless.
try:
    sys.path.insert(0, str(ROOT / '.gpu_runtime'))
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
TRUTH_PATH = DATASET_DIR / 'train/train_ground_truth.tsv'
MODEL_PATH = ROOT / 'artifacts/gpu_challenger/model.cbm'
OUT_SAMPLE = ROOT / 'reports/retrieval_v4_sample_ids.json'
OUT_REPORT = ROOT / 'reports/retrieval_v5_diagnostic_results.json'
OUT_PACKET = ROOT / 'reports/v5_eval_packet.pkl'

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def score_metrics(cands_dict, preds_dict, truth_dict, req_refs):
    total_true, tp_cands, zero_retrieved = 0, 0, 0
    c_counts = []
    m_scores, tp_m, fp_m, fn_m, m_singles, m_single_fps = [], 0, 0, 0, 0, 0
    o_scores = []
    
    for r in req_refs:
        g = truth_dict.get(r, set())
        c = set(cands_dict.get(r, [])) # Convert the verified list to a set for math
        p = preds_dict.get(r, set())
        
        c_counts.append(len(c))
        total_true += len(g)
        tp_cands += len(c & g)
        if len(g) > 0 and len(c & g) == 0: zero_retrieved += 1
        
        o_fp, o_fn = 0, len(g) - len(c & g)
        if len(g) == 0: o_scores.append(1.0)
        elif len(c & g) == 0: o_scores.append(0.0)
        else: o_scores.append((1.25 * len(c & g)) / (1.25 * len(c & g) + o_fp + 0.25 * o_fn))
        
        if HAS_CATBOOST:
            m_t, m_fp, m_fn = len(p & g), len(p - g), len(g) - len(p & g)
            tp_m += m_t; fp_m += m_fp; fn_m += m_fn
            if len(g) == 0:
                m_singles += 1
                if m_fp > 0: m_single_fps += 1; m_scores.append(0.0)
                else: m_scores.append(1.0)
            else:
                if m_t == 0: m_scores.append(0.0)
                else: m_scores.append((1.25 * m_t) / (1.25 * m_t + m_fp + 0.25 * m_fn))
            
    c_counts.sort()
    res = {
        'cands': {'total': sum(c_counts), 'mean': statistics.mean(c_counts) if c_counts else 0, 'median': c_counts[len(c_counts)//2] if c_counts else 0, 'p95': c_counts[int(len(c_counts)*0.95)] if c_counts else 0, 'max': c_counts[-1] if c_counts else 0},
        'link_recall': tp_cands / total_true if total_true else 0.0,
        'zero_retrieved_positive_queries': zero_retrieved,
        'oracle_macro': statistics.mean(o_scores) if o_scores else 0.0
    }
    if HAS_CATBOOST:
        res['model'] = {'macro': statistics.mean(m_scores) if m_scores else 0.0, 'tp': tp_m, 'fp': fp_m, 'fn': fn_m, 'singleton_fp_rate': m_single_fps/m_singles if m_singles else 0.0}
    return res

def main():
    print(f"--- 1. Validation & Setup ---")
    sample_ids = json.loads(OUT_SAMPLE.read_text())
    sample_set = set(sample_ids)
    assert len(sample_ids) == len(sample_set), "CRITICAL: Sample IDs are not unique!"
    
    conn_dev = sqlite3.connect(DB_DEV.as_uri() + '?mode=ro', uri=True)
    conn_dev.row_factory = sqlite3.Row
    conn_dev.execute(f"ATTACH DATABASE '{DB_TRAIN}' AS train_db")
    
    refs_by_country = defaultdict(list)
    found_refs = 0
    for i in range(0, len(sample_ids), 999):
        batch = sample_ids[i:i+999]
        cur = conn_dev.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for r in cur: 
            refs_by_country[r['country']].append(dict(r))
            found_refs += 1
            
    assert found_refs == len(sample_ids), f"CRITICAL: Only retrieved {found_refs}/{len(sample_ids)} reference rows."
    
    truth = {}
    with open(TRUTH_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split('\t')
            if parts[0] in sample_set:
                truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
                
    missing_truth = sample_set - set(truth.keys())
    if missing_truth: raise ValueError(f"CRITICAL: Missing ground truth for {len(missing_truth)} queries!")

    configs = {
        'Combined_DF20k_K100': RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000, top_k_cands=100),
        'Union_PreserveBase': RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000, top_k_cands=100, union_baseline=True)
    }

    if HAS_CATBOOST: clf = CatBoostClassifier().load_model(str(MODEL_PATH))
    else: print("WARNING: CatBoost not found. Skipping local model scoring.")

    # Shared Cache for Eval Packet
    target_cache = {}
    def get_target(eid):
        if eid not in target_cache:
            row = conn_dev.execute("SELECT * FROM main.records WHERE entity_id=?", (eid,)).fetchone()
            if not row: row = conn_dev.execute("SELECT * FROM train_db.records WHERE entity_id=?", (eid,)).fetchone()
            target_cache[eid] = dict(row)
        return target_cache[eid]

    report = {
        'metadata': {
            'code_hash': get_file_hash(ROOT / 'src/business_entity_resolution/retrieval_v5.py'),
            'query_count': len(sample_ids),
            'target_counts': {}
        },
        'configs': {}
    }
    
    # Initialize Eval Packet Structure
    eval_packet = {
        'queries': {r['entity_id']: {'country': c, 'truth': list(truth[r['entity_id']])} for c, refs in refs_by_country.items() for r in refs},
        'candidates': defaultdict(dict),
        'features': defaultdict(dict)
    }

    print(f"\n--- 2. Running Ablations ---")
    for name, cfg in configs.items():
        print(f"\nEvaluating {name}...")
        t_cfg = time.time()
        cands_dict, preds_dict = {}, {}
        
        for country, c_refs in refs_by_country.items():
            t_idx = time.time()
            idx = CountryIndexV5(conn_dev, country, cfg, include_train_db=True)
            report['metadata']['target_counts'][country] = len(idx.target_int_to_str)
            
            for r in c_refs:
                eid = r['entity_id']
                cands = idx.retrieve(r)
                assert isinstance(cands, list), f"Retrieve must return list, got {type(cands)}"
                
                cands_dict[eid] = cands
                eval_packet['candidates'][name][eid] = cands
                
                # Precompute features for the eval packet securely
                ref_v = values(r)
                X_batch = []
                for c in cands:
                    if c not in eval_packet['features'][eid]:
                        f_vec = features(ref_v, values(get_target(c)))
                        eval_packet['features'][eid][c] = f_vec
                    X_batch.append(eval_packet['features'][eid][c])
                
                if HAS_CATBOOST and X_batch:
                    preds_dict[eid] = set([cands[i] for i, p in enumerate(clf.predict_proba(X_batch)[:,1]) if p >= 0.362])
                    
            del idx
            gc.collect()

        res = score_metrics(cands_dict, preds_dict, truth, sample_ids)
        res['timings'] = {'total_sec': round(time.time() - t_cfg, 1)}
        res['memory'] = {'peak_mb': get_peak_ram_mb()}
        res['config_dump'] = asdict(cfg)
        report['configs'][name] = res
        
        print(f"  Oracle Macro: {res['oracle_macro']:.4f} | Link Recall: {res['link_recall']:.4f}")
        if HAS_CATBOOST: print(f"  CatBoost Actual Macro: {res['model']['macro']:.4f}")
        print(f"  Cands (Median/Max): {res['cands']['median']} / {res['cands']['max']} | Build Time: {res['timings']['total_sec']}s")

    with open(OUT_REPORT, 'w') as f: json.dump(report, f, indent=2)
    with open(OUT_PACKET, 'wb') as f: pickle.dump(eval_packet, f)
    
    print(f"\n--- Diagnostic Complete ---")
    print(f"Report: {OUT_REPORT}")
    print(f"Eval Packet for Luna: {OUT_PACKET}")

if __name__ == "__main__": main()