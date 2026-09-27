import sqlite3
import json
import time
import sys
import gc
from collections import defaultdict
from pathlib import Path

# Provide Windows/Linux compatible memory check
try:
    import resource
    def get_peak_ram_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:
    import psutil
    def get_peak_ram_mb(): return psutil.Process().memory_info().rss / (1024 * 1024)

# --- Paths ---
cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

from business_entity_resolution.retrieval_v4 import RetrievalConfig, CountryIndexV4
from business_entity_resolution.workspace import DATASET_DIR

DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
DB_SPLIT = ROOT / 'data/processed/phase2_split.sqlite'
TRUTH_PATH = DATASET_DIR / 'train/train_ground_truth.tsv'

OUT_SAMPLE = ROOT / 'reports/retrieval_v4_sample_ids.json'
OUT_REPORT = ROOT / 'reports/retrieval_v4_diagnostic.json'
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

# --- Metric Logic ---
def score_macro(preds_dict, truth_dict, req_refs):
    scores = []
    tp_s, fp_s, fn_s, singletons, singleton_fps = 0, 0, 0, 0, 0
    c_counts = []
    
    for ref in req_refs:
        g_set = truth_dict.get(ref, set())
        p_set = preds_dict.get(ref, set())
        c_counts.append(len(p_set))
        
        g = len(g_set)
        t = len(p_set & g_set)
        fp = len(p_set - g_set)
        fn = g - t
        tp_s += t; fp_s += fp; fn_s += fn
        
        if g == 0:
            singletons += 1
            if fp > 0:
                singleton_fps += 1
                scores.append(0.0)
            else:
                scores.append(1.0)
        else:
            if t == 0: scores.append(0.0)
            else: scores.append((1.25 * t) / (1.25 * t + fp + 0.25 * fn))
            
    c_counts.sort()
    return {
        'macro': sum(scores) / len(scores) if scores else 0.0,
        'tp': tp_s, 'fp': fp_s, 'fn': fn_s,
        'singleton_fp_rate': singleton_fps / singletons if singletons else 0.0,
        'cand_med': c_counts[len(c_counts)//2] if c_counts else 0,
        'cand_p95': c_counts[int(len(c_counts)*0.95)] if c_counts else 0
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

def synthetic_self_check():
    print("--- Running Synthetic Self-Check ---")
    conn = sqlite3.connect(':memory:')
    conn.execute("CREATE TABLE records (entity_id TEXT, source INTEGER, country TEXT, name_raw TEXT, address_raw TEXT, name_norm TEXT, address_norm TEXT)")
    conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", [
        ("T1", 2, "US", "Alpha Corp", "123 Main St", "alpha corp", "123 main st"),
        ("T2", 3, "US", "Alpha Corporation", "123 Main Street", "alpha corporation", "123 main street"),
    ])
    
    cfg = RetrievalConfig(route_exact_name=True)
    idx = CountryIndexV4(conn, "US", cfg, include_train_db=False)
    
    res, _ = idx.retrieve({'name_raw': 'Alpha Corp', 'name_norm': 'alpha corp', 'address_raw': 'Unknown', 'address_norm': 'unknown'})
    assert "T1" in res, "Self-check failed: Exact bucket did not retrieve T1."
    print("Synthetic self-check PASSED.")
    conn.close()

def main():
    if '--self-check' in sys.argv:
        synthetic_self_check()
        return

    t_start = time.time()
    
    print("--- 1. Generating Deterministic Dev Sample ---")
    exclude_files = ['phase4_train_sample_ids.json', 'phase4_dev_tuning_ids.json', 'phase3_pilot_sample_ids.json']
    exclusions = set()
    for f in exclude_files:
        p = ROOT / 'data/processed' / f
        if p.exists(): exclusions.update(json.loads(p.read_text()))
        
    conn_split = sqlite3.connect(DB_SPLIT.as_uri() + '?mode=ro', uri=True)
    cur = conn_split.execute("SELECT reference_id, country FROM reference_assignments WHERE split='dev' ORDER BY reference_id")
    eligible = [r for r in cur.fetchall() if r[0] not in exclusions]
    conn_split.close()
    
    import random
    random.seed(20260927)
    sample_data = random.sample(eligible, 2000)
    sample_ids = [r[0] for r in sample_data]
    with open(OUT_SAMPLE, 'w') as f: json.dump(sample_ids, f)
    print(f"Sampled 2000 queries (ex: {len(exclusions)} previous IDs).")

    conn_dev = sqlite3.connect(DB_DEV.as_uri() + '?mode=ro', uri=True)
    conn_dev.row_factory = sqlite3.Row
    refs_by_country = defaultdict(list)
    for i in range(0, 2000, 999):
        batch = sample_ids[i:i+999]
        cur = conn_dev.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for r in cur: refs_by_country[r['country']].append(dict(r))
        
    truth = load_truth(sample_ids)
    
    # Ablations - Focusing purely on Retrieval
    configs = {
        'Baseline': RetrievalConfig(),
        'A_ExactBuckets': RetrievalConfig(route_exact_name=True, route_exact_address=True),
        'B_JointScoring': RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True),
        'C1_MaxDF_20k': RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000),
        'C2_TopK_100': RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, top_k_cands=100)
    }

    report = {'configs': {}}
    
    print(f"--- 2. Attaching Databases for Large-Pool Retrieval ---")
    conn_dev.execute(f"ATTACH DATABASE '{DB_TRAIN}' AS train_db")

    for name, cfg in configs.items():
        print(f"\nEvaluating Retrieval Oracle: {name}...")
        t_cfg = time.time()
        
        preds_oracle = {}
        retrieval_fns = []
        source_dist = {'S2': 0, 'S3': 0}
        
        for country, c_refs in refs_by_country.items():
            print(f"  [{get_peak_ram_mb():.1f} MB] Indexing {country} pool...")
            idx = CountryIndexV4(conn_dev, country, cfg, include_train_db=True)
            
            for r in c_refs:
                eid = r['entity_id']
                g_set = truth.get(eid, set())
                cands, sources = idx.retrieve(r)
                
                for s in sources:
                    if s == 2: source_dist['S2'] += 1
                    elif s == 3: source_dist['S3'] += 1
                
                c_set = set(cands)
                preds_oracle[eid] = c_set & g_set
                
                missed = g_set - c_set
                if missed and len(retrieval_fns) < 10:
                    retrieval_fns.append({
                        "query": eid, "missed_target": list(missed)[0], 
                        "route_name": name, "q_name": r['name_norm'], "q_addr": r['address_norm']
                    })
                    
            del idx
            gc.collect()

        res_oracle = score_macro(preds_oracle, truth, sample_ids)
        
        report['configs'][name] = {
            'oracle': res_oracle,
            'source_distribution': source_dist,
            'retrieval_fn_examples': retrieval_fns,
            'runtime_sec': round(time.time() - t_cfg, 1)
        }
        
        print(f"  Oracle Macro (Ceiling): {res_oracle['macro']:.4f}")
        print(f"  Oracle TP: {res_oracle['tp']} | FN (Missed True Links): {res_oracle['fn']}")
        
    conn_dev.close()
    
    with open(OUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n--- Diagnostic Complete in {time.time()-t_start:.1f}s ---")
    print(f"Peak RAM: {get_peak_ram_mb():.1f} MB")
    print(f"Report written to: {OUT_REPORT}")

if __name__ == "__main__":
    main()