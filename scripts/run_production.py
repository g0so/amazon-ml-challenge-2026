import sqlite3, json, time, sys, gc, csv, hashlib
from pathlib import Path

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
from business_entity_resolution.inference_features import values, features

sys.path.insert(0, str(ROOT / '.gpu_runtime'))
from catboost import CatBoostClassifier

# USE THE VERIFIED TEST DB AS REQUESTED
DB_TEST = ROOT / 'data/processed/test_inference_verified.sqlite'
MODEL_PATH = ROOT / 'artifacts/gpu_challenger/model.cbm'
OUTPUT_DIR = ROOT / 'reports/final_submission'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = OUTPUT_DIR / 'resume_manifest.json'
OUT_TSV_1 = OUTPUT_DIR / 'submission_1.tsv'
OUT_TSV_2 = OUTPUT_DIR / 'submission_2.tsv'

# Locked Config based on empirical diagnostic
PROD_CONFIG = RetrievalConfig(
    route_exact_name=True, route_exact_address=True, route_joint=True, 
    max_df_abs=20000, top_k_cands=100, union_baseline=False
)
PROD_THRESHOLD = 0.362
BATCH_SIZE = 5000

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def init_outputs():
    for p in [OUT_TSV_1, OUT_TSV_2]:
        if not p.exists():
            with open(p, 'w', newline='', encoding='utf-8') as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")

def run_pipeline(is_pilot=True):
    print(f"--- 1. Initializing {'PILOT' if is_pilot else 'PRODUCTION'} Pipeline ---")
    if not DB_TEST.exists(): raise FileNotFoundError(f"Missing verified DB at {DB_TEST}")
    
    config_hash = str(hash(str(PROD_CONFIG)))
    model_hash = get_file_hash(MODEL_PATH)
    
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps({'completed_batches': [], 'config_hash': config_hash, 'model_hash': model_hash}))
    manifest = json.loads(MANIFEST.read_text())
    
    if manifest['config_hash'] != config_hash or manifest['model_hash'] != model_hash:
        raise ValueError("CRITICAL: Manifest configuration/model mismatch. Cannot resume safely.")
        
    init_outputs()
    clf = CatBoostClassifier().load_model(str(MODEL_PATH))
    
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM records WHERE source=1 ORDER BY country").fetchall()]
    
    t_start = time.time()
    total_refs_processed = 0
    est_total_time = 0
    
    for country in countries:
        print(f"\n--- Loading {country} ---")
        cur = conn.execute("SELECT COUNT(*) FROM records WHERE source=1 AND country=?", (country,))
        country_total = cur.fetchone()[0]
        if country_total == 0: continue
        
        print(f"[{get_peak_ram_mb():.1f} MB] Building DF20k Index for {country}...")
        t_idx = time.time()
        idx = CountryIndexV5(conn, country, PROD_CONFIG, include_train_db=False)
        idx_mem = get_peak_ram_mb()
        print(f"[{idx_mem:.1f} MB] Index Build Time: {time.time()-t_idx:.1f}s")
        
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM records WHERE source=1 AND country=? ORDER BY entity_id", (country,))
        
        batch_idx = 0
        while True:
            t_batch_start = time.time()
            refs = [dict(r) for r in cur.fetchmany(100 if is_pilot else BATCH_SIZE)]
            if not refs: break
            
            batch_id = f"{country}_batch_{batch_idx}"
            batch_idx += 1
            if batch_id in manifest['completed_batches']: continue
            
            cands_dict, targets = {}, {}
            all_cands = set()
            
            t_ret = time.time()
            for r in refs:
                cands_dict[r['entity_id']] = idx.retrieve(r)
                all_cands.update(cands_dict[r['entity_id']])
            ret_time = time.time() - t_ret
            
            # Fetch Targets securely
            t_ftc = time.time()
            c_conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
            c_conn.row_factory = sqlite3.Row
            cands_list = list(all_cands)
            for i in range(0, len(cands_list), 999):
                b = cands_list[i:i+999]
                tc = c_conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(b))})", b)
                for tgt in tc: targets[tgt['entity_id']] = dict(tgt)
            c_conn.close()
            ftc_time = time.time() - t_ftc
            
            t_feat = time.time()
            predictions = {}
            for r in refs:
                eid = r['entity_id']
                cands = cands_dict[eid]
                if not cands:
                    predictions[eid] = []; continue
                X = [features(values(r), values(targets[c])) for c in cands]
                probs = clf.predict_proba(X)[:,1]
                predictions[eid] = [cands[i] for i, p in enumerate(probs) if p >= PROD_THRESHOLD]
            feat_pred_time = time.time() - t_feat
            
            # Atomic Write
            with open(OUT_TSV_1, 'a', newline='', encoding='utf-8') as f1, open(OUT_TSV_2, 'a', newline='', encoding='utf-8') as f2:
                w1, w2 = csv.writer(f1, delimiter='\t'), csv.writer(f2, delimiter='\t')
                for r in refs:
                    v = ",".join(predictions.get(r['entity_id'], []))
                    w1.writerow([r['entity_id'], v])
                    w2.writerow([r['entity_id'], v])
                    
            manifest['completed_batches'].append(batch_id)
            with open(MANIFEST.with_suffix('.tmp'), 'w') as f: json.dump(manifest, f)
            MANIFEST.with_suffix('.tmp').replace(MANIFEST)
            
            total_refs_processed += len(refs)
            worker_mem = get_peak_ram_mb()
            batch_time = time.time() - t_batch_start
            
            print(f"  {batch_id} | Refs: {len(refs)} | Ret: {ret_time:.1f}s | Ftc: {ftc_time:.1f}s | Feat+Pred: {feat_pred_time:.1f}s")
            print(f"  -> Batch Time: {batch_time:.1f}s | Worker Peak: {worker_mem:.1f} MB (Delta vs Index: +{worker_mem - idx_mem:.1f} MB)")
            
            if is_pilot:
                # Extrapolate for pilot based on 100 refs
                c_est = (batch_time / len(refs)) * country_total
                est_total_time += c_est
                break # 1 batch per country for pilot
                
        del idx
        gc.collect()

    if is_pilot:
        print(f"\n=== PILOT COMPLETE ===")
        print(f"Estimated Total Runtime: {est_total_time/3600:.2f} HOURS (Target: ~3.5 hours available)")
        print(f"Absolute Peak RAM: {get_peak_ram_mb():.1f} MB")
        print(f"To commence full production, remove --pilot flag.")
    else:
        print(f"\n=== FULL PRODUCTION COMPLETE in {time.time()-t_start:.1f}s ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run_pipeline(is_pilot=args.pilot)