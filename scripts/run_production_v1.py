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

try: from catboost import CatBoostClassifier
except ImportError:
    sys.path.insert(0, str(ROOT / '.gpu_runtime'))
    from catboost import CatBoostClassifier

DB_TEST = ROOT / 'data/processed/phase6_test_targets.sqlite'
PILOT_DIR = ROOT / 'reports/inference_pilot'
PILOT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = PILOT_DIR / 'manifest.json'
OUT_TSV_1 = PILOT_DIR / 'submission_1.tsv'
OUT_TSV_2 = PILOT_DIR / 'submission_2.tsv'

def run_pipeline(model_path, threshold, config, is_pilot=True):
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps({'completed_batches': [], 'config_hash': str(hash(str(config)))}))
    manifest = json.loads(MANIFEST.read_text())
    
    # Init Outputs safely
    for p in [OUT_TSV_1, OUT_TSV_2]:
        if not p.exists():
            with open(p, 'w') as f: f.write("source1_entity_id\tmatched_entity_ids\n")
            
    clf = CatBoostClassifier().load_model(str(model_path))
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM records WHERE source=1").fetchall()]
    
    t_start = time.time()
    for country in countries:
        batch_id = f"{country}_batch_0"
        if batch_id in manifest['completed_batches']: continue
        
        print(f"\n--- Processing {country} ---")
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM records WHERE source=1 AND country=?", (country,))
        if is_pilot: refs = [dict(r) for r in cur.fetchmany(100)]
        else: refs = [dict(r) for r in cur.fetchall()]
        if not refs: continue
        
        print(f"[{get_peak_ram_mb():.1f} MB] Indexing Targets...")
        idx = CountryIndexV5(conn, country, config)
        mem_index = get_peak_ram_mb()
        
        # Batch inference
        print(f"[{mem_index:.1f} MB] Extracting & Predicting...")
        predictions = {}
        for r in refs:
            eid = r['entity_id']
            cands = idx.retrieve(r)
            if not cands:
                predictions[eid] = []; continue
            
            # Fetch targets securely
            c_conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
            c_conn.row_factory = sqlite3.Row
            targets = {}
            for i in range(0, len(cands), 999):
                batch = cands[i:i+999]
                tc = c_conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
                for tgt in tc: targets[tgt['entity_id']] = dict(tgt)
            c_conn.close()
            
            X = [features(values(r), values(targets[c])) for c in cands]
            probs = clf.predict_proba(X)[:,1]
            predictions[eid] = [cands[i] for i, p in enumerate(probs) if p >= threshold]

        # Atomic Write
        with open(OUT_TSV_1, 'a', newline='') as f1, open(OUT_TSV_2, 'a', newline='') as f2:
            w1, w2 = csv.writer(f1, delimiter='\t'), csv.writer(f2, delimiter='\t')
            for r in refs:
                w1.writerow([r['entity_id'], ",".join(predictions.get(r['entity_id'], []))])
                w2.writerow([r['entity_id'], ",".join(predictions.get(r['entity_id'], []))])
                
        manifest['completed_batches'].append(batch_id)
        MANIFEST.write_text(json.dumps(manifest))
        
        print(f"Batch completed. Worker Peak Memory: {get_peak_ram_mb():.1f} MB")
        del idx
        gc.collect()

    print(f"\nPipeline Finished. Time: {time.time()-t_start:.1f}s")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    
    cfg = RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000, top_k_cands=100, union_baseline=True)
    run_pipeline(ROOT / 'artifacts/gpu_challenger/model.cbm', 0.362, cfg, is_pilot=args.pilot)