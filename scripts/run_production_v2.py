import sqlite3, json, time, sys, gc, csv, hashlib
from pathlib import Path

# Provide Windows/Linux compatible memory check
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

# NOTE FOR LUNA: Luna must have this DB copied or built from test files.
DB_TEST = ROOT / 'data/processed/test_inference_verified.sqlite'
MODEL_PATH = ROOT / 'artifacts/gpu_challenger/model.cbm'
OUTPUT_DIR = ROOT / 'reports/final_submission'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = OUTPUT_DIR / 'resume_manifest.json'
OUT_MATCHES = OUTPUT_DIR / 'matching_results.tsv'
OUT_CANDS = OUTPUT_DIR / 'candidate_pairs.tsv'

PROD_CONFIG = RetrievalConfig(
    route_exact_name=True, route_exact_address=True, route_joint=True, 
    max_df_abs=20000, top_k_cands=100, union_baseline=False
)
PROD_THRESHOLD = 0.362
BATCH_SIZE = 5000

def get_file_hash(filepath):
    h = hashlib.sha256()
    if not filepath.exists(): return "missing"
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def init_outputs():
    # Codex specified EXACT dual headers
    if not OUT_MATCHES.exists():
        with open(OUT_MATCHES, 'w', newline='', encoding='utf-8') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
    if not OUT_CANDS.exists():
        with open(OUT_CANDS, 'w', newline='', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

def run_pipeline():
    print(f"--- INITIALIZING LUNA GPU PRODUCTION PIPELINE ---")
    if not DB_TEST.exists(): raise FileNotFoundError(f"Missing verified DB at {DB_TEST}")
    
    config_hash = str(hash(str(PROD_CONFIG)))
    model_hash = get_file_hash(MODEL_PATH)
    retrieval_hash = get_file_hash(ROOT / 'src/business_entity_resolution/retrieval_v5.py')
    feature_hash = get_file_hash(ROOT / 'src/business_entity_resolution/inference_features.py')
    
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps({
            'completed_batches': [], 
            'config_hash': config_hash, 'model_hash': model_hash,
            'retrieval_hash': retrieval_hash, 'feature_hash': feature_hash,
            'threshold': PROD_THRESHOLD, 'batch_size': BATCH_SIZE
        }))
    manifest = json.loads(MANIFEST.read_text())
    
    if manifest['model_hash'] != model_hash or manifest['retrieval_hash'] != retrieval_hash:
        raise ValueError("CRITICAL: Manifest configuration/model mismatch. Cannot resume safely.")
        
    init_outputs()
    
    # LUNA NOTE: Specify task_type='GPU' to force CUDA acceleration if not default
    try:
        clf = CatBoostClassifier().load_model(str(MODEL_PATH))
        clf.set_params(task_type='GPU') 
        print("CatBoost forced to GPU mode.")
    except Exception as e:
        print(f"Warning: GPU binding failed, defaulting to native CBM setting. {e}")
        clf = CatBoostClassifier().load_model(str(MODEL_PATH))
    
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM records WHERE source=1 ORDER BY country").fetchall()]
    
    t_start = time.time()
    total_refs = 0
    
    for country in countries:
        print(f"\n--- Loading {country} ---")
        cur = conn.execute("SELECT COUNT(*) FROM records WHERE source=1 AND country=?", (country,))
        country_total = cur.fetchone()[0]
        if country_total == 0: continue
        
        print(f"Building DF20k Index for {country}...")
        idx = CountryIndexV5(conn, country, PROD_CONFIG, include_train_db=False)
        
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM records WHERE source=1 AND country=? ORDER BY entity_id", (country,))
        
        batch_idx = 0
        while True:
            t_b = time.time()
            refs = [dict(r) for r in cur.fetchmany(BATCH_SIZE)]
            if not refs: break
            
            batch_id = f"{country}_batch_{batch_idx}"
            batch_idx += 1
            if batch_id in manifest['completed_batches']: continue
            
            cands_dict, targets = {}, {}
            all_cands = set()
            for r in refs:
                cands_dict[r['entity_id']] = idx.retrieve(r)
                all_cands.update(cands_dict[r['entity_id']])
            
            c_conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
            c_conn.row_factory = sqlite3.Row
            cands_list = list(all_cands)
            for i in range(0, len(cands_list), 999):
                b = cands_list[i:i+999]
                tc = c_conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(b))})", b)
                for tgt in tc: targets[tgt['entity_id']] = dict(tgt)
            c_conn.close()
            
            # Predict & Write
            with open(OUT_MATCHES, 'a', newline='', encoding='utf-8') as fm, open(OUT_CANDS, 'a', newline='', encoding='utf-8') as fc:
                wm, wc = csv.writer(fm, delimiter='\t'), csv.writer(fc, delimiter='\t')
                
                for r in refs:
                    eid = r['entity_id']
                    cands = cands_dict[eid]
                    
                    if not cands:
                        wm.writerow([eid, ""])
                        wc.writerow([eid, ""])
                        continue
                        
                    X = [features(values(r), values(targets[c])) for c in cands]
                    probs = clf.predict_proba(X)[:,1]
                    matches = [cands[i] for i, p in enumerate(probs) if p >= PROD_THRESHOLD]
                    
                    wm.writerow([eid, ",".join(matches)])
                    wc.writerow([eid, ",".join(cands)]) # Candidate file gets ALL cands
                    
            manifest['completed_batches'].append(batch_id)
            with open(MANIFEST.with_suffix('.tmp'), 'w') as f: json.dump(manifest, f)
            MANIFEST.with_suffix('.tmp').replace(MANIFEST)
            
            total_refs += len(refs)
            print(f"  {batch_id} | {len(refs)} Refs | Batch Time: {time.time()-t_b:.1f}s")
                
        del idx
        gc.collect()

    print(f"\n=== LUNA GPU PRODUCTION COMPLETE in {time.time()-t_start:.1f}s ===")

if __name__ == "__main__":
    run_pipeline()