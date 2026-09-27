import sqlite3, json, time, sys, gc, csv, hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

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

DB_TEST = ROOT / 'data/processed/test_inference_verified.sqlite'
MODEL_PATH = ROOT / 'artifacts/gpu_challenger_v2/model.cbm'
V3_CACHE_DIR = ROOT / 'data/processed/inference_runs/v3_first_submission/batches'

OUTPUT_DIR = ROOT / 'reports/final_submission_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUTPUT_DIR / 'resume_manifest.json'
OUT_MATCHES = OUTPUT_DIR / 'matching_results.tsv'
OUT_CANDS = OUTPUT_DIR / 'candidate_pairs.tsv'

PROD_CONFIG = RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000, top_k_cands=100, union_baseline=False)
PROD_THRESHOLD = 0.665
BATCH_SIZE = 5000

def _extract_worker(args):
    ref_v, tgt_v = args
    return features(ref_v, tgt_v)

class V3FeatureCache:
    def __init__(self, base_dir):
        self.ref_map = {}
        self.loaded_files = 0
        base = Path(base_dir)
        
        if base.exists():
            # Recursively find ALL batch folders that contain features.npy
            for npy_path in base.rglob("features.npy"):
                batch_dir = npy_path.parent
                tsv_path = batch_dir / 'candidate_pairs.tsv'
                
                if not tsv_path.exists(): continue
                
                try: mmap_feats = np.load(npy_path, mmap_mode='r')
                except Exception: continue
                
                self.loaded_files += 1
                current_row = 0
                with open(tsv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if not row or row[0] == 'source1_entity_id': continue
                        ref_id = row[0]
                        cands = row[1].split(',') if len(row) > 1 and row[1] else []
                        self.ref_map[ref_id] = (mmap_feats, cands, current_row)
                        current_row += len(cands)

    def get_features(self, ref_id, cand_id):
        if ref_id not in self.ref_map: return None
        mmap_feats, cands, start_row = self.ref_map[ref_id]
        try:
            idx = cands.index(cand_id)
            return mmap_feats[start_row + idx]
        except ValueError:
            return None

def run_pipeline(benchmark_batches=None):
    print(f"--- INITIALIZING DEADLOCK-FREE CACHED PRODUCTION ---")
    if not DB_TEST.exists(): raise FileNotFoundError(f"Missing DB at {DB_TEST}")
    
    if not OUT_MATCHES.exists():
        with open(OUT_MATCHES, 'w', newline='', encoding='utf-8') as f: f.write("source1_entity_id\tmatched_entity_ids\n")
    if not OUT_CANDS.exists():
        with open(OUT_CANDS, 'w', newline='', encoding='utf-8') as f: f.write("source1_entity_id\tcandidate_entity_ids\n")
        
    clf = CatBoostClassifier().load_model(str(MODEL_PATH))
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM records WHERE source=1 ORDER BY country").fetchall()]
    
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=mp.cpu_count())
    t_start = time.time()
    
    total_refs, est_total_time = 0, 0
    total_cands_overall, total_hits_overall = 0, 0
    
    # LOAD CACHE ONCE GLOBALLY
    t_cache = time.time()
    cache = V3FeatureCache(V3_CACHE_DIR)
    print(f"Loaded Global V3 Cache: {cache.loaded_files} batch files ({time.time()-t_cache:.1f}s)")
    
    for country in countries:
        print(f"\n--- Loading {country} ---")
        cur = conn.execute("SELECT COUNT(*) FROM records WHERE source=1 AND country=?", (country,))
        country_total = cur.fetchone()[0]
        if country_total == 0: continue
        
        t_idx = time.time()
        idx = CountryIndexV5(conn, country, PROD_CONFIG, include_train_db=False)
        print(f"[{get_peak_ram_mb():.1f} MB] Index Build Time: {time.time()-t_idx:.1f}s")
        
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM records WHERE source=1 AND country=? ORDER BY entity_id", (country,))
        
        b_idx = 0
        country_batch_times = []
        
        while True:
            if benchmark_batches and b_idx >= benchmark_batches: break
            refs = [dict(r) for r in cur.fetchmany(BATCH_SIZE)]
            if not refs: break
            
            t_b = time.time()
            cands_dict, targets = {}, {}
            miss_pairs, miss_refs, sanity_check_targets = [], [], set()
            hits, total = 0, 0
            
            t_ret = time.time()
            for i, r in enumerate(refs):
                eid = r['entity_id']
                cands = idx.retrieve(r)
                cands_dict[eid] = cands
                is_sanity_ref = (b_idx == 0 and i < 10)
                
                for c in cands:
                    total += 1
                    feat = cache.get_features(eid, c)
                    if feat is None:
                        miss_pairs.append((eid, c))
                        miss_refs.append(r)
                    else: hits += 1
                    if is_sanity_ref: sanity_check_targets.add(c)
            ret_time = time.time() - t_ret
            
            t_ftc = time.time()
            fetch_ids = list(set([p[1] for p in miss_pairs] + list(sanity_check_targets)))
            if fetch_ids:
                c_conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
                c_conn.row_factory = sqlite3.Row
                for i in range(0, len(fetch_ids), 999):
                    b = fetch_ids[i:i+999]
                    tc = c_conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(b))})", b)
                    for tgt in tc: targets[tgt['entity_id']] = dict(tgt)
                c_conn.close()
            ftc_time = time.time() - t_ftc
            
            t_feat = time.time()
            fresh_features = {}
            if miss_pairs:
                args_list = [(values(miss_refs[i]), values(targets[miss_pairs[i][1]])) for i in range(len(miss_pairs))]
                res_list = pool.map(_extract_worker, args_list)
                for i, pair in enumerate(miss_pairs):
                    fresh_features[pair] = res_list[i]
            
            X_batch, valid_pairs = [], []
            for r in refs:
                eid = r['entity_id']
                for c in cands_dict[eid]:
                    feat = cache.get_features(eid, c)
                    if feat is None: feat = fresh_features[(eid, c)]
                    X_batch.append(feat)
                    valid_pairs.append((eid, c))
            feat_time = time.time() - t_feat
            
            t_pred = time.time()
            predictions = defaultdict(list)
            if X_batch:
                probs = clf.predict_proba(X_batch)[:,1]
                for i, p in enumerate(probs):
                    if p >= PROD_THRESHOLD: predictions[valid_pairs[i][0]].append(valid_pairs[i][1])
            pred_time = time.time() - t_pred
            
            t_wr = time.time()
            with open(OUT_MATCHES, 'a', newline='', encoding='utf-8') as fm, open(OUT_CANDS, 'a', newline='', encoding='utf-8') as fc:
                wm, wc = csv.writer(fm, delimiter='\t'), csv.writer(fc, delimiter='\t')
                for r in refs:
                    eid = r['entity_id']
                    wm.writerow([eid, ",".join(predictions.get(eid, []))])
                    wc.writerow([eid, ",".join(cands_dict.get(eid, []))])
            wr_time = time.time() - t_wr
            
            batch_time = time.time() - t_b
            country_batch_times.append(batch_time)
            total_cands_overall += total
            total_hits_overall += hits
            
            print(f"  Batch {b_idx} | Hit Rate: {hits/total*100:.1f}% ({hits}/{total}) | Feat/Pred Time: {feat_time:.1f}s / {pred_time:.1f}s")
            print(f"  Batch Total: {batch_time:.1f}s | Worker Peak: {get_peak_ram_mb():.1f} MB")
            b_idx += 1
            
        if country_batch_times:
            avg_b_time = sum(country_batch_times) / len(country_batch_times)
            c_est = avg_b_time * (country_total / BATCH_SIZE)
            est_total_time += c_est
            print(f"  -> Extrapolated remaining {country} inference: ~{c_est/3600:.2f} hours")
            
        del idx; gc.collect()

    pool.close(); pool.join()
    
    if benchmark_batches:
        print(f"\n=== BENCHMARK COMPLETE ===")
        print(f"Overall Cache Hit Rate: {total_hits_overall/total_cands_overall*100:.1f}%")
        print(f"Projected Total Runtime: {est_total_time/3600:.2f} HOURS (Target: < 4 hours)")
        print(f"Absolute Peak RAM: {get_peak_ram_mb():.1f} MB")
        print(f"To commence full production, run without the --benchmark flag.")
    else:
        print(f"\n=== PRODUCTION RUN COMPLETE in {time.time()-t_start:.1f}s ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, help="Number of batches per country to benchmark")
    args = parser.parse_args()
    run_pipeline(benchmark_batches=args.benchmark)