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

OUTPUT_DIR = ROOT / 'reports/final_submission_v4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUTPUT_DIR / 'resume_manifest.json'
OUT_MATCHES = OUTPUT_DIR / 'matching_results.tsv'
OUT_CANDS = OUTPUT_DIR / 'candidate_pairs.tsv'

PROD_CONFIG = RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000, top_k_cands=100, union_baseline=False)
PROD_THRESHOLD = 0.665

# Top-level worker for clean serialization
def _extract_worker(args):
    ref_v, tgt_v = args
    return features(ref_v, tgt_v)

def group_batches_by_country(conn):
    print("Scanning V3 caches to map batches to countries...")
    batch_dirs = [p.parent for p in Path(V3_CACHE_DIR).rglob("features.npy")]
    country_batches = defaultdict(list)
    cur = conn.cursor()
    
    for b in batch_dirs:
        tsv = b / 'candidate_pairs.tsv'
        if not tsv.exists(): continue
        with open(tsv, 'r', encoding='utf-8') as f:
            next(f, None) # Skip header
            first_line = next(f, None)
            if not first_line: continue
            ref_id = first_line.split('\t')[0]
            cur.execute("SELECT country FROM records WHERE entity_id=?", (ref_id,))
            row = cur.fetchone()
            if row: country_batches[row[0]].append(b)
    return country_batches

def run_pipeline(benchmark_batches=None):
    print(f"--- INITIALIZING STREAMING CACHED PRODUCTION ---")
    if not DB_TEST.exists(): raise FileNotFoundError(f"Missing DB at {DB_TEST}")
    
    if not OUT_MATCHES.exists():
        with open(OUT_MATCHES, 'w', newline='', encoding='utf-8') as f: f.write("source1_entity_id\tmatched_entity_ids\n")
    if not OUT_CANDS.exists():
        with open(OUT_CANDS, 'w', newline='', encoding='utf-8') as f: f.write("source1_entity_id\tcandidate_entity_ids\n")
        
    clf = CatBoostClassifier().load_model(str(MODEL_PATH))
    conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    
    country_batches = group_batches_by_country(conn)
    
    # 2 workers to keep memory overhead tiny while gaining parallel string processing
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=2)
    t_start = time.time()
    
    est_total_time, total_cands, total_hits = 0, 0, 0
    
    for country, batches in country_batches.items():
        print(f"\n--- Processing {country} ({len(batches)} batches mapped) ---")
        t_idx = time.time()
        idx = CountryIndexV5(conn, country, PROD_CONFIG, include_train_db=False)
        print(f"[{get_peak_ram_mb():.1f} MB] Index Build Time: {time.time()-t_idx:.1f}s")
        
        b_idx = 0
        country_batch_times = []
        
        for b in batches:
            if benchmark_batches and b_idx >= benchmark_batches: break
            t_b = time.time()
            
            # 1. Parse ONLY this batch's old candidates and offsets
            old_cands_map = {}
            refs_in_order = []
            offset = 0
            with open(b / 'candidate_pairs.tsv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if not row or row[0] == 'source1_entity_id': continue
                    ref_id = row[0]
                    cands = row[1].split(',') if len(row) > 1 and row[1] else []
                    old_cands_map[ref_id] = (offset, cands)
                    offset += len(cands)
                    refs_in_order.append(ref_id)
                    
            mmap = np.load(b / 'features.npy', mmap_mode='r')
            
            # 2. Fetch reference data dynamically
            b_refs = []
            for i in range(0, len(refs_in_order), 999):
                chunk = refs_in_order[i:i+999]
                c = conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(chunk))})", chunk)
                for row in c: b_refs.append(dict(row))
            ref_dict = {r['entity_id']: r for r in b_refs}
            ordered_refs = [ref_dict[rid] for rid in refs_in_order if rid in ref_dict]
            
            hits, total = 0, 0
            miss_pairs, miss_refs, sanity_check_targets = [], [], set()
            cands_dict = {}
            
            # 3. Retrieve new candidates and intersect
            t_ret = time.time()
            for i, r in enumerate(ordered_refs):
                eid = r['entity_id']
                cands = idx.retrieve(r)
                cands_dict[eid] = cands
                
                old_offset, old_cands = old_cands_map.get(eid, (0, []))
                old_cands_dict = {c: old_offset + j for j, c in enumerate(old_cands)}
                
                is_sanity = (b_idx == 0 and i < 5)
                
                for c in cands:
                    total += 1
                    if c in old_cands_dict: hits += 1
                    else:
                        miss_pairs.append((eid, c))
                        miss_refs.append(r)
                    if is_sanity: sanity_check_targets.add(c)
            ret_time = time.time() - t_ret
            
            # 4. Fetch missing targets
            t_ftc = time.time()
            targets = {}
            fetch_ids = list(set([p[1] for p in miss_pairs] + list(sanity_check_targets)))
            if fetch_ids:
                c_conn = sqlite3.connect(DB_TEST.as_uri() + '?mode=ro', uri=True)
                c_conn.row_factory = sqlite3.Row
                for i in range(0, len(fetch_ids), 999):
                    chunk = fetch_ids[i:i+999]
                    tc = c_conn.execute(f"SELECT * FROM records WHERE entity_id IN ({','.join(['?']*len(chunk))})", chunk)
                    for tgt in tc: targets[tgt['entity_id']] = dict(tgt)
                c_conn.close()
            ftc_time = time.time() - t_ftc
            
            # 5. Extract fresh features (Pool of 2)
            t_feat = time.time()
            fresh_features = {}
            if miss_pairs:
                args_list = [(values(miss_refs[i]), values(targets[miss_pairs[i][1]])) for i in range(len(miss_pairs))]
                res_list = pool.map(_extract_worker, args_list)
                for i, pair in enumerate(miss_pairs): fresh_features[pair] = res_list[i]
                    
            # Sanity Check
            if b_idx == 0 and ordered_refs:
                for r in ordered_refs[:5]:
                    eid = r['entity_id']
                    old_offset, old_cands = old_cands_map.get(eid, (0, []))
                    old_cands_dict = {c: old_offset + j for j, c in enumerate(old_cands)}
                    for c in cands_dict[eid]:
                        if c in old_cands_dict:
                            cached_val = mmap[old_cands_dict[c]]
                            fresh_val = features(values(r), values(targets[c]))
                            assert np.allclose(cached_val, fresh_val, atol=1e-5)
                print("  -> Sanity Check PASSED")
                
            # 6. Assemble & Predict
            X_batch, valid_pairs = [], []
            for r in ordered_refs:
                eid = r['entity_id']
                old_offset, old_cands = old_cands_map.get(eid, (0, []))
                old_cands_dict = {c: old_offset + j for j, c in enumerate(old_cands)}
                
                for c in cands_dict[eid]:
                    if c in old_cands_dict: X_batch.append(mmap[old_cands_dict[c]])
                    else: X_batch.append(fresh_features[(eid, c)])
                    valid_pairs.append((eid, c))
            feat_time = time.time() - t_feat
            
            t_pred = time.time()
            predictions = defaultdict(list)
            if X_batch:
                probs = clf.predict_proba(X_batch)[:,1]
                for i, p in enumerate(probs):
                    if p >= PROD_THRESHOLD: predictions[valid_pairs[i][0]].append(valid_pairs[i][1])
            pred_time = time.time() - t_pred
            
            # 7. Write outputs atomically
            t_wr = time.time()
            with open(OUT_MATCHES, 'a', newline='', encoding='utf-8') as fm, open(OUT_CANDS, 'a', newline='', encoding='utf-8') as fc:
                wm, wc = csv.writer(fm, delimiter='\t'), csv.writer(fc, delimiter='\t')
                for r in ordered_refs:
                    eid = r['entity_id']
                    wm.writerow([eid, ",".join(predictions.get(eid, []))])
                    wc.writerow([eid, ",".join(cands_dict.get(eid, []))])
            wr_time = time.time() - t_wr
            
            del mmap; gc.collect()
            
            batch_time = time.time() - t_b
            country_batch_times.append(batch_time)
            total_cands += total
            total_hits += hits
            
            print(f"  Batch {b_idx} | Hit Rate: {hits/total*100:.1f}% ({hits}/{total}) | Feat/Pred Time: {feat_time:.1f}s / {pred_time:.1f}s")
            print(f"  Batch Total: {batch_time:.1f}s | Worker Peak: {get_peak_ram_mb():.1f} MB")
            b_idx += 1
            
        if country_batch_times:
            avg_time = sum(country_batch_times) / len(country_batch_times)
            c_est = avg_time * len(batches)
            est_total_time += c_est
            print(f"  -> Extrapolated {country} remaining: {c_est/3600:.2f} hours")
            
        del idx; gc.collect()
        
    pool.close(); pool.join()
    
    if benchmark_batches:
        print(f"\n=== BENCHMARK COMPLETE ===")
        print(f"Overall Cache Hit Rate: {total_hits/total_cands*100:.1f}%")
        print(f"Projected Total Runtime: {est_total_time/3600:.2f} HOURS")
        print(f"Absolute Peak RAM: {get_peak_ram_mb():.1f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, help="Number of batches per country to benchmark")
    args = parser.parse_args()
    run_pipeline(benchmark_batches=args.benchmark)