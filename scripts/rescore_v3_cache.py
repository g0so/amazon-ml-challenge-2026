import sqlite3, json, time, sys, csv, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

sys.path.insert(0, str(ROOT / '.gpu_runtime'))
from catboost import CatBoostClassifier

V3_CACHE_DIR = ROOT / 'data/processed/inference_runs/v3_first_submission/batches'
OUTPUT_DIR = ROOT / 'reports/rescored_v3_submission'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MATCHES = OUTPUT_DIR / 'matching_results.tsv'
OUT_CANDS = OUTPUT_DIR / 'candidate_pairs.tsv'

def rescore_cache(model_path, threshold):
    print(f"--- RESCORING V3 CACHES WITH {model_path} (Threshold: {threshold}) ---")
    t_start = time.time()
    
    clf = CatBoostClassifier().load_model(str(model_path))
    
    if not OUT_MATCHES.exists():
        with open(OUT_MATCHES, 'w', newline='', encoding='utf-8') as f: f.write("source1_entity_id\tmatched_entity_ids\n")
    if not OUT_CANDS.exists():
        with open(OUT_CANDS, 'w', newline='', encoding='utf-8') as f: f.write("source1_entity_id\tcandidate_entity_ids\n")
        
    batch_dirs = [p.parent for p in Path(V3_CACHE_DIR).rglob("features.npy")]
    print(f"Found {len(batch_dirs)} V3 batch directories to rescore.")
    
    total_refs = 0
    for b in batch_dirs:
        tsv_path = b / 'candidate_pairs.tsv'
        npy_path = b / 'features.npy'
        if not tsv_path.exists() or not npy_path.exists(): continue
        
        mmap_feats = np.load(npy_path, mmap_mode='r')
        
        batch_refs, batch_cands = [], []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or row[0] == 'source1_entity_id': continue
                batch_refs.append(row[0])
                batch_cands.append(row[1].split(',') if len(row) > 1 and row[1] else [])
                
        if not batch_refs: continue
        
        # Batch predict
        probs = clf.predict_proba(mmap_feats)[:, 1] if len(mmap_feats) > 0 else []
        
        # Reconstruct matches per reference based on offsets
        curr_idx = 0
        with open(OUT_MATCHES, 'a', newline='', encoding='utf-8') as fm, open(OUT_CANDS, 'a', newline='', encoding='utf-8') as fc:
            wm, wc = csv.writer(fm, delimiter='\t'), csv.writer(fc, delimiter='\t')
            for i, ref_id in enumerate(batch_refs):
                cands = batch_cands[i]
                n_cands = len(cands)
                ref_probs = probs[curr_idx : curr_idx + n_cands] if n_cands > 0 else []
                curr_idx += n_cands
                
                matches = [cands[j] for j, p in enumerate(ref_probs) if p >= threshold]
                
                wm.writerow([ref_id, ",".join(matches)])
                wc.writerow([ref_id, ",".join(cands)])
                total_refs += 1
                
    print(f"\nRescoring Complete in {time.time()-t_start:.1f}s. Total references processed: {total_refs}")
    print(f"Output written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/gpu_challenger_v2/model.cbm")
    parser.add_argument("--threshold", type=float, default=0.665)
    args = parser.parse_args()
    rescore_cache(args.model, args.threshold)