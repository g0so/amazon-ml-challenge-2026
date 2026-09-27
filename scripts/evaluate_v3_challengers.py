import sqlite3, json, time, sys, csv
import numpy as np
from pathlib import Path
from collections import defaultdict

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

sys.path.insert(0, str(ROOT / '.gpu_runtime'))
from catboost import CatBoostClassifier

V3_CACHE_DIR = ROOT / 'data/processed/inference_runs/v3_first_submission/batches'
TRUTH_PATH = ROOT / 'dataset/train/train_ground_truth.tsv'
DEV_TRUTH_PATH = ROOT / 'dataset/dev/dev_ground_truth.tsv'

def evaluate_models():
    print("--- V3 CACHED CHALLENGER COMPARISON ---")
    
    # 1. Load Ground Truth (Train + Dev if available)
    truth = {}
    for p in [DEV_TRUTH_PATH, TRUTH_PATH]:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if len(parts) > 0:
                        truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
    print(f"Loaded ground truth for {len(truth)} entities.")
    
    # 2. Load Models
    orig_model_path = ROOT / 'artifacts/baseline/model.cbm' # or original path if known, else we use probabilities from v3 cache if stored
    repaired_model_path = ROOT / 'artifacts/gpu_challenger_v2/model.cbm'
    
    repaired_clf = CatBoostClassifier().load_model(str(repaired_model_path))
    
    # We will score a sample of V3 batches to compute macro F0.5, FP, FN, singleton errors
    batch_dirs = [p.parent for p in Path(V3_CACHE_DIR).rglob("features.npy")][:20] # Sample 20 batches for speed
    
    total_pairs = 0
    y_true_all, y_prob_all = [], []
    
    for b in batch_dirs:
        tsv_path = b / 'candidate_pairs.tsv'
        npy_path = b / 'features.npy'
        if not tsv_path.exists() or not npy_path.exists(): continue
        
        mmap_feats = np.load(npy_path, mmap_mode='r')
        if len(mmap_feats) == 0: continue
        
        batch_refs, batch_cands = [], []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or row[0] == 'source1_entity_id': continue
                ref_id = row[0]
                cands = row[1].split(',') if len(row) > 1 and row[1] else []
                batch_refs.append(ref_id)
                batch_cands.append(cands)
                
        # Compute probabilities with repaired model
        probs = repaired_clf.predict_proba(mmap_feats)[:, 1]
        
        curr = 0
        for i, ref_id in enumerate(batch_refs):
            cands = batch_cands[i]
            true_set = truth.get(ref_id, set())
            for j, c in enumerate(cands):
                y_true_all.append(1 if c in true_set else 0)
                y_prob_all.append(probs[curr + j])
            curr += len(cands)
            
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    print(f"Evaluated {len(y_true_all)} candidate pairs across sampled V3 batches.")
    
    # Test Thresholds: A (0.362), B/C (0.665 or tuned)
    for thresh in [0.362, 0.665]:
        preds = (y_prob_all >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true_all == 1))
        fp = np.sum((preds == 1) & (y_true_all == 0))
        fn = np.sum((preds == 0) & (y_true_all == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold {thresh:.3f} -> Precision: {precision:.4f} | Recall: {recall:.4f} | F0.5: {f05:.4f} | FP: {fp} | FN: {fn}")

if __name__ == "__main__":
    evaluate_models()