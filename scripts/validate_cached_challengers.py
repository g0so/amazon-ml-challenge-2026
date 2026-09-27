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
TRUTH_PATH = ROOT / 'dataset/train/train_ground_truth.tsv' # Or dev truth if available
# Let's inspect tuning predictions from Luna's package to evaluate quickly!
REPAIRED_DIR = ROOT / 'artifacts/gpu_challenger_v2'

def evaluate_cached_predictions():
    print("--- 20-MINUTE MODEL / THRESHOLD VALIDATION ON V3 CACHES ---")
    
    # 1. Load ground truth mapping
    truth = {}
    # Check if dev/train truth exists
    for p in [ROOT / 'dataset/dev/dev_ground_truth.tsv', ROOT / 'dataset/train/train_ground_truth.tsv']:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if len(parts) > 0:
                        truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()

    print(f"Loaded ground truth for {len(truth)} entities.")
    
    # 2. Load Repaired Model Predictions & Search Thresholds from Luna's artifacts if available
    eval_report_path = REPAIRED_DIR / 'evaluation_report.json'
    if eval_report_path.exists():
        report = json.loads(eval_report_path.read_text())
        print("\n--- Luna Repaired Model Evaluation Report ---")
        print(json.dumps(report, indent=2))
    else:
        print("Warning: evaluation_report.json not found in artifacts/gpu_challenger_v2.")

    print("\nRunnable command to score entire V3 cache with Repaired Model at 0.665:")
    print(".venv/bin/python scripts/rescore_v3_cache.py --model artifacts/gpu_challenger_v2/model.cbm --threshold 0.665")

if __name__ == "__main__":
    evaluate_cached_predictions()