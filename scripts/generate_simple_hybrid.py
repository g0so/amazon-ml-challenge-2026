import csv
import sqlite3
from pathlib import Path
import hashlib

def main():
    print("--- GENERATING SIMPLE HYBRID CHALLENGER ---")
    DB_TEST = Path('data/processed/test_inference_verified.sqlite')
    
    # EXACT PATHS
    FILE_A = Path('data/processed/inference_runs/catboost_first_submission/output/matching_results.tsv')
    FILE_B = Path('reports/rescored_v3_submission/matching_results.tsv')
    
    OUT_HYBRID = Path('reports/hybrid_submission/matching_results.tsv')
    OUT_HYBRID.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Map Test References to Countries
    print("Mapping Test Entities to Countries...")
    conn = sqlite3.connect(f"file:{DB_TEST}?mode=ro", uri=True)
    ref_country_map = {}
    for row in conn.execute("SELECT entity_id, country FROM records WHERE source=1"):
        ref_country_map[row[0]] = row[1]
    
    print(f"Mapped {len(ref_country_map)} references.")
    
    # 2. Load Predictions A (Baseline 0.362 - 0.764 LB)
    preds_a = {}
    if FILE_A.exists():
        with open(FILE_A, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if row: preds_a[row[0]] = row[1]
    else:
        raise FileNotFoundError(f"ERROR: Could not find original submission at {FILE_A}")
        
    # 3. Load Predictions B (Repaired 0.665 - 0.787 LB)
    preds_b = {}
    with open(FILE_B, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if row: preds_b[row[0]] = row[1]
            
    # 4. Generate Hybrid (India -> A, US/France -> B)
    print("Executing Simple Hybrid Policy...")
    counts = {'India (Model A)': 0, 'US/France (Model B)': 0, 'Missing': 0}
    
    with open(OUT_HYBRID, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        
        # We iterate over the safely mapped exact test reference IDs
        for ref_id, country in ref_country_map.items():
            if country == 'India' and ref_id in preds_a:
                writer.writerow([ref_id, preds_a[ref_id]])
                counts['India (Model A)'] += 1
            elif ref_id in preds_b:
                writer.writerow([ref_id, preds_b[ref_id]])
                counts['US/France (Model B)'] += 1
            else:
                writer.writerow([ref_id, ""])
                counts['Missing'] += 1
                
    print(f"\nHybrid Generation Complete:")
    for k, v in counts.items(): print(f"  {k}: {v} references")
    
    # 5. Cryptographic Hash
    h = hashlib.sha256()
    with open(OUT_HYBRID, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    
    print(f"\nSaved to: {OUT_HYBRID}")
    print(f"SHA-256 Hash: {h.hexdigest()}")

if __name__ == '__main__':
    main()
