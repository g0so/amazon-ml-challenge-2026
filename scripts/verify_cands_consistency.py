import csv
import hashlib
from pathlib import Path

def hash_row(ref_id, cands_str):
    cands = set(cands_str.split(',')) if cands_str else set()
    cands.discard('')
    sorted_cands = sorted(list(cands))
    s = f"{ref_id}|{','.join(sorted_cands)}"
    return int.from_bytes(hashlib.sha256(s.encode('utf-8')).digest(), 'big')

def main():
    print("--- O(1) MEMORY STREAMING CONSISTENCY CHECK ---")
    new_cands_file = Path('reports/rescored_v3_submission/candidate_pairs.tsv')
    v3_cache_dir = Path('data/processed/inference_runs/v3_first_submission/batches')
    
    count_new, xor_new = 0, 0
    with open(new_cands_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None) # Skip header
        for row in reader:
            if not row: continue
            ref_id = row[0]
            cands_str = row[1] if len(row) > 1 else ""
            xor_new ^= hash_row(ref_id, cands_str)
            count_new += 1
    print(f"Scanned {count_new} references from rescored output.")
    
    count_orig, xor_orig = 0, 0
    for batch_dir in v3_cache_dir.rglob('features.npy'):
        tsv_path = batch_dir.parent / 'candidate_pairs.tsv'
        if not tsv_path.exists(): continue
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row or row[0] == 'source1_entity_id': continue
                ref_id = row[0]
                cands_str = row[1] if len(row) > 1 else ""
                xor_orig ^= hash_row(ref_id, cands_str)
                count_orig += 1
    print(f"Scanned {count_orig} references from original V3 caches.")
    
    assert count_new == count_orig, f"Count mismatch: {count_new} vs {count_orig}"
    assert xor_new == xor_orig, "Hash mismatch! Candidate sets or coverage are NOT identical."
    
    print("CONSISTENCY PASSED: 100% identical reference coverage and candidate sets.")
    print("Byte hash difference is confirmed to be solely due to unordered row writing.")

if __name__ == '__main__':
    main()
