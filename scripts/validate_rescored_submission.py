import csv
import sqlite3
from pathlib import Path

OUT_MATCHES = Path('reports/rescored_v3_submission/matching_results.tsv')
OUT_CANDS = Path('reports/rescored_v3_submission/candidate_pairs.tsv')
DB_TEST = Path('data/processed/test_inference_verified.sqlite')

def main():
    print("--- STRUCTURAL VALIDATION ---")
    conn = sqlite3.connect(f"file:{DB_TEST}?mode=ro", uri=True)
    expected_ids = set(r[0] for r in conn.execute("SELECT entity_id FROM records WHERE source=1"))
    
    matches_ids, cands_ids = set(), set()
    errors = []
    
    with open(OUT_CANDS, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if header != ['source1_entity_id', 'candidate_entity_ids']: errors.append(f"Invalid cands header: {header}")
        for i, row in enumerate(reader):
            if not row: continue
            rid, cands = row[0], row[1].split(',') if len(row)>1 and row[1] else []
            cands_ids.add(rid)
            if len(cands) != len(set(cands)): errors.append(f"Row {i+2}: Dupes in {rid}")
                
    with open(OUT_MATCHES, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if header != ['source1_entity_id', 'matched_entity_ids']: errors.append(f"Invalid matches header: {header}")
        for i, row in enumerate(reader):
            if not row: continue
            rid, matches = row[0], row[1].split(',') if len(row)>1 and row[1] else []
            matches_ids.add(rid)
            if len(matches) != len(set(matches)): errors.append(f"Row {i+2}: Dupes in {rid}")
                
    if expected_ids != cands_ids: errors.append(f"Coverage mismatch: {len(expected_ids)} vs {len(cands_ids)} in cands.")
    if expected_ids != matches_ids: errors.append(f"Coverage mismatch: {len(expected_ids)} vs {len(matches_ids)} in matches.")
        
    if not errors: print(f"Validation PASSED. 100% test coverage ({len(expected_ids)} queries). No duplicates. Headers valid.")
    else:
        print("Validation FAILED:"); 
        for e in errors[:10]: print("  -", e)

if __name__ == '__main__':
    main()
