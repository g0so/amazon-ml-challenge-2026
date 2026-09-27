import sqlite3
import re
import math
import array
import json
import random
import psutil
import time
import sys
from collections import defaultdict, Counter
from pathlib import Path

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))
from business_entity_resolution.workspace import DATASET_DIR

# --- Configuration ---
M_TOKENS = 5
MAX_DF_RATIO = 0.05
MAX_DF_ABS = 5000
SAMPLE_SIZE = 2000
SEED = 2027

DB_PATH = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
TRUTH_PATH = DATASET_DIR / 'train' / 'train_ground_truth.tsv'
OUTPUT_IDS = ROOT / 'data/processed/phase3_pilot_sample_ids.json'

def get_ram_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def tokenize(text):
    if not text:
        return set()
    return set(re.findall(r'[^\W_]+', str(text).lower()))

def eval_set(candidates_dict, truth_dict, limit=None):
    oracle_scores = []
    total_true_links = 0
    retrieved_true_links = 0
    candidate_counts = []
    zero_target_non_singletons = 0
    total_non_singletons = 0

    for ref_id, cands in candidates_dict.items():
        pool = set(cands[:limit]) if limit else set(cands)
        true_t = truth_dict.get(ref_id, set())
        
        g = len(true_t)
        t = len(pool & true_t)
        
        total_true_links += g
        retrieved_true_links += t
        candidate_counts.append(len(pool))
        
        if g == 0:
            oracle_scores.append(1.0)
        else:
            total_non_singletons += 1
            if t == 0:
                oracle_scores.append(0.0)
                zero_target_non_singletons += 1
            else:
                oracle_scores.append((1.25 * t) / (t + 0.25 * g))
                
    link_recall = retrieved_true_links / total_true_links if total_true_links > 0 else 0.0
    macro_f05 = sum(oracle_scores) / len(oracle_scores)
    
    candidate_counts.sort()
    
    return {
        'refs_evaluated': len(candidates_dict),
        'singletons': len(candidates_dict) - total_non_singletons,
        'total_true_links': total_true_links,
        'retrieved_true_links': retrieved_true_links,
        'total_candidates': sum(candidate_counts),
        'link_recall': link_recall,
        'oracle_f05': macro_f05,
        'cands_median': candidate_counts[len(candidate_counts)//2] if candidate_counts else 0,
        'cands_p95': candidate_counts[int(len(candidate_counts)*0.95)] if candidate_counts else 0,
        'cands_max': candidate_counts[-1] if candidate_counts else 0,
        'zero_retrieved_rate': zero_target_non_singletons / total_non_singletons if total_non_singletons > 0 else 0.0
    }

def main():
    t0 = time.time()
    print(f"[{get_ram_mb():.1f} MB] Starting Phase 3 Pilot...")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB missing: {DB_PATH}. Ensure Phase 2 completed.")

    conn = sqlite3.connect(DB_PATH.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    
    print(f"[{get_ram_mb():.1f} MB] Mapping Target IDs to integers...")
    cur = conn.cursor()
    cur.execute("SELECT entity_id, country FROM records WHERE source IN (2, 3) ORDER BY entity_id")
    
    target_string_to_int = {}
    target_int_to_string = []
    country_target_counts = Counter()
    
    for idx, row in enumerate(cur):
        target_int_to_string.append(row['entity_id'])
        target_string_to_int[row['entity_id']] = idx
        country_target_counts[row['country']] += 1
        
    print(f"Loaded {len(target_int_to_string)} targets. Country counts: {dict(country_target_counts)}")

    print(f"[{get_ram_mb():.1f} MB] Building token frequencies...")
    name_df = defaultdict(lambda: defaultdict(int))
    addr_df = defaultdict(lambda: defaultdict(int))
    
    cur.execute("SELECT entity_id, country, name_raw, address_raw FROM records WHERE source IN (2, 3)")
    for row in cur:
        c = row['country']
        for t in tokenize(row['name_raw']):
            name_df[c][t] += 1
        for t in tokenize(row['address_raw']):
            addr_df[c][t] += 1

    name_idf = defaultdict(dict)
    addr_idf = defaultdict(dict)
    skipped_tokens = {'name': 0, 'address': 0}
    
    for c, count in country_target_counts.items():
        limit = min(MAX_DF_ABS, count * MAX_DF_RATIO)
        for t, df in name_df[c].items():
            if df > limit:
                skipped_tokens['name'] += 1
            else:
                name_idf[c][t] = math.log(count / df)
                
        for t, df in addr_df[c].items():
            if df > limit:
                skipped_tokens['address'] += 1
            else:
                addr_idf[c][t] = math.log(count / df)

    print(f"[{get_ram_mb():.1f} MB] Frequencies built. Skipped {skipped_tokens['name']} name tokens and {skipped_tokens['address']} address tokens.")

    print(f"[{get_ram_mb():.1f} MB] Building compact postings arrays...")
    name_index = defaultdict(lambda: defaultdict(lambda: array.array('I')))
    addr_index = defaultdict(lambda: defaultdict(lambda: array.array('I')))
    exact_name_index = defaultdict(lambda: defaultdict(list))
    exact_addr_index = defaultdict(lambda: defaultdict(list))
    
    cur.execute("SELECT entity_id, country, name_raw, address_raw, name_norm, address_norm FROM records WHERE source IN (2, 3)")
    for row in cur:
        c = row['country']
        tgt_int = target_string_to_int[row['entity_id']]
        
        for t in tokenize(row['name_raw']):
            if t in name_idf[c]:
                name_index[c][t].append(tgt_int)
        for t in tokenize(row['address_raw']):
            if t in addr_idf[c]:
                addr_index[c][t].append(tgt_int)
                
        norm_n = row['name_norm']
        norm_a = row['address_norm']
        if norm_n and norm_a:
            exact_name_index[c][norm_n].append(row['entity_id'])
            exact_addr_index[c][norm_a].append(row['entity_id'])

    indexing_time = time.time() - t0
    peak_indexing_ram = get_ram_mb()
    print(f"[{peak_indexing_ram:.1f} MB] Indexing complete in {indexing_time:.1f}s.")

    cur.execute("SELECT entity_id, country, name_raw, address_raw, name_norm, address_norm FROM records WHERE source = 1 ORDER BY entity_id")
    all_refs = cur.fetchall()
    
    random.seed(SEED)
    sample_refs = random.sample(all_refs, SAMPLE_SIZE)
    OUTPUT_IDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_IDS, 'w') as f:
        json.dump([r['entity_id'] for r in sample_refs], f)
        
    print(f"Sampled {SAMPLE_SIZE} reproducible references. ID order seeded with {SEED}.")

    print(f"[{get_ram_mb():.1f} MB] Querying references...")
    q0 = time.time()
    
    results = {}
    
    for row in sample_refs:
        ref_id = row['entity_id']
        c = row['country']
        
        norm_n = row['name_norm']
        norm_a = row['address_norm']
        route_a_matches = set()
        if norm_n and norm_a:
            cand_n = set(exact_name_index[c].get(norm_n, []))
            cand_a = set(exact_addr_index[c].get(norm_a, []))
            route_a_matches = cand_n & cand_a
            
        def get_top_k(text, index_dict, idf_dict):
            tokens = tokenize(text)
            surviving_tokens = [(idf_dict[c][t], t) for t in tokens if t in idf_dict[c]]
            surviving_tokens.sort(key=lambda x: (-x[0], x[1]))
            query_tokens = surviving_tokens[:M_TOKENS]
            
            scores = defaultdict(float)
            for idf, t in query_tokens:
                for tgt_int in index_dict[c][t]:
                    scores[tgt_int] += idf
                    
            ranked_ints = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            top_100_strings = [target_int_to_string[tgt] for tgt, _ in ranked_ints[:100]]
            return top_100_strings
            
        b_candidates = get_top_k(row['name_raw'], name_index, name_idf)
        c_candidates = get_top_k(row['address_raw'], addr_index, addr_idf)
        
        results[ref_id] = {
            'A': list(route_a_matches),
            'B': b_candidates,
            'C': c_candidates
        }
        
    query_time = time.time() - q0
    peak_query_ram = get_ram_mb()
    print(f"[{peak_query_ram:.1f} MB] Querying complete in {query_time:.1f}s.")

    truth = defaultdict(set)
    if TRUTH_PATH.exists():
        with open(TRUTH_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 2 and parts[1]:
                    truth[parts[0]] = set(parts[1].split(','))

    print("\nEvaluating results...")
    def eval_union(k):
        unions = {ref: list(set(r['A'] + r['B'][:k] + r['C'][:k])) for ref, r in results.items()}
        return eval_set(unions, truth)

    a_res = eval_set({r: v['A'] for r, v in results.items()}, truth)
    b_res = eval_set({r: v['B'] for r, v in results.items()}, truth, limit=50)
    c_res = eval_set({r: v['C'] for r, v in results.items()}, truth, limit=50)
    
    u10_res = eval_union(10)
    u50_res = eval_union(50)
    u100_res = eval_union(100)
    
    def print_row(name, cap, res):
        print(f"| {name} | {cap} | {res['refs_evaluated']} / {res['singletons']} | {res['total_true_links']} | {res['retrieved_true_links']} | {res['link_recall']:.4f} | {res['oracle_f05']:.4f} | {res['cands_median']}/{res['cands_p95']}/{res['cands_max']} | {res['zero_retrieved_rate']:.4f} |")

    print("\n| Configuration | Refs / Singletons | True Links | Retrieved Links | Link Recall | Oracle F0.5 | Cands (Med/p95/Max) | Zero-Retrieved Rate |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    print_row("A (Exact)", "-", a_res)
    print_row("B (Name)", "50", b_res)
    print_row("C (Addr)", "50", c_res)
    print_row("Union (A+B+C)", "10", u10_res)
    print_row("Union (A+B+C)", "50", u50_res)
    print_row("Union (A+B+C)", "100", u100_res)
    
    print(f"\nSystem overhead -> Peak Indexing RAM: {peak_indexing_ram:.1f} MB, Indexing Time: {indexing_time:.1f} s")
    print(f"System overhead -> Peak Query RAM: {peak_query_ram:.1f} MB, Query Time: {query_time:.1f} s")

    misses_saved = 0
    missed_pairs = []
    unions_50 = {ref: list(set(r['A'] + r['B'][:50] + r['C'][:50])) for ref, r in results.items()}
    
    for ref_id, cands in unions_50.items():
        if misses_saved >= 5: break
        true_t = truth.get(ref_id, set())
        missed = true_t - set(cands)
        if missed:
            for m in missed:
                missed_pairs.append({
                    'reference_id': ref_id,
                    'missed_target_id': m,
                    'ref_raw_name': [row['name_raw'] for row in sample_refs if row['entity_id'] == ref_id][0],
                    'ref_raw_address': [row['address_raw'] for row in sample_refs if row['entity_id'] == ref_id][0]
                })
                misses_saved += 1
                if misses_saved >= 5: break

    with open(ROOT / 'reports' / f'phase3_misses_{SEED}.json', 'w', encoding='utf-8') as f:
        json.dump(missed_pairs, f, indent=2)

if __name__ == "__main__":
    main()