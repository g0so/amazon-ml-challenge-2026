# Frozen retrieval copy for fair replay/export; original pilot is unchanged.
import sqlite3
import re
import math
import array
import json
import random
import hashlib
import zipfile
import csv
import io
import resource
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

    bundle = ROOT / 'data/processed/phase3_comparison/phase3_fair_comparison.zip'
    with zipfile.ZipFile(bundle) as z:
        saved_bytes = z.read('reference_ids.json')
        saved_ids = json.loads(saved_bytes)
        comparison_metadata = json.loads(z.read('metadata.json'))
    assert len(saved_ids) == len(set(saved_ids)) == 2000
    assert hashlib.sha256(saved_bytes).hexdigest() == comparison_metadata['sample_file_sha256']
    output_dir = ROOT / 'data/processed/phase3_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

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

    pool_hash = hashlib.sha256(b'entity_id\n')
    for target_id in target_int_to_string:
        pool_hash.update((target_id + '\n').encode())
    assert len(target_int_to_string) == comparison_metadata['target_count']
    assert pool_hash.hexdigest() == comparison_metadata['dev_target_ids_tsv_sha256']
    print('Exact comparison target-pool hash verified.')

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

    # Replay the saved IDs exactly; never resample or overwrite the manifest.
    sample_refs = []
    for ref in saved_ids:
        row = conn.execute("SELECT entity_id, country, name_raw, address_raw, name_norm, address_norm FROM records WHERE source=1 AND entity_id=?", (ref,)).fetchone()
        if row is None:
            raise ValueError('Saved reference missing: ' + ref)
        sample_refs.append(row)
    print(f"Loaded all {len(sample_refs)} saved reference IDs unchanged.")

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

    # Persist candidates before opening any ground truth.
    assert set(results) == set(saved_ids)
    artifact = output_dir / 'our_phase3_candidates.zip'
    with zipfile.ZipFile(artifact, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('reference_ids.json', saved_bytes)
        with z.open('routes.jsonl', 'w') as out:
            for ref in saved_ids:
                row = {'reference_id': ref, 'A': sorted(results[ref]['A']),
                       'B': results[ref]['B'], 'C': results[ref]['C']}
                out.write((json.dumps(row) + '\n').encode())
        for k in (10,50,100):
            with z.open(f'union_k{k}.jsonl', 'w') as out:
                for ref in saved_ids:
                    r = results[ref]
                    ids = sorted(set(r['A'] + r['B'][:k] + r['C'][:k]))
                    out.write((json.dumps({'reference_id': ref, 'candidate_ids': ids}) + '\n').encode())

    # Ground truth is read only after the inference artifact is frozen.
    with zipfile.ZipFile(bundle) as z:
        reader = csv.DictReader(io.StringIO(z.read('sample_ground_truth.tsv').decode()), delimiter='\t')
        truth_rows = list(reader)
    assert len(truth_rows) == len(saved_ids)
    truth = {r['source1_entity_id']: set(r['matched_entity_ids'].split(',')) if r['matched_entity_ids'] else set() for r in truth_rows}
    assert set(truth) == set(saved_ids)
    assert sum(map(len, truth.values())) == comparison_metadata['sample_true_links']
    measurements = {}
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None
        for k in (10,50,100):
            with z.open(f'union_k{k}.jsonl') as f:
                rows = [json.loads(line) for line in f]
            assert len(rows) == len(saved_ids)
            candidates = {r['reference_id']: r['candidate_ids'] for r in rows}
            assert set(candidates) == set(saved_ids)
            for ids in candidates.values():
                assert len(ids) == len(set(ids))
                assert all(t in target_string_to_int for t in ids)
            m = eval_set(candidates, truth)
            m['mean_candidates'] = m['total_candidates']/len(saved_ids)
            # Independent oracle check using the shared Phase 2 Python metric.
            from business_entity_resolution.metrics import score_entity
            independent = sum(score_entity(truth[r], set(candidates[r]) & truth[r]) for r in saved_ids)/len(saved_ids)
            assert abs(independent-m['oracle_f05']) < 1e-12
            measurements[f'union_k{k}'] = m
    report = {'comparison_metadata': comparison_metadata,
              'retrieval_settings': {'M_TOKENS':M_TOKENS, 'MAX_DF_RATIO':MAX_DF_RATIO, 'MAX_DF_ABS':MAX_DF_ABS},
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'index_seconds': indexing_time, 'query_seconds': query_time,
              'peak_process_ram_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'metrics': measurements,
              'notes': ['Retrieval/index code copied unchanged from existing pilot; saved sample replay and export added.',
                        'All candidate files contain every saved reference, including empty lists.',
                        'Targets verified against supplied pool hash before retrieval; labels loaded after artifact creation.',
                        'Oracle values are ceilings, not classifier scores.']}
    report_path = ROOT/'reports/phase3_fair_comparison_ours.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(artifact,'a',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('results.json',json.dumps(report,indent=2)+'\n')
        z.write(Path(__file__), 'run_phase3_comparison.py')
        z.writestr('README.txt','Each union_k*.jsonl contains every saved reference and sorted candidate_ids. K is per token route. routes.jsonl preserves B/C rank order and includes exact A matches. Use union_k50.jsonl for the primary comparison. Compare with the separately shared ground truth only after retrieval. results.json records exact sample/pool hashes, measured scores, and process resources. No retrieval settings were tuned. Overlap with the other method still requires its actual candidate artifact.\n')
    conn.close()
    print(json.dumps(report,indent=2))
    print('Candidate archive:', artifact)

if __name__ == '__main__':
    main()
