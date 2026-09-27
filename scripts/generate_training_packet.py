import sqlite3, json, time, sys, gc, hashlib, pickle, csv
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

from business_entity_resolution.retrieval_v5 import RetrievalConfig, CountryIndexV5
from business_entity_resolution.workspace import DATASET_DIR
from business_entity_resolution.inference_features import values, features

try:
    from business_entity_resolution.normalize import normalize_text
except ImportError:
    import unicodedata
    def normalize_text(s):
        if not s: return ""
        return " ".join(unicodedata.normalize('NFC', str(s)).casefold().split())

sys.path.insert(0, str(ROOT / '.gpu_runtime'))
from catboost import CatBoostClassifier

DB_DEV = ROOT / 'data/processed/phase2_baseline_dev.sqlite'
DB_TRAIN = ROOT / 'data/processed/phase4_train_text.sqlite'
DB_SPLIT = ROOT / 'data/processed/phase2_split.sqlite'
TRUTH_PATH = DATASET_DIR / 'train/train_ground_truth.tsv'
MODEL_PATH = ROOT / 'artifacts/gpu_challenger/model.cbm'

OUT_DIR = ROOT / 'reports/training_packet'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PACKET = OUT_DIR / 'luna_training_packet.pkl'

TRAIN_SAMPLE_SIZE = 5000
TUNE_SAMPLE_SIZE = 2000
CONFIRM_SAMPLE_SIZE = 2000

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def get_exclusions():
    exclude_files = [
        'phase4_train_sample_ids.json', 'phase4_dev_tuning_ids.json', 
        'phase3_pilot_sample_ids.json', 'retrieval_v4_sample_ids.json'
    ]
    exclusions = set()
    for f in exclude_files:
        for p in [ROOT / 'data/processed' / f, ROOT / 'reports' / f]:
            if p.exists(): exclusions.update(json.loads(p.read_text()))
    return exclusions

def load_train_s1_refs(req_ids):
    req_set = set(req_ids)
    train_s1_file = DATASET_DIR / 'train/train_source1.tsv'
    print(f"Reading {len(req_ids)} Train S1 records from {train_s1_file.name}...")
    refs = {}
    with open(train_s1_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            eid = r['entity_id']
            if eid in req_set:
                refs[eid] = {
                    'entity_id': eid,
                    'source': 1,
                    'country': r['country'],
                    'name_raw': r.get('business_name', '') or '',
                    'address_raw': r.get('business_address', '') or '',
                    'name_norm': normalize_text(r.get('business_name', '') or ''),
                    'address_norm': normalize_text(r.get('business_address', '') or ''),
                }
                if len(refs) == len(req_set): break
    assert len(refs) == len(req_set), f"CRITICAL: Found only {len(refs)}/{len(req_set)} train S1 records!"
    return refs

def load_dev_s1_refs(req_ids, conn):
    req_list = list(req_ids)
    refs = {}
    for i in range(0, len(req_list), 999):
        batch = req_list[i:i+999]
        cur = conn.execute(f"SELECT * FROM main.records WHERE entity_id IN ({','.join(['?']*len(batch))})", batch)
        for r in cur: refs[r['entity_id']] = dict(r)
    assert len(refs) == len(req_ids), f"CRITICAL: Found only {len(refs)}/{len(req_ids)} dev S1 records!"
    return refs

def main():
    t_start = time.time()
    exclusions = get_exclusions()
    
    # 1. Sample Disjoint Sets
    conn_split = sqlite3.connect(DB_SPLIT.as_uri() + '?mode=ro', uri=True)
    cur_train = conn_split.execute("SELECT reference_id FROM reference_assignments WHERE split='train'")
    eligible_train = [r[0] for r in cur_train.fetchall() if r[0] not in exclusions]
    cur_dev = conn_split.execute("SELECT reference_id FROM reference_assignments WHERE split='dev'")
    eligible_dev = [r[0] for r in cur_dev.fetchall() if r[0] not in exclusions]
    conn_split.close()
    
    random.seed(42)
    train_ids = random.sample(eligible_train, min(TRAIN_SAMPLE_SIZE, len(eligible_train)))
    dev_samples = random.sample(eligible_dev, TUNE_SAMPLE_SIZE + CONFIRM_SAMPLE_SIZE)
    tune_ids = dev_samples[:TUNE_SAMPLE_SIZE]
    confirm_ids = dev_samples[TUNE_SAMPLE_SIZE:]
    
    (OUT_DIR / 'packet_train_ids.json').write_text(json.dumps(train_ids))
    (OUT_DIR / 'packet_tune_ids.json').write_text(json.dumps(tune_ids))
    (OUT_DIR / 'packet_confirm_ids.json').write_text(json.dumps(confirm_ids))
    
    # 2. Load Ground Truth
    truth = {}
    all_req_ids = set(train_ids + tune_ids + confirm_ids)
    with open(TRUTH_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\n').split('\t')
            if parts[0] in all_req_ids:
                truth[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
    missing_truth = all_req_ids - set(truth.keys())
    assert not missing_truth, f"Missing ground truth for {len(missing_truth)} queries!"
    
    # 3. Load References
    conn_main = sqlite3.connect(DB_DEV.as_uri() + '?mode=ro', uri=True)
    conn_main.row_factory = sqlite3.Row
    conn_main.execute(f"ATTACH DATABASE '{DB_TRAIN}' AS train_db")
    
    train_refs = load_train_s1_refs(train_ids)
    tune_refs = load_dev_s1_refs(tune_ids, conn_main)
    confirm_refs = load_dev_s1_refs(confirm_ids, conn_main)
    
    splits_refs = {
        'train': train_refs,
        'tune': tune_refs,
        'confirm': confirm_refs
    }
    
    # Target Cache for Fast Feature Extraction
    target_cache = {}
    def get_target(eid):
        if eid not in target_cache:
            row = conn_main.execute("SELECT * FROM main.records WHERE entity_id=?", (eid,)).fetchone()
            if not row: row = conn_main.execute("SELECT * FROM train_db.records WHERE entity_id=?", (eid,)).fetchone()
            target_cache[eid] = dict(row)
        return target_cache[eid]

    cfg = RetrievalConfig(route_exact_name=True, route_exact_address=True, route_joint=True, max_df_abs=20000, top_k_cands=100)
    clf = CatBoostClassifier().load_model(str(MODEL_PATH))
    
    packet_data = {
        split: {
            'query_ids': [], 'countries': [], 'truth': {},
            'offsets': [0], 'candidate_ids': [], 
            'features': [], 'labels': [], 'baseline_probs': []
        }
        for split in ['train', 'tune', 'confirm']
    }
    
    # 4. Single-Pass Indexing Per Country
    for country in ['US', 'India']:
        print(f"\n--- Indexing {country} Target Pool (TRAIN+DEV) ---")
        t_idx = time.time()
        idx = CountryIndexV5(conn_main, country, cfg, include_train_db=True)
        print(f"Index built in {time.time()-t_idx:.1f}s.")
        
        for split_name, refs_dict in splits_refs.items():
            c_refs = [r for r in refs_dict.values() if r['country'] == country]
            if not c_refs: continue
            
            print(f"  Extracting {split_name.upper()} ({len(c_refs)} {country} queries)...")
            sd = packet_data[split_name]
            
            for r in c_refs:
                eid = r['entity_id']
                cands = idx.retrieve(r)
                true_targets = truth.get(eid, set())
                
                sd['query_ids'].append(eid)
                sd['countries'].append(country)
                sd['truth'][eid] = list(true_targets)
                
                if not cands:
                    sd['offsets'].append(len(sd['candidate_ids']))
                    continue
                    
                X_batch = [features(values(r), values(get_target(c))) for c in cands]
                probs = clf.predict_proba(X_batch)[:,1]
                
                for i, c in enumerate(cands):
                    sd['candidate_ids'].append(c)
                    sd['features'].append(X_batch[i])
                    sd['labels'].append(1 if c in true_targets else 0)
                    sd['baseline_probs'].append(float(probs[i]))
                    
                sd['offsets'].append(len(sd['candidate_ids']))
                
        del idx
        gc.collect()

    conn_main.close()

    # 5. Format & Validate Arrays
    for split_name in ['train', 'tune', 'confirm']:
        sd = packet_data[split_name]
        assert len(sd['query_ids']) > 0, f"FATAL: {split_name} still has 0 queries!"
        assert len(sd['candidate_ids']) > 0, f"FATAL: {split_name} has 0 candidate pairs!"
        sd['features'] = np.array(sd['features'], dtype=np.float32)
        sd['labels'] = np.array(sd['labels'], dtype=np.int8)
        sd['baseline_probs'] = np.array(sd['baseline_probs'], dtype=np.float32)
        sd['offsets'] = np.array(sd['offsets'], dtype=np.int32)
        print(f"Final {split_name.upper()}: {len(sd['query_ids'])} queries | {len(sd['candidate_ids'])} candidate pairs.")

    packet = {
        'metadata': {
            'retrieval_config': cfg.__dict__,
            'schema': 'v5_22_features',
            'baseline_threshold': 0.362,
            'code_hash': get_file_hash(ROOT / 'src/business_entity_resolution/retrieval_v5.py')
        },
        'splits': packet_data
    }
    
    print(f"\nWriting {OUT_PACKET.name}...")
    with open(OUT_PACKET, 'wb') as f:
        pickle.dump(packet, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Packet Generation Complete in {time.time()-t_start:.1f}s.")

if __name__ == "__main__":
    main()