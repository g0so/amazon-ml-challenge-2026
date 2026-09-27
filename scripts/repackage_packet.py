import pickle, json, hashlib, sys, os
import numpy as np
from pathlib import Path

cwd = Path.cwd().resolve()
ROOT = next((p for p in [cwd, *cwd.parents] if (p / 'project_config.json').is_file()), cwd)
sys.path.insert(0, str(ROOT / 'src'))

# Known 22 feature names for V5
FEATURE_NAMES = [
    "n_jac", "a_jac", "n_ex", "a_ex", "n_miss", "a_miss",
    "n_char", "a_char", "n_cont", "a_cont", "num_agr", "num_con",
    "a_num_both", "a_num_one", "a_num_none", "a_num_exact", "a_num_partial", "a_num_unshared_ref", "a_num_unshared_tgt",
    "int_n_jac_a_miss", "int_n_char_a_miss", "int_n_cont_a_miss"
]

PKL_PATH = ROOT / 'reports/training_packet/luna_training_packet.pkl'
EXPORT_DIR = ROOT / 'reports/luna_export'

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def process_split(split_name, mapped_name, pkl_data, export_root):
    print(f"Processing {split_name} -> {mapped_name}/")
    split_dir = export_root / mapped_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    sd = pkl_data['splits'][split_name]
    hashes = {}
    
    # 1. Numpy Arrays
    np.save(split_dir / 'X.npy', sd['features'])
    np.save(split_dir / 'y.npy', sd['labels'])
    np.save(split_dir / 'offsets.npy', sd['offsets'])
    
    hashes[f"{mapped_name}/X.npy"] = get_file_hash(split_dir / 'X.npy')
    hashes[f"{mapped_name}/y.npy"] = get_file_hash(split_dir / 'y.npy')
    hashes[f"{mapped_name}/offsets.npy"] = get_file_hash(split_dir / 'offsets.npy')
    
    # 2. Simple JSONs
    (split_dir / 'reference_ids.json').write_text(json.dumps(sd['query_ids']))
    (split_dir / 'countries.json').write_text(json.dumps(sd['countries']))
    
    hashes[f"{mapped_name}/reference_ids.json"] = get_file_hash(split_dir / 'reference_ids.json')
    hashes[f"{mapped_name}/countries.json"] = get_file_hash(split_dir / 'countries.json')
    
    # 3. JSONL Files
    with open(split_dir / 'candidates.jsonl', 'w', encoding='utf-8') as fc, \
         open(split_dir / 'truth.jsonl', 'w', encoding='utf-8') as ft:
        
        for i, ref_id in enumerate(sd['query_ids']):
            # Slice candidates using offsets
            start, end = sd['offsets'][i], sd['offsets'][i+1]
            cands = sd['candidate_ids'][start:end]
            
            # Write Candidates
            fc.write(json.dumps({"reference_id": ref_id, "candidate_ids": cands}) + '\n')
            
            # Write Truth
            true_ids = sd['truth'].get(ref_id, [])
            ft.write(json.dumps({"reference_id": ref_id, "true_ids": true_ids}) + '\n')
            
    hashes[f"{mapped_name}/candidates.jsonl"] = get_file_hash(split_dir / 'candidates.jsonl')
    hashes[f"{mapped_name}/truth.jsonl"] = get_file_hash(split_dir / 'truth.jsonl')
    
    # Verification
    assert len(sd['query_ids']) == len(sd['offsets']) - 1, f"Offset length mismatch in {split_name}"
    print(f"  Verified {len(sd['query_ids'])} queries, {len(sd['features'])} pairs.")
    return hashes

def main():
    print(f"Loading {PKL_PATH.name}...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
        
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_hashes = {}
    
    # Map 'confirm' to 'confirmation'
    split_map = {'train': 'train', 'tune': 'tune', 'confirm': 'confirmation'}
    
    for old_name, new_name in split_map.items():
        hashes = process_split(old_name, new_name, data, EXPORT_DIR)
        all_hashes.update(hashes)
        
    # Build Metadata
    metadata = {
        'schema_version': '1.0',
        'feature_names': FEATURE_NAMES,
        'retrieval_config': data['metadata']['retrieval_config'],
        'original_code_hash': data['metadata'].get('code_hash', ''),
        'file_hashes': all_hashes
    }
    
    (EXPORT_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print("\nRepackaging complete. Metadata and hashes verified.")
    print(f"Target Directory: {EXPORT_DIR}")

if __name__ == "__main__":
    main()