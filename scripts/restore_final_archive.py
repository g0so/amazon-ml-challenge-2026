"""Verify and restore final frozen assets without overwriting different files."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    manifest = json.loads((ROOT / 'archive/SHA256_MANIFEST.json').read_text())
    for name, entry in manifest.items():
        p = ROOT / name
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], name
    for label, folder in [('A', 'gpu_challenger'), ('B', 'gpu_challenger_v2')]:
        for src in (ROOT / 'archive/models' / label).iterdir():
            dest = ROOT / 'artifacts' / folder / src.name
            if dest.exists():
                assert sha(dest) == sha(src), 'Different existing file: ' + str(dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    selection = json.loads((ROOT / 'archive/final/selection.json').read_text())
    dest = ROOT / 'reports/final_blend_submission/matching_results.tsv'
    if dest.exists():
        assert sha(dest) == selection['matching_sha256'], 'Different existing prediction file'
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.tsv.restoring')
        with gzip.open(ROOT / 'archive/final/matching_results.tsv.gz', 'rb') as src, temp.open('wb') as out:
            shutil.copyfileobj(src, out)
        assert sha(temp) == selection['matching_sha256']
        temp.replace(dest)
    print('Archive verified. Frozen models and final 0.789 predictions are ready.')

if __name__ == '__main__':
    main()
