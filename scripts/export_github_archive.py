"""Create a clean source/archive snapshot without bulk caches or Git history."""
from pathlib import Path
import hashlib
import zipfile
ROOT = Path(__file__).resolve().parents[1]

def main():
    paths = set()
    for folder in ('src', 'scripts', 'docs', 'notebooks', 'tests', 'models', 'archive'):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts:
                paths.add(p)
    for name in ('README.md', 'START_HERE.md', '.gitignore', 'project_config.json', 'requirements.txt', 'requirements.in', 'requirements-final.txt', 'start_notebook.sh', 'data/README.md'):
        paths.add(ROOT / name)
    paths.update(ROOT.glob('*.pdf'))
    for p in (ROOT / 'reports').rglob('*'):
        if p.is_file() and p.suffix in ('.md', '.json') and p.stat().st_size < 2_000_000:
            paths.add(p)
    resource = ROOT / '6ab10eb3b23ba_student_resource/student_resource'
    for p in resource.rglob('*'):
        if p.is_file() and 'dataset' not in p.relative_to(resource).parts and '__pycache__' not in p.parts and p.stat().st_size < 2_000_000:
            paths.add(p)
    out = ROOT / 'output/github';out.mkdir(parents=True, exist_ok=True)
    target = out / 'Spartans_0.789_source_archive.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(paths):
            assert p.is_file() and not p.is_symlink(), str(p)
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert 'archive/final/matching_results.tsv.gz' in z.namelist()
        assert all('/dataset/' not in n and not n.startswith(('.git/', '.venv/', 'data/processed/', 'output/')) for n in z.namelist())
    with target.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    print(f'Exported {len(paths)} files, {target.stat().st_size / 1024**2:.1f} MiB: {target}')

if __name__ == '__main__':
    main()
