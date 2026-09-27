"""Verify shared output archives and optionally restore their original paths."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from prepare_rtx_handoff import digest, destination

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    folder = ROOT / 'output/verification'
    manifest = json.loads((folder / 'manifest.json').read_text())
    with tempfile.TemporaryFile() as combined:
        for part in manifest['parts']:
            path = destination(folder, part['name'])
            with path.open('rb') as source:
                if path.stat().st_size != part['bytes'] or digest(source) != part['sha256']:
                    raise ValueError('Corrupt archive part: ' + part['name'])
                source.seek(0)
                shutil.copyfileobj(source, combined)
        combined.seek(0)
        with zipfile.ZipFile(combined) as archive:
            if json.loads(archive.read('MANIFEST.json')) != manifest['files']:
                raise ValueError('Archive manifest differs from tracked manifest')
            missing = []
            for name, expected in manifest['files'].items():
                target = destination(args.destination.resolve(), name)
                with archive.open(name) as source:
                    if archive.getinfo(name).file_size != expected['bytes'] or digest(source) != expected['sha256']:
                        raise ValueError('Payload verification failed: ' + name)
                if target.exists():
                    with target.open('rb') as source:
                        if digest(source) != expected['sha256']:
                            raise ValueError('Existing different file preserved; use --destination: ' + name)
                else:
                    missing.append((name, target))
            if args.extract:
                for name, target in missing:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                        temporary = Path(output.name)
                        try:
                            with archive.open(name) as source:
                                shutil.copyfileobj(source, output)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                    temporary.replace(target)
            print(f"Verified {len(manifest['files'])} files; "
                  f"{len(missing)} {'restored' if args.extract else 'missing (use --extract)' }.")


if __name__ == '__main__':
    main()
