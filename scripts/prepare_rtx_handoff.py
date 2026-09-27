"""Verify and restore the GitHub handoff assets using only Python's stdlib."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if '\\' in name or path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return path


def destination(root, name):
    target = root.joinpath(*safe_name(name).parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Destination escapes project through symlink: ' + name)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'output/handoffs/recovery_scaleup_20260927.zip')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    root = args.destination.resolve()
    expected = args.bundle.with_suffix('.zip.sha256').read_text().split()[0]
    with args.bundle.open('rb') as stream:
        if digest(stream) != expected:
            raise ValueError('Bundle SHA-256 mismatch; restore it from GitHub before continuing')
    with zipfile.ZipFile(args.bundle) as outer:
        if len(outer.namelist()) != len(set(outer.namelist())):
            raise ValueError('Duplicate archive entries')
        for name in outer.namelist():
            safe_name(name)
        manifest = json.loads(outer.read('HANDOFF_MANIFEST.json'))
        restore = []
        for name, metadata in manifest.items():
            if outer.getinfo(name).file_size != metadata['bytes']:
                raise ValueError('Payload size mismatch: ' + name)
            with outer.open(name) as stream:
                if digest(stream) != metadata['sha256']:
                    raise ValueError('Payload hash mismatch: ' + name)
            if name.startswith(('artifacts/gpu_challenger/', 'data/processed/')):
                restore.append((name, outer, name, metadata['sha256']))
        packet_bytes = outer.read('data/processed/gpu_feature_packet_v1.zip')
        with zipfile.ZipFile(io.BytesIO(packet_bytes)) as packet:
            if len(packet.namelist()) != len(set(packet.namelist())):
                raise ValueError('Duplicate feature packet entries')
            for info in packet.infolist():
                safe_name(info.filename)
                if info.is_dir():
                    continue
                with packet.open(info.filename) as stream:
                    sha = digest(stream)
                restore.append(('data/processed/gpu_feature_packet_v1/' + info.filename,
                                packet, info.filename, sha))
            missing = []
            conflicts = []
            for name, archive, member, sha in restore:
                target = destination(root, name)
                if target.exists():
                    if not target.is_file():
                        conflicts.append(name)
                        continue
                    with target.open('rb') as stream:
                        if digest(stream) != sha:
                            conflicts.append(name)
                else:
                    missing.append((target, archive, member))
            if conflicts:
                raise ValueError('Existing different files preserved. Use a separate checkout/destination:\n' + '\n'.join(conflicts))
            print(f'Verified bundle and {len(manifest)} payload hashes. '
                  f'{len(restore) - len(missing)} assets already match; {len(missing)} missing.')
            if args.extract:
                for target, archive, member in missing:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                            temporary = Path(output.name)
                            with archive.open(member) as source:
                                shutil.copyfileobj(source, output, length=1024 * 1024)
                            output.flush()
                            os.fsync(output.fileno())
                        os.replace(temporary, target)
                    finally:
                        if temporary is not None and temporary.exists():
                            temporary.unlink()
                print(f'Restored {len(missing)} files under {root}.')
            else:
                print('Check only. Add --extract to restore missing assets.')
    print('Next: read docs/LUNA_RTX_EXECUTION_PROMPT.txt. Raw datasets remain local and are not bundled.')


if __name__ == '__main__':
    main()
