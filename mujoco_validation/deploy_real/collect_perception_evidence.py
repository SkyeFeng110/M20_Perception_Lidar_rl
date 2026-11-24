#!/usr/bin/env python3
"""Export existing GOS residual or AOS frozen incident logs. Never uses motor I/O."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def collect(root, output):
    root = root.resolve(strict=True)
    if not root.is_dir() or output.exists():
        raise ValueError('An existing log directory and a new archive path are required')
    files = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or not p.is_file():
            continue
        # AOS directories may also contain older explicit R2 incident archives.
        if p.name.startswith('segment_') and p.suffix == '.jsonl' or p.name in {
            'first_fault.json', 'first_rejection.json', 'recording_status.json'
        }:
            files.append(p)
    if not files:
        raise ValueError('No perception or incident evidence files found')
    manifest = {'schema': 1, 'collected_wall_s': time.time(), 'source': str(root),
                'motor_commands': 0, 'files': [],
                'snapshot': 'independent file snapshots; use frozen AOS archives for a stable incident'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as destination, tarfile.open(fileobj=destination, mode='w:gz') as archive:
        for p in files:
            # Bound a live file snapshot at its initial length. Rotation can be
            # detected through record timestamps; this never pauses a controller.
            with p.open('rb') as f:
                data = f.read(p.stat().st_size)
            name = str(p.relative_to(root))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
            manifest['files'].append({'path': name, 'bytes': len(data),
                                      'sha256': hashlib.sha256(data).hexdigest()})
        payload = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('collection_manifest.json')
        info.size = len(payload)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(payload))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='GOS run/registration_evidence or AOS incident directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.input, args.output)
    print(json.dumps({'archive': str(args.output), 'files': len(result['files']),
                      'bytes': args.output.stat().st_size, 'motor_commands': 0}))


if __name__ == '__main__':
    main()
