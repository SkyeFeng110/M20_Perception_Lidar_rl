#!/usr/bin/env python3
"""Start GOS perception with its actual registration residual recorder; no motor API."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=int, default=0,
                        help='0 runs until stopped; finite runs allow 20..120 seconds')
    args = parser.parse_args()
    root = args.release.resolve(strict=True)
    if args.seconds != 0 and not 20 <= args.seconds <= 120:
        parser.error('seconds must be 0 or 20..120')
    status = json.loads((root / 'artifacts/recording_build_status.json').read_text())
    binary = root / 'runtime/build/gos6d_registration'
    if status.get('status') != 'pass' or hashlib.sha256(binary.read_bytes()).hexdigest() != status.get('registration_elf_sha256'):
        raise SystemExit('The residual recorder needs a passing native build and matching binary')
    for process in Path('/proc').glob('[0-9]*/exe'):
        try:
            name = process.resolve(strict=True).name
        except (OSError, RuntimeError):
            continue
        if name == 'gos6d_registration':
            raise SystemExit('GOS perception is already running; stop its owned supervisor before switching')
    output = args.output or root / 'artifacts' / ('residual_live_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        raise SystemExit('A new output directory is required')
    command = [str(root / 'native_tools/m20_deploy'), 'gos-run',
               '--runtime', str(root / 'runtime'), '--stage', str(root / 'stage'),
               '--output', str(output), '--expected-mac', '46:40:20:07:09:44',
               '--seconds', str(args.seconds), '--diagnostics', 'off', '--acceptance', 'off']
    print(json.dumps({'output': str(output), 'residual_directory': str(output / 'registration_evidence'),
                      'motor_commands': 0}), flush=True)
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
