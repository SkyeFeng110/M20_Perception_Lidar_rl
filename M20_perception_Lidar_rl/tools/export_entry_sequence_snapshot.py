#!/usr/bin/env python3
"""Export a reviewed working-tree snapshot without touching running training."""
import argparse
from collections import Counter
from datetime import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = Path(__file__).with_name('entry_sequence_export_policy.json')
WEIGHTS = {'.pt', '.pth', '.ckpt', '.onnx', '.safetensors', '.jit'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode().strip()


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def reason(path, policy):
    name = path.as_posix()
    if any(part in policy['excluded_directory_names'] or part.endswith('.egg-info')
           or part.startswith('.venv') for part in path.parts):
        return 'cache, environment or local settings'
    if name in policy.get('allowed_launchers', []):
        return None
    if any(fnmatch.fnmatchcase(name, pattern) for pattern in policy['excluded_paths']):
        return 'historical or unrelated content'
    selected = path.is_relative_to(policy.get('selected_run_root', policy['training_run_root']))
    interview = path.is_relative_to('M20_perception_Lidar_rl/interview_prep')
    allowed = name in policy.get('checkpoint_allowlist', [])
    if path.suffix.lower() in WEIGHTS and not (allowed or (interview and not policy.get('strict_checkpoint_allowlist'))):
        return 'checkpoint outside selected training'
    if name in policy.get('log_report_paths', []):
        return None
    if not selected and (path.suffix.lower() in policy['excluded_suffixes']
                         or 'tfevents' in path.name):
        return 'generated artifact'
    if path.name == '.env' or path.name.startswith('.env.'):
        return 'local environment'
    return None


def copy_snapshot(source, target):
    # Bound the copy to the size at open time, including for growing event logs.
    # An opened checkpoint stays readable if training prunes its directory entry.
    with source.open('rb') as src, target.open('wb') as dst:
        remaining = os.fstat(src.fileno()).st_size
        while remaining:
            block = src.read(min(1024 * 1024, remaining))
            if not block:
                raise RuntimeError('Source truncated during snapshot: '+str(source))
            dst.write(block)
            remaining -= len(block)
    target.chmod(0o755 if source.suffix == '.sh' else 0o644)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory outside the source repository')
    parser.add_argument('--published-root', type=Path,
                        help='Existing publication checkout used to preserve previously uploaded logs unchanged')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('Output must be a new directory outside the source repository')
    policy = json.loads(POLICY_PATH.read_text())
    if policy.get('preserved_log_roots') and args.published_root is None:
        parser.error('--published-root is required to preserve previously published logs')
    files, skipped = {}, Counter()
    for scope in policy['include_roots']:
        source = ROOT / scope
        if not source.exists():
            raise FileNotFoundError(source)
        candidates = sorted(source.rglob('*')) if source.is_dir() else [source]
        for source in candidates:
            if source.is_dir():
                continue
            path = source.relative_to(ROOT)
            why = reason(path, policy)
            if why:
                skipped[why] += 1
                continue
            if source.is_symlink():
                raise ValueError('Symlink requires review: '+str(path))
            if source.stat().st_size > policy['maximum_file_bytes']:
                raise ValueError('File exceeds GitHub size budget: '+str(path))
            if SECRET.search(source.read_bytes()):
                raise ValueError('Credential pattern detected: '+str(path))
            files[path] = source
    if args.published_root:
        published = args.published_root.resolve()
        entries = subprocess.check_output(['git', '-C', str(published), 'ls-files', '-z']).decode().split('\0')
        for name in filter(None, entries):
            path = Path(name)
            if any(path.is_relative_to(scope) for scope in policy.get('preserved_log_roots', [])):
                source = published / path
                if not source.is_file() or source.is_symlink():
                    raise ValueError('Missing previously published log: '+name)
                files[path] = source
    output.mkdir(parents=True)
    records = []
    for path, source in sorted(files.items()):
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_snapshot(source, target)
        records.append(dict(path=path.as_posix(), bytes=target.stat().st_size, sha256=digest(target)))
    weights = [r for r in records if Path(r['path']).suffix.lower() in WEIGHTS]
    expected = set(policy['checkpoint_allowlist'])
    actual = {r['path'] for r in weights if policy.get('strict_checkpoint_allowlist') or not Path(r['path']).is_relative_to('M20_perception_Lidar_rl/interview_prep')}
    if actual != expected:
        raise ValueError('Published training checkpoints differ from the allowlist: '+repr(actual ^ expected))
    for row in weights:
        if row['path'] in expected:
            run = (output / row['path']).parent
            for name in ('env.yaml', 'agent.yaml'):
                if not (run / 'params' / name).is_file():
                    raise ValueError('Missing saved run configuration: '+str(run / 'params' / name))
    manifest = dict(format_version=2,
        snapshot_time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        source_head=git('rev-parse', 'HEAD'),
        source_has_uncommitted_changes=bool(git('status', '--porcelain')),
        contains_original_git_history=False, local_source_files_deleted=0,
        training_snapshot_note=policy['notes'],
        robot_submodule_commit=git('-C', str(ROOT / 'M20_perception_Lidar_rl/deep_robotics_model'), 'rev-parse', 'HEAD'),
        policy=policy, checkpoint_files=weights, files=records, filtered=dict(skipped),
        total_bytes=sum(r['bytes'] for r in records))
    (output / 'PUBLISH_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), files=len(records), total_bytes=manifest['total_bytes'],
                         checkpoints=[r['path'] for r in weights], local_source_files_deleted=0), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
