"""Install only local discovery and saved LIO input topic compatibility changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from finalize_joint_guard_release import atomic_write, check_identity, verify
from release_utils import refresh_checksums


def install(root, contract, expected_mac, expected_serial):
    check_identity(expected_mac, expected_serial)
    if os.geteuid() != 0:
        raise ValueError('GOS root required')
    if subprocess.run(['pgrep', '-x', 'm20_deploy'], stdout=subprocess.DEVNULL).returncode == 0:
        raise ValueError('Stop the native GOS supervisor before changing inputs')
    verify(root)
    c = json.loads(contract.read_text())
    changes = []
    specs = [('stage/direct_udp.xml', 'stage/direct_udp.before_local_lidar.disabled', 'direct_udp_baseline_sha256'),
             ('stage/lio_release/gos_lio.yaml', 'stage/lio_release/gos_lio.before_qos_bridge.disabled', 'lio_baseline_sha256')]
    candidates = []
    for name, backup_name, key in specs:
        current, backup = root/name, root/backup_name
        original = backup.read_bytes() if backup.exists() else current.read_bytes()
        if hashlib.sha256(original).hexdigest() != c[key]:
            raise ValueError('Unreviewed original LIO input configuration: '+name)
        text = original.decode()
        if name.endswith('.xml'):
            if text.count('<interfaceWhiteList>') != 1 or '<address>127.0.0.1</address>' in text:
                raise ValueError('Unexpected original discovery profile')
            text = text.replace('<interfaceWhiteList>', '<interfaceWhiteList><address>127.0.0.1</address>', 1)
        else:
            for channel, field in [('cloud', 'lidar_topic'), ('imu', 'imu_topic')]:
                pair = c['qos_topics'][channel]
                before = 'lio.ros.'+field+': '+pair['input']
                if text.count(before) != 1:
                    raise ValueError('Unknown original LIO topic: '+channel)
                text = text.replace(before, 'lio.ros.'+field+': '+pair['output'])
        expected = text.encode()
        if current.read_bytes() not in (original, expected):
            raise ValueError('Unreviewed current LIO input configuration: '+name)
        candidates.append((current, backup, original, expected))
    for current, backup, original, expected in candidates:
        if not backup.exists():
            atomic_write(backup, original, 0o444)
        if current.read_bytes() != expected:
            atomic_write(current, expected, current.stat().st_mode & 0o777)
        changes.append(dict(path=str(current.relative_to(root)), sha256=hashlib.sha256(expected).hexdigest()))
    manifest = root/'GOS_RELEASE_MANIFEST.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        data['sensor_input_repair'] = dict(local_discovery=True, qos_topics=c['qos_topics'],
                                           geometry_timing_and_source_guards_unchanged=True)
        atomic_write(manifest, (json.dumps(data, indent=2)+'\n').encode(), 0o644)
    refresh_checksums(root)
    verify(root)
    result = dict(motor_commands=0, files=changes, vendor_files_changed=False,
                  geometry_timing_and_source_guards_unchanged=True)
    atomic_write(root/'artifacts/lio_input_transport_repair.json', (json.dumps(result, indent=2)+'\n').encode(), 0o644)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('release', type=Path)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--expected-mac', required=True)
    p.add_argument('--expected-serial', required=True)
    a = p.parse_args()
    install(a.release.resolve(), a.contract, a.expected_mac, a.expected_serial)
