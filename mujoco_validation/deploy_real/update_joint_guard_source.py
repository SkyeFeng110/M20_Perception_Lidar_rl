"""Apply reviewed guard-source updates to an inactive validated release only."""
import argparse
import json
import os
from pathlib import Path
import hashlib
import tempfile
from finalize_joint_guard_release import check_identity,validate,verify,atomic_write
from release_utils import refresh_checksums


def update(root,sources,mac,serial,ledger=None):
    check_identity(mac,serial)
    if os.geteuid()!=0:raise ValueError('Root required to inspect all controller processes')
    verify(root)
    manifest=json.loads((root/'AOS_RELEASE_MANIFEST.json').read_text())
    if manifest.get('joint_safety_revision')!='joint_publish_guard_v1':raise ValueError('Unknown safety release')
    if manifest.get('native_rebuild_required') is False and manifest.get('offline_validation_complete') is True:
        validate(root)
    # A failed staging build may be repaired; it cannot be admitted for motion.
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:exe=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe.name=='rl_deploy' and exe!=Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
            raise ValueError('Custom controller active')
    ledger_header=None
    if ledger is not None:
        from generate_joint_safety import generate
        current=json.loads((root/'joint_safety_limits.json').read_text())
        candidate=json.loads(ledger.read_text())
        allowed={'source','knee_limit_source'}
        if ({k:v for k,v in current.items() if k not in allowed} !=
                {k:v for k,v in candidate.items() if k not in allowed} or
                candidate.get('knee_limit_source')!='operator_confirmed'):
            raise ValueError('Only the confirmed knee-bound selection may change')
        with tempfile.TemporaryDirectory() as tmp:
            header=Path(tmp)/'joint_safety_limits.hpp';generate(ledger,header)
            ledger_header=header.read_bytes()
    # Revoke admission before changing any source; rebuild must earn it again.
    manifest.update(native_rebuild_required=True,offline_validation_complete=False,joint_guard_finalized=False)
    if ledger_header is not None:
        manifest.update(joint_safety_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
                        angle_limit_revision='operator_knee_bounds_v1')
    atomic_write(root/'AOS_RELEASE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644)
    if ledger_header is not None:
        atomic_write(root/'joint_safety_limits.json',ledger.read_bytes(),0o644)
        atomic_write(root/'src/M20_sdk_deploy/run_policy/joint_safety_limits.hpp',ledger_header,0o644)
    for name,dest in [('guarded_dds_publish.inc','run_policy/guarded_dds_publish.inc'),
                      ('incident_recorder.hpp','run_policy/incident_recorder.hpp'),
                      ('incident_recorder_test.cpp','scripts/incident_recorder_test.cpp'),
                      ('dds_joint_guard_no_motor_test.cpp','scripts/dds_joint_guard_no_motor_test.cpp')]:
        atomic_write(root/'src/M20_sdk_deploy'/dest,(sources/name).read_bytes(),0o644)
    refresh_checksums(root)
    print('JOINT_GUARD_SOURCE_UPDATED motor_admission_revoked_until_rebuild')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--expected-mac',required=True);p.add_argument('--expected-serial',required=True)
    p.add_argument('--ledger',type=Path)
    a=p.parse_args();update(a.release.resolve(),a.sources.resolve(),a.expected_mac,a.expected_serial,a.ledger)
