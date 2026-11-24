"""Install the requested HR HipX .44 rad override on a stopped, pinned release.

Archives the previous inputs and ELF and invalidates admission before writing.
Never starts a controller or changes perception, policy, or motion shaping.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
from finalize_joint_guard_release import check_identity, validate, atomic_write
from generate_joint_safety import generate
from release_utils import digest, refresh_checksums

PINS={
    'joint_safety_limits.json':'cf73b8fa23d1bce9f913c4f679cea31d183d2ce286ec483b423781e38cb38e41',
    'src/M20_sdk_deploy/run_policy/joint_command_guard.hpp':'f1b942f393b9f001842968dd1060f062d9c65e4ba896102af7518ade6d586078',
    'src/M20_sdk_deploy/run_policy/joint_safety_limits.hpp':'36c8b21df6a9d222239192a52513f1e00a8f4356a5eddf6d55629814902a9f6f',
    'src/M20_sdk_deploy/scripts/joint_command_guard_test.cpp':'cd749c6b6cd85345ebd8de8d9dbf006da1b1705f16badf2c392beac47d14a2cb',
}
PRESERVED={
    'policy/model.onnx':'42558572cc833e278aebb3793ba3ca600395e3d0445658b07d400fe5bd61f594',
    'perception_grace_limits.json':'f51cd9a3d3325244f3733bf39fb008dc3677ebd733d1e9382d034a999cc2dfc2',
    'policy_target_limits.json':'e413943203f797e1dec7277e672a1fa0be3af1c8692685eb36ab992433282fd5',
    'src/M20_sdk_deploy/state_machine/quadruped_wheel/rl_control_state.hpp':'333bab723f6515b974be4517e9bc894239371b164116b7606969f898b7c68529',
}


def stage(root,payload):
    for name,expected in {**PINS,**PRESERVED}.items():
        if digest(root/name)!=expected:raise ValueError('Unknown release input: '+name)
    ledger=json.loads((payload/'joint_safety_limits.json').read_text())
    old=json.loads((root/'joint_safety_limits.json').read_text())
    remaining=copy.deepcopy(ledger)
    if remaining.pop('tracking_error_overrides_confirmed',None) is not True:
        raise ValueError('Explicit tracking override confirmation required')
    override=remaining.pop('tracking_error_overrides',{})
    if (set(override)!={'hr_hipx_joint'} or
        override['hr_hipx_joint'].get('maximum_error_rad')!=.44 or
        override['hr_hipx_joint'].get('reference_position_gain')!=80):
        raise ValueError('Only the requested HR HipX .44 rad override is accepted')
    remaining['derivation']['tracking_error']=old['derivation']['tracking_error']
    if remaining!=old:raise ValueError('Unrelated safety ledger changes rejected')
    edits={'joint_safety_limits.json':(payload/'joint_safety_limits.json').read_bytes(),
           'src/M20_sdk_deploy/run_policy/joint_command_guard.hpp':(payload/'joint_command_guard.hpp').read_bytes(),
           'src/M20_sdk_deploy/scripts/joint_command_guard_test.cpp':(payload/'joint_command_guard_test.cpp').read_bytes()}
    with tempfile.TemporaryDirectory() as tmp:
        header=Path(tmp)/'joint_safety_limits.hpp'
        generate(payload/'joint_safety_limits.json',header)
        edits['src/M20_sdk_deploy/run_policy/joint_safety_limits.hpp']=header.read_bytes()
    for name in ('finalize_joint_guard_release.py','generate_joint_safety.py','apply_tracking_error_override.py'):
        edits['validation_tools/'+name]=(payload/name).read_bytes()
    return edits


def apply(root,payload,mac,serial):
    check_identity(mac,serial)
    if os.geteuid()!=0:raise ValueError('Root required to inspect custom controllers')
    manifest,_=validate(root)
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:exe=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if exe.name=='rl_deploy' and exe!=Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
            raise ValueError('Custom controller active')
    edits=stage(root,payload)
    backup=root/'artifacts'/('hr_hipx_044_before_'+time.strftime('%Y%m%d_%H%M%S')+'.tar.gz')
    names=set(edits)|{'AOS_RELEASE_MANIFEST.json','RELEASE_SHA256SUMS',
                     'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy'}
    with tarfile.open(backup,'x:gz') as tar:
        for name in sorted(names):
            if (root/name).is_file():tar.add(root/name,arcname=name)
    manifest.update(native_rebuild_required=True,offline_validation_complete=False,
                    joint_guard_finalized=False,tracking_error_revision='operator_hr_hipx_044_v1',
                    joint_safety_ledger_sha256=hashlib.sha256(edits['joint_safety_limits.json']).hexdigest(),
                    ready_for_motor_control=False,hardware_locomotion_validation='not_performed',
                    weights_only_update=False,tracking_error_override_backup=str(backup))
    atomic_write(root/'AOS_RELEASE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644)
    for name,data in edits.items():atomic_write(root/name,data,0o644)
    refresh_checksums(root)
    print('HR_HIPX_TRACKING_STAGED scope=not_built_not_started backup='+str(backup))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path)
    p.add_argument('--sources',required=True,type=Path)
    p.add_argument('--expected-mac',required=True);p.add_argument('--expected-serial',required=True)
    a=p.parse_args();apply(a.release.resolve(),a.sources.resolve(),a.expected_mac,a.expected_serial)
