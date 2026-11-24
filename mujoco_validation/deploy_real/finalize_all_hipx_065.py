"""Finalize a validated 1515 release and redirect its previous launch script.

Only reads identity/checksums and writes release files. Never imports ROS, starts
a controller, stops a service, or sends a motor command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
from release_utils import refresh_checksums

HERE=Path(__file__).resolve().parent
JOINTS=(0,1,2,4,5,6,8,9,10,12,13,14)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_identity(expected_mac,expected_serial):
    serial_path=Path('/sys/firmware/devicetree/base/serial-number')
    serial=serial_path.read_bytes().rstrip(b'\0').decode() if serial_path.exists() else None
    macs={p.read_text().strip().lower() for p in Path('/sys/class/net').glob('*/address')}
    if serial!=expected_serial or expected_mac.lower() not in macs:
        raise ValueError('Target identity mismatch; no release files changed')


def verify(root):
    for line in (root/'RELEASE_SHA256SUMS').read_text().splitlines():
        expected,name=line.split(None,1)
        if digest(root/name)!=expected:raise ValueError('Release file changed: '+str(root/name))


def atomic_write(path,content,mode):
    owner=path.stat() if path.exists() else path.parent.stat()
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.chmod(tmp,mode)
        if os.geteuid()==0:os.chown(tmp,owner.st_uid,owner.st_gid)
        os.replace(tmp,path)
        parent=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(parent)
        finally:os.close(parent)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def validate(root):
    verify(root)
    m=json.loads((root/'AOS_RELEASE_MANIFEST.json').read_text())
    binary=root/'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy'
    ledger=root/'joint_safety_limits.json'
    if (m.get('joint_safety_revision')!='joint_publish_guard_v1' or
        m.get('native_rebuild_required') is not False or m.get('offline_validation_complete') is not True or
        digest(binary)!=m.get('joint_guard_elf_sha256') or digest(ledger)!=m.get('joint_safety_ledger_sha256')):
        raise ValueError('Guard build/validation/ELF/ledger identity incomplete')
    header=(root/'src/M20_sdk_deploy/run_policy/joint_safety_limits.hpp').read_text()
    if digest(ledger) not in header:raise ValueError('Compiled header not generated from this ledger')
    exits=sorted((root/'artifacts').glob('joint_guard_build_*.exit'),key=lambda p:p.stat().st_mtime)
    if not exits or exits[-1].read_text().strip()!='0':raise ValueError('Latest native validation failed or missing')
    log=exits[-1].with_suffix('.log').read_text()
    required=['JOINT_COMMAND_GUARD=PASS','INCIDENT_RECORDER=PASS','INCIDENT_IO_FAILURE=PASS',
              'INCIDENT_FREEZE=PASS','DDS_JOINT_GUARD=PASS','DDS_GUARD_TRACE=PASS','NATIVE_1516_BUILD=PASS']
    if m.get('incident_startup_revision')=='startup_safety_wait_v1':
        required += ['INCIDENT_STARTUP=PASS','DDS_STARTUP=PASS']
    if m.get('perception_grace_revision')=='bounded_loss_v1':
        grace_ledger=root/'perception_grace_limits.json'
        if digest(grace_ledger)!=m.get('perception_grace_ledger_sha256') or digest(grace_ledger) not in (
                root/'src/M20_sdk_deploy/run_policy/perception_grace_limits.hpp').read_text():
            raise ValueError('Perception grace ledger/header identity incomplete')
        required += ['PERCEPTION_GRACE=PASS']
        required += ['PERCEPTION_GRACE_WORKER=PASS case='+case+' ' for case in
                     ('ascent','descent','flat','recovery','hard_fault')]
    required += ['DDS_JOINT_BOUND=PASS joint='+str(j)+' ' for j in JOINTS]
    if m.get('yaw_ramp_revision')=='yaw_onset_v1':
        yaw_ledger=root/'yaw_command_limits.json'
        if digest(yaw_ledger)!=m.get('yaw_ledger_sha256') or digest(yaw_ledger) not in (
                root/'src/M20_sdk_deploy/run_policy/yaw_command_limits.hpp').read_text():
            raise ValueError('Yaw ramp ledger/header mismatch')
        required += ['YAW_COMMAND_RAMP=PASS','YAW_ACTOR_WORKER=PASS']
    if m.get('policy_target_revision')=='prior_workspace_projection_v1':
        ledger=root/'policy_target_limits.json'
        if digest(ledger)!=m.get('policy_target_ledger_sha256') or digest(ledger) not in (
                root/'src/M20_sdk_deploy/run_policy/policy_target_limits.hpp').read_text():
            raise ValueError('Policy target ledger/header mismatch')
        required += ['POLICY_TARGET_LIMITER=PASS','POLICY_TARGET_WORKER=PASS']
    tracking_revision=m.get('tracking_error_revision')
    if tracking_revision in ('operator_hr_hipx_044_v1','operator_all_hipx_044_v1','operator_all_hipx_065_v1'):
        tracking=json.loads((root/'joint_safety_limits.json').read_text())
        override=tracking.get('tracking_error_overrides',{})
        expected={'hr_hipx_joint'} if tracking_revision=='operator_hr_hipx_044_v1' else {
            leg+'_hipx_joint' for leg in ('fl','fr','hl','hr')}
        if (tracking.get('tracking_error_overrides_confirmed') is not True or
            set(override)!=expected or any(
                override[name].get('maximum_error_rad')!=(.65 if tracking_revision=='operator_all_hipx_065_v1' else .44) or
                override[name].get('reference_position_gain')!=80 for name in expected)):
            raise ValueError('HipX tracking overrides do not match requested revision')
        required += ['HR_HIPX_TRACKING_OVERRIDE=PASS max_error_rad=.44 max_proxy_nm=35.2 ff_limit_nm=32.4'
                     if tracking_revision=='operator_hr_hipx_044_v1' else
                     ('ALL_HIPX_TRACKING_OVERRIDE=PASS joints=0,4,8,12 max_error_rad=.65 max_proxy_nm=52.0 ff_limit_nm=32.4' if tracking_revision=='operator_all_hipx_065_v1' else 'ALL_HIPX_TRACKING_OVERRIDE=PASS joints=0,4,8,12 max_error_rad=.44 max_proxy_nm=35.2 ff_limit_nm=32.4')]
    if any(token not in log for token in required):raise ValueError('Missing safety acceptance result')
    return m,exits[-1].name


def finalize(release,previous,expected_mac,expected_serial):
    check_identity(expected_mac,expected_serial)
    if os.geteuid()!=0:raise ValueError('Finalization requires root to inspect all active controller executables')
    release,previous=release.resolve(),previous.resolve()
    if release==previous or release.parent!=previous.parent:raise ValueError('Expected separate sibling releases')
    manifest,validation=validate(release)
    verify(previous)
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:target=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if target in [release/'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy',previous/'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy']:
            raise ValueError('Custom controller active; entrypoint not changed')
    backup=previous/'launch_aos.before_joint_guard.disabled'
    old_script=previous/'launch_aos.sh'
    if not backup.exists():atomic_write(backup,old_script.read_bytes(),0o444)
    for source,destination,mode in [
        (HERE/'launch_aos.sh',release/'launch_aos.sh',0o755),
        (HERE/'JOINT_SAFETY.md',release/'JOINT_SAFETY.md',0o644),
        (HERE/'summarize_incident.py',release/'validation_tools/summarize_incident.py',0o644)]:
        destination.parent.mkdir(exist_ok=True)
        atomic_write(destination,source.read_bytes(),mode)
    for name in ('START_COMMANDS.md','LIDAR_INPUT_REPAIR_20261118.md','JOINT_SAFETY_DEPLOYMENT_20261118.json'):
        if (HERE/name).is_file():atomic_write(release/name,(HERE/name).read_bytes(),0o644)
    manifest.update(aos_elf_sha256=manifest['joint_guard_elf_sha256'],joint_guard_finalized=True,incident_directory='/var/opt/robot/data/m20_joint_guard_incidents',
                    ready_for_motor_control=False,hardware_locomotion_validation='not_performed')
    atomic_write(release/'AOS_RELEASE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644)
    refresh_checksums(release)
    verify(release)
    wrapper=('#!/usr/bin/env bash\n# Superseded unsafe entry: always use the validated guard release.\n'
             'set -euo pipefail\nexec bash '+shlex.quote(str(release/'launch_aos.sh'))+' "$@"\n')
    # Switch only after the new entry and manifest are complete. No controller
    # process is started by this replacement.
    atomic_write(old_script,wrapper.encode(),0o755)
    old_manifest=json.loads((previous/'AOS_RELEASE_MANIFEST.json').read_text())
    old_manifest['launch_redirects_to']=str(release)
    atomic_write(previous/'AOS_RELEASE_MANIFEST.json',(json.dumps(old_manifest,indent=2)+'\n').encode(),0o644)
    refresh_checksums(previous);verify(previous)
    result={'files_finalized':True,'release':str(release),'old_entry_redirected':str(old_script),
            'native_validation_exit':validation,'elf_sha256':manifest['joint_guard_elf_sha256'],
            'policy_sha256':digest(release/'policy/model.onnx'),'ledger_sha256':digest(release/'joint_safety_limits.json'),
            'incident_directory':manifest['incident_directory'],'custom_motor_control_started':False,
            'physical_validation':'not_performed','live_preflight':'not_run_by_finalizer'}
    atomic_write(release/'artifacts/joint_guard_finalization.json',(json.dumps(result,indent=2)+'\n').encode(),0o644)
    print(json.dumps(result,indent=2))
    return result


def finalize_in_place(release,expected_mac,expected_serial):
    """Admit an already guarded release after a validated safety rebuild."""
    check_identity(expected_mac,expected_serial)
    if os.geteuid()!=0:raise ValueError('Root required to inspect all custom controllers')
    release=release.resolve()
    manifest,validation=validate(release)
    if manifest.get('perception_grace_revision')!='bounded_loss_v1':
        raise ValueError('In-place finalization requires the reviewed perception grace revision')
    for p in Path('/proc').glob('[0-9]*/exe'):
        try:target=p.resolve(strict=True)
        except (OSError,RuntimeError):continue
        if target.name=='rl_deploy' and target!=Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
            raise ValueError('Custom controller active; release not finalized')
    manifest.update(joint_guard_finalized=True,ready_for_motor_control=False,
                    hardware_locomotion_validation='not_performed',latest_safety_validation=validation)
    atomic_write(release/'AOS_RELEASE_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644)
    refresh_checksums(release);verify(release)
    result={'files_finalized':True,'native_validation_exit':validation,
            'elf_sha256':manifest['joint_guard_elf_sha256'],
            'grace_ledger_sha256':manifest['perception_grace_ledger_sha256'],
            'custom_motor_control_started':False,'physical_validation':'not_performed'}
    atomic_write(release/'artifacts/perception_grace_finalization.json',(json.dumps(result,indent=2)+'\n').encode(),0o644)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--previous',type=Path);modes.add_argument('--in-place',action='store_true')
    p.add_argument('--expected-mac',required=True);p.add_argument('--expected-serial',required=True)
    a=p.parse_args()
    if a.in_place:finalize_in_place(a.release,a.expected_mac,a.expected_serial)
    else:finalize(a.release,a.previous,a.expected_mac,a.expected_serial)
