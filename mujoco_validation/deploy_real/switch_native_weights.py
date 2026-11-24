"""Export a continued-training actor, then cold-swap only a guarded release's policy.

The caller explicitly inherits the existing training/execution contract. No
configuration search, controller compilation, ROS/DDS publisher or motor start.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from export_native_actor import replace_weights
    torch.set_num_threads(1)
    if args.output.exists():
        raise ValueError('Export destination already exists')
    before = torch.load(args.previous_checkpoint, map_location='cpu', weights_only=True)
    after = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    baseline = onnx.load(args.previous_policy / 'model.onnx')
    metadata = {item.key: item.value for item in baseline.metadata_props}
    if metadata['checkpoint_sha256'] != sha(args.previous_checkpoint):
        raise ValueError('Previous checkpoint identity differs')
    model = replace_weights(baseline, before['actor_state_dict'], after['actor_state_dict'])
    metadata.update(checkpoint_path=str(args.checkpoint.resolve()),
                    checkpoint_sha256=sha(args.checkpoint),
                    m20_checkpoint_iteration=str(after['iter']),
                    m20_export_method='continued_training_actor_tensor_replacement')
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=['CPUExecutionProvider'])
    obs = np.zeros((1, 63), dtype=np.float32)
    obs[0, 5] = -1
    value = torch.from_numpy(obs)
    with torch.inference_mode():
        for layer in (0, 2, 4, 6):
            value = torch.nn.functional.linear(value, after['actor_state_dict'][f'mlp.{layer}.weight'],
                                               after['actor_state_dict'][f'mlp.{layer}.bias'])
            if layer != 6:
                value = torch.nn.functional.elu(value)
    actual = runtime.run(['actions'], {'obs': obs})[0]
    if actual.shape != (1, 16) or not np.isfinite(actual).all() or not np.allclose(
            actual, value.numpy(), atol=2e-5, rtol=1e-4):
        raise ValueError('Converted model failed minimal inference check')
    args.output.mkdir(parents=True)
    shutil.copy2(args.previous_policy / 'model.onnx.feature_contract', args.output)
    shutil.copytree(args.previous_policy / 'training_params', args.output / 'training_params')
    onnx.save(model, args.output / 'model.onnx')
    policy_sha = sha(args.output / 'model.onnx')
    (args.output / 'model.onnx.sha256').write_text(policy_sha + '  model.onnx\n')
    receipt = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                   iteration=int(after['iter']), deployment_sha256=policy_sha,
                   previous_policy_sha256=sha(args.previous_policy / 'model.onnx'),
                   execution_contract='inherited unchanged by explicit user instruction',
                   configuration_reaudit='skipped by explicit user instruction',
                   actor_tensors_replaced=8, graph_and_opset_unchanged=True,
                   minimal_inference_max_error=float(np.abs(actual-value.numpy()).max()),
                   full_replay_validation='skipped', hardware_locomotion_validation='not_performed',
                   no_joint_commands=True)
    save(args.output / 'model.onnx.provenance.json', receipt)
    print(json.dumps(receipt, indent=2))


def apply(args):
    from release_utils import refresh_checksums
    root = args.release.resolve(strict=True)
    if os.geteuid() != 0:
        raise ValueError('Root required for controller process identity and deployment lock')
    if Path('/proc/device-tree/serial-number').read_text().strip('\0\n') != args.serial:
        raise ValueError('Wrong board serial')
    if Path('/sys/class/net/eth2/address').read_text().strip() != args.mac:
        raise ValueError('Wrong board MAC')
    with open('/run/lock/m20-sim2real-control.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                if (process / 'comm').read_text().strip() == 'rl_deploy':
                    if (process / 'exe').resolve(strict=True) != Path('/opt/robot/share/rl_deploy/bin/rl_deploy'):
                        raise ValueError('Custom controller is active; stop it before changing weights')
            except FileNotFoundError:
                continue
        manifest = json.loads((root / 'AOS_RELEASE_MANIFEST.json').read_text())
        elf = root / 'install/m20_sdk_deploy/lib/m20_sdk_deploy/rl_deploy'
        if (manifest.get('joint_safety_revision') != 'joint_publish_guard_v1' or
                manifest.get('native_rebuild_required') is not False or
                manifest.get('offline_validation_complete') is not True or
                sha(elf) != manifest.get('joint_guard_elf_sha256')):
            raise ValueError('Verified joint guard binary required')
        receipt = json.loads((args.policy / 'model.onnx.provenance.json').read_text())
        if sha(root / 'policy/model.onnx') != receipt['previous_policy_sha256']:
            raise ValueError('Current policy differs from the exported baseline')
        if sha(args.policy / 'model.onnx') != receipt['deployment_sha256']:
            raise ValueError('Uploaded policy checksum differs')
        # One existing ARM runner, synthetic observations only; no DDS interface.
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        if available < 512 * 1024**2 or shutil.disk_usage(root).free < 256 * 1024**2:
            raise ValueError('Need 512 MiB free RAM and 256 MiB disk for one bounded loader and rollback')
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(root / 'src/M20_sdk_deploy/third_party/onnxruntime/arm/lib') + ':' + str(root / 'install/drdds/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        env['M20_V11_FEATURE_CONTRACT'] = (args.policy / 'model.onnx.feature_contract').read_text().strip()
        completed = subprocess.run(['nice', '-n', '10', 'taskset', '-c', '6,7',
                                    str(elf.parent / 'policy_smoke_test'), str(args.policy / 'model.onnx')],
                                   env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        print(completed.stdout, flush=True)
        if completed.returncode:
            raise ValueError('ARM model loader failed; current deployment untouched')
        subprocess.run(['sha256sum', '-c', '--quiet', 'RELEASE_SHA256SUMS'], cwd=root, check=True)
        backup = Path(tempfile.mkdtemp(prefix='policy_backup_', dir=root.parent))
        for name in ('AOS_RELEASE_MANIFEST.json', 'RELEASE_SHA256SUMS'):
            shutil.copy2(root / name, backup / name)
        if (root / 'MODEL_WEIGHT_SWITCH.json').exists():
            shutil.copy2(root / 'MODEL_WEIGHT_SWITCH.json', backup / 'MODEL_WEIGHT_SWITCH.json')
        staged = root / '.policy_update'
        if staged.exists():
            raise ValueError('Staging directory already exists')
        shutil.copytree(args.policy, staged)
        switched = False
        try:
            os.replace(root / 'policy', backup / 'policy')
            switched = True
            os.replace(staged, root / 'policy')
            manifest.update(policy_sha256=receipt['deployment_sha256'], checkpoint_sha256=receipt['checkpoint_sha256'],
                            checkpoint_iteration=receipt['iteration'], source_policy=receipt['checkpoint'],
                            weights_only_update=True, full_revalidation_after_weight_update='skipped by user instruction',
                            ready_for_motor_control=False, hardware_locomotion_validation='not_performed')
            save(root / 'AOS_RELEASE_MANIFEST.json', manifest)
            receipt.update(board_serial=args.serial, board_mac=args.mac, release=str(root),
                           backup=str(backup), joint_guard_elf_sha256=sha(elf),
                           arm_model_load='passed', current_custom_controller_active=False,
                           resource_available_bytes=available, switched_at_unix=time.time())
            save(root / 'MODEL_WEIGHT_SWITCH.json', receipt)
            refresh_checksums(root)
            subprocess.run(['sha256sum', '-c', '--quiet', 'RELEASE_SHA256SUMS'], cwd=root, check=True)
        except BaseException:
            if switched:
                if (root / 'policy').exists():
                    shutil.rmtree(root / 'policy')
                os.replace(backup / 'policy', root / 'policy')
                for name in ('AOS_RELEASE_MANIFEST.json', 'RELEASE_SHA256SUMS'):
                    shutil.copy2(backup / name, root / name)
                if (backup / 'MODEL_WEIGHT_SWITCH.json').exists():
                    shutil.copy2(backup / 'MODEL_WEIGHT_SWITCH.json', root / 'MODEL_WEIGHT_SWITCH.json')
                else:
                    (root / 'MODEL_WEIGHT_SWITCH.json').unlink(missing_ok=True)
            raise
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    local = commands.add_parser('export')
    for name in ('previous-checkpoint', 'checkpoint', 'previous-policy', 'output'):
        local.add_argument('--' + name, type=Path, required=True)
    robot = commands.add_parser('apply')
    robot.add_argument('--release', type=Path, required=True)
    robot.add_argument('--policy', type=Path, required=True)
    robot.add_argument('--serial', required=True)
    robot.add_argument('--mac', required=True)
    args = parser.parse_args()
    (export if args.command == 'export' else apply)(args)
