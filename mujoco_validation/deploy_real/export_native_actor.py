"""Replace actor tensors in a verified native ONNX baseline; no robot actions.

The baseline ONNX must match every MLP tensor of the original checkpoint.
Keeps its tested graph, opset and I/O names. Emits a matching native execution
header from saved training parameters, including the corrected nominal pose.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch


NAMES = tuple(f'{layer}.{field}' for layer in (0, 2, 4, 6) for field in ('weight', 'bias'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_weights(model, baseline, candidate):
    if {t.name for t in model.graph.initializer} != set(NAMES):
        raise ValueError('Unexpected baseline ONNX initializers')
    updated = copy.deepcopy(model)
    for old, new in zip(model.graph.initializer, updated.graph.initializer):
        name = 'mlp.' + old.name
        value = numpy_helper.to_array(old)
        expected = baseline[name].detach().cpu().numpy()
        target = candidate[name].detach().cpu().numpy()
        if not np.array_equal(value, expected):
            raise ValueError('Baseline checkpoint/ONNX weights differ: ' + name)
        if target.dtype != np.float32 or target.shape != value.shape or not np.isfinite(target).all():
            raise ValueError('Invalid candidate actor tensor: ' + name)
        new.CopyFrom(numpy_helper.from_array(target.copy(), name=old.name))
    return updated


def native_contract(selected):
    names = ('base_ang_vel', 'projected_gravity', 'velocity_commands', 'joint_pos',
             'joint_vel', 'actions', 'height_scan')
    widths = (3, 3, 3, 16, 16, 16, 6)
    scales = (.25, 1., 1., 1., .05, 1., 1.)
    if selected.period != .02 or selected.perception['forward_command_threshold'] != .05:
        raise ValueError('Native control period or forward feature gate differs')
    if [selected.obs_cfg[name]['scale'] for name in names] != list(scales):
        raise ValueError('Unsupported native observation scales')
    low, high = [], []
    for name, width in zip(names, widths):
        term = selected.obs_cfg[name]
        lo, hi = np.asarray(term['clip']) * term['scale']
        low.extend([float(lo)] * width)
        high.extend([float(hi)] * width)
    permutation = [i for leg in range(4) for i in (leg*3, leg*3+1, leg*3+2, 12+leg)]
    contract = dict(default_policy=selected.nominal.tolist(),
                    default_robot=selected.nominal[permutation].tolist(),
                    obs_low=low, obs_high=high, action_clip=selected.clip_actions)
    header = '#pragma once\n#include <array>\nnamespace m20 { namespace trained {\n'
    for key in ('default_policy', 'default_robot', 'obs_low', 'obs_high'):
        values = contract[key]
        header += 'inline constexpr std::array<float, %d> %s = {%s};\n' % (
            len(values), key, ','.join(format(float(v), '.9f')+'f' for v in values))
    header += 'inline constexpr float action_clip = %.9ff;\n}}\n' % selected.clip_actions
    return contract, header


def main():
    root = Path(__file__).resolve().parents[2]/'M20_perception_Lidar_rl'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-onnx', type=Path, required=True)
    p.add_argument('--baseline-checkpoint', type=Path, default=root/'interview_prep/model_35300.pt')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config-dir', type=Path, required=True)
    p.add_argument('--replay', type=Path, default=root/'data/m20_recovery/reference_observations_v2.npz')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite an existing export directory')
    sys.path.insert(0, str(root.parent/'mujoco_validation/deploy/mujoco_v135'))
    from policy import SavedPolicy, JOINT_NAMES
    torch.set_num_threads(1)
    original = SavedPolicy(a.baseline_checkpoint)
    selected = SavedPolicy(a.checkpoint, a.config_dir)
    for field in ('nominal', 'action_scale', 'kp', 'kd', 'effort', 'armature'):
        if not np.array_equal(getattr(original, field), getattr(selected, field)):
            raise ValueError('Candidate execution contract differs: ' + field)
    if original.obs_cfg != selected.obs_cfg or original.period != selected.period:
        raise ValueError('Candidate observation/control-period contract differs')
    baseline = torch.load(a.baseline_checkpoint, map_location='cpu', weights_only=True)
    candidate = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = onnx.load(a.baseline_onnx)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if metadata.get('m20_feature_contract') != selected.perception['feature_contract']:
        raise ValueError('Native baseline feature contract differs')
    if metadata.get('joint_names') != ','.join(JOINT_NAMES):
        raise ValueError('Native joint order differs')
    native_nominal = np.fromstring(metadata['default_joint_pos'], sep=',')
    historical_nominal = np.array([0,-.6,1]*2+[0,.6,-1]*2+[0]*4)
    if not np.allclose(native_nominal, historical_nominal, atol=1.e-5, rtol=0):
        raise ValueError('Unknown historical native nominal pose')
    for key, expected in (('action_scale', selected.action_scale),
                          ('joint_stiffness', selected.kp), ('joint_damping', selected.kd)):
        if not np.allclose(np.fromstring(metadata[key], sep=','), expected, atol=1.e-5, rtol=0):
            raise ValueError('Native metadata differs from saved config: ' + key)
    updated = replace_weights(model, baseline['actor_state_dict'], candidate['actor_state_dict'])
    contract, header = native_contract(selected)
    metadata['default_joint_pos'] = ','.join(str(float(v)) for v in selected.nominal)
    metadata.update(checkpoint_path=str(a.checkpoint.resolve()), checkpoint_sha256=sha256(a.checkpoint),
                    m20_checkpoint_iteration=str(candidate['iter']),
                    m20_baseline_onnx_sha256=sha256(a.baseline_onnx),
                    m20_export_method='verified_graph_actor_tensor_replacement')
    onnx.helper.set_model_props(updated, metadata)
    onnx.checker.check_model(updated)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(updated.SerializeToString(), opts, providers=['CPUExecutionProvider'])
    if [(i.name, i.shape) for i in session.get_inputs()] != [('obs', [1, 63])]:
        raise ValueError('Native input ABI differs')
    if [(i.name, i.shape) for i in session.get_outputs()] != [('actions', [1, 16])]:
        raise ValueError('Native output ABI differs')
    with np.load(a.replay, allow_pickle=False) as replay:
        observations = replay['observation'].copy()
    if observations.ndim != 2 or observations.shape[1] != 63 or not np.isfinite(observations).all():
        raise ValueError('Invalid physical replay observations')
    maximum_error = 0.
    clipping_samples = 0
    with torch.inference_mode():
        for obs in observations:
            expected = selected.actor(torch.from_numpy(obs[None])).numpy()
            actual = session.run(['actions'], {'obs': obs[None]})[0]
            if not np.isfinite(actual).all() or not np.allclose(expected, actual, atol=2.e-5, rtol=1.e-4):
                raise ValueError('ONNX actor numerical parity failed')
            maximum_error = max(maximum_error, float(np.abs(expected-actual).max()))
            clipping_samples += int((np.abs(expected) > selected.clip_actions).any())
    a.output.mkdir(parents=True)
    target = a.output/'model.onnx'
    onnx.save(updated, target)
    digest = sha256(target)
    (a.output/'model.onnx.sha256').write_text(digest+'  model.onnx\n')
    (a.output/'model.onnx.feature_contract').write_text(selected.perception['feature_contract']+'\n')
    (a.output/'trained_execution_contract.hpp').write_text(header)
    (a.output/'trained_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (a.output/'training_params').mkdir()
    for name in ('env.yaml', 'agent.yaml'):
        shutil.copy2(a.config_dir/name, a.output/'training_params'/name)
    report = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha256(a.checkpoint),
        iteration=int(candidate['iter']), baseline_checkpoint_sha256=sha256(a.baseline_checkpoint),
        baseline_onnx_sha256=sha256(a.baseline_onnx), deployment_sha256=digest,
        baseline_actor_tensor_exact_matches=len(NAMES), graph_and_opset_unchanged=True,
        execution_contract=contract, historical_native_nominal=native_nominal.tolist(),
        nominal_pose_corrected_to_saved_config=True, physical_replay_samples=len(observations),
        execution_header_sha256=sha256(a.output/'trained_execution_contract.hpp'),
        max_absolute_inference_error=maximum_error, action_clipping_samples=clipping_samples,
        config_dir=str(a.config_dir.resolve()), feature_contract=selected.perception['feature_contract'],
        config_sha256={n:sha256(a.config_dir/n) for n in ('env.yaml', 'agent.yaml')},
        config_provenance='explicit compatible run config; checkpoint origin not proven by identical log copy',
        no_joint_commands=True, hardware_validation='not_performed')
    (a.output/'model.onnx.provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
