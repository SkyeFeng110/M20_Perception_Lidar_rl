"""Behavioral counterexamples for flat-start repair and command ABI parity."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
sys.path.insert(0, str(ROOT.parent/'mujoco_validation/deploy/mujoco_v135'))
from command_filter import forward_slew
from flat_start_selection import regressions, passed

spec = importlib.util.spec_from_file_location('flat_math', ROOT/'source/rl_training/rl_training/flat_start.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_forward_filter_cpp_numpy_torch_and_full_speed(tmp_path):
    source = tmp_path/'probe.cpp'
    source.write_text('#include "forward_command_ramp.hpp"\n#include <iostream>\n'
        'int main(){m20::ForwardCommandRamp r; float x; while(std::cin>>x) '
        'std::cout<<r.sample(x)<<"\\n"; r.reset(); if(r.sample(1.f)>.031f)return 1;}\n')
    binary = tmp_path/'probe'
    subprocess.run(['g++', '-std=c++17', '-I', str(ROOT.parent/'mujoco_validation/deploy_real'),
        str(source), '-o', str(binary)], check=True)
    requests = np.r_[np.ones(60), np.full(10, .3), np.full(50, -1.), np.zeros(20)]
    native = subprocess.run([str(binary)], input='\n'.join(map(str, requests)), text=True, capture_output=True, check=True)
    cpp = np.array([float(v) for v in native.stdout.split()])
    numpy, tensor, current, t = [], [], 0., torch.tensor([0.])
    for request in requests:
        current = float(forward_slew(current, request, .02, where=np.where, clip=np.clip))
        t = m.filter_forward(t, torch.tensor([request]), .02)
        numpy.append(current); tensor.append(float(t))
    np.testing.assert_allclose(cpp, numpy, atol=2.e-6)
    np.testing.assert_allclose(tensor, numpy, atol=2.e-6)
    assert numpy[33] == pytest.approx(1.) and numpy[-1] == 0.
    assert max(np.diff(numpy[:60])) <= .030001
    assert min(np.abs(numpy[70:90])) < 1.e-6  # reverse crosses zero
    assert float(forward_slew(0., 1., 10., where=np.where, clip=np.clip)) == pytest.approx(.03)


def test_direct_entry_stress_cohort_is_half_ascent_and_keeps_legacy_contract():
    terrain_types = torch.tensor([0, 1, 1, 1, 2, 2, 0, 1, 2, 3, 3, 0])
    flat = torch.tensor([True, False, False, False])
    ascent = torch.tensor([False, True, True, False])
    ids = torch.arange(len(terrain_types))
    legacy = m.direct_entry_stress_mask(ids, terrain_types, flat, ascent, enabled=False)
    repair = m.direct_entry_stress_mask(ids, terrain_types, flat, ascent, enabled=True)
    assert legacy.tolist() == [False, False, False, True, False, False, False, False,
                               False, False, False, True]
    # All ascent rows with even environment ids and flat rows with id % 4 == 0.
    assert repair.tolist() == [True, False, True, False, True, False, False, False,
                               True, False, False, False]
    assert int(repair[ascent[terrain_types]].sum()) == 3


def test_strict_entry_transfer_requires_rear_support_until_both_fronts_are_loaded():
    spec_entry = importlib.util.spec_from_file_location(
        'entry_math', ROOT/'source/rl_training/rl_training/entry_transfer.py')
    entry = importlib.util.module_from_spec(spec_entry); spec_entry.loader.exec_module(entry)
    s = entry.make_state(1, torch.device('cpu'))
    common = dict(x=torch.tensor([[-.05, -.05]]), bottom=torch.tensor([[.16, .16]]),
        force=torch.tensor([[30., 30.]]), wheel_vz=torch.zeros(1, 2),
        root_speed=torch.tensor([.4]), front_speed=torch.tensor([[.4, .4]]),
        pitch=torch.zeros(1), pitch_rate=torch.zeros(1), lower=torch.tensor([0.]),
        rise=torch.tensor([.16]), tread=torch.tensor([.26]), active=torch.tensor([True]),
        reset=torch.tensor([False]), dt=.02, rear_x=torch.tensor([[-.3, -.3]]),
        rear_bottom=torch.tensor([[0., 0.]]), rear_force=torch.tensor([[30., 30.]]))
    s['seen_lower'][:] = True; s['eligible'][:] = True
    safe = entry.transfer_step(s, strict=True, **common)
    assert not bool(safe['rear_early_event']) and bool(safe['rear_hold'])
    # Rear lift before the first front support must add a strict cost.
    s = entry.make_state(1, torch.device('cpu'))
    s['seen_lower'][:] = True; s['eligible'][:] = True
    common['rear_bottom'] = torch.tensor([[.20, .08]])
    unsafe = entry.transfer_step(s, strict=True, **common)
    assert bool(unsafe['rear_early_event']) and float(unsafe['cost']) > 0.


def args():
    return dict(wheel_y=torch.tensor([[.26, -.26, .26, -.26]]), hipx=torch.zeros(1, 4),
        pitch=torch.zeros(1), pitch_rate=torch.zeros(1), action=torch.zeros(1, 16),
        previous_action=torch.zeros(1, 16), command=torch.ones(1), previous_command=torch.tensor([.3]),
        window=torch.zeros(1), active=torch.tensor([True]), reset=torch.tensor([False]), dt=.02)


def test_flat_inward_tuck_is_penalized_symmetrically_and_stairs_are_free():
    a = args(); a['wheel_y'][0, 1] = -.10; a['hipx'][0, 1] = -.33
    a['pitch_rate'][:] = 1.3; a['action'][:] = 20.
    r = m.flat_start_terms(**a)
    assert .1 < float(r['cost']) <= .35 and r['transient'].all()
    a['wheel_y'] = -a['wheel_y'][:, [1, 0, 3, 2]]
    a['hipx'] = -a['hipx'][:, [1, 0, 3, 2]]
    torch.testing.assert_close(m.flat_start_terms(**a)['cost'], r['cost'])
    a['active'][:] = False
    assert m.flat_start_terms(**a)['cost'] == 0
    a['active'][:] = True; a['reset'][:] = True
    assert m.flat_start_terms(**a)['cost'] == 0


def test_normal_progress_has_no_cost_and_partial_full_has_settling_window():
    a = args()
    assert m.flat_start_terms(**a)['cost'] == 0
    a['action'][:, [0, 3]] = 3.
    r = m.flat_start_terms(**a)
    assert r['impulse_cost'] > 0 and r['cost'] > 0
    a['previous_command'][:] = 1.; a['window'] = r['window']
    assert m.flat_start_terms(**a)['cost'] > 0  # not only a single command tick
    a['window'][:] = 0
    assert m.flat_start_terms(**a)['cost'] == 0


def test_physical_selector_rejects_stalling_and_narrow_stance():
    # Reuse real physics evidence rather than fabricate an acceptance row.
    import json
    path = ROOT/'artifacts/flat_start_repair_20261123/reference67100_a15/results.json'
    if not path.is_file():
        pytest.skip('collect physical baseline first')
    rows = json.loads(path.read_text())
    filtered = [r for r in rows if r['case']['forward_filter']]
    assert passed(filtered)
    assert not passed(rows)  # raw stress defect is honestly still unresolved
    import copy
    bad = copy.deepcopy(rows); bad[0]['final_speed_m_s'] = 0.
    assert any('speed' in f for f in regressions(bad, rows))
    bad = copy.deepcopy(rows); bad[0]['front_width_min_m'] -= .10
    assert any('width' in f for f in regressions(bad, rows))


def test_full_stick_bindings_follow_deployed_mapping():
    from key_bindings import MOTION_COMMANDS
    from unittest.mock import patch
    from teleop import TeleopKeyboard
    assert MOTION_COMMANDS['8'] == (1., 0., 0.)
    assert MOTION_COMMANDS['4'] == (0., .5, 0.)
    assert MOTION_COMMANDS['7'] == (0., 0., .6)
    with patch('teleop.X11WindowKeys.open', return_value=None):
        current = TeleopKeyboard(direct=True)
        candidate = TeleopKeyboard()
    current.on_char('8'); candidate.on_char('8')
    assert current.velocity_command()[0] == 1.
    assert candidate.velocity_command()[0] == pytest.approx(.03)
    candidate.clear_motion(); candidate.on_char('7')
    assert candidate.velocity_command()[2] == pytest.approx(.6)


def test_best_archive_uses_measured_flat_and_current_full_stick_evidence(tmp_path):
    import json
    from flat_start_selection import FlatEntryBehaviorSelector
    teacher = ROOT/'logs/rsl_rl/M20-V11.1-TractionTransition/2026-11-20_17-37-01_011240_traction_transition_50000_from48900/model_66800.pt'
    native = ROOT/'artifacts/entry_alternation_20261123/reference66800/native/results.json'
    entry = ROOT/'artifacts/entry_alternation_20261123/reference66800/immediate/results.json'
    flat = ROOT/'artifacts/flat_start_repair_20261123/reference66800_a15/results.json'
    forward = ROOT/'artifacts/flat_start_repair_20261123/transition66800/results.json'
    if not forward.is_file(): pytest.skip('collect production physics first')
    kwargs = dict(flat_reference_results=flat, flat_reference_checkpoint=teacher, forward_reference_results=forward)
    s = FlatEntryBehaviorSelector(tmp_path/'best', teacher, native, entry, **kwargs)
    result = s.consider(teacher, native, entry, flat, forward)
    assert not result['promoted'] and not result['regressions']
    assert not result['flat_start_passed'] and not result['production_passed']
    restored = FlatEntryBehaviorSelector(tmp_path/'restored', teacher, native, entry,
        previous_best=s.export_state(), **kwargs)
    assert restored.export_state()['flat_results']
    # A flat improvement with unchanged entry can now earn an archive; a
    # stop-and-save strategy or a narrowed stance cannot be promoted.
    import copy, shutil
    rows = json.loads(flat.read_text())
    changed = tmp_path/'improved'; changed.mkdir()
    shutil.copy2(flat.parent/'manifest.json', changed/'manifest.json')
    better = copy.deepcopy(rows)
    for r in better:
        r['pitch_peak_deg'] *= .9
    (changed/'results.json').write_text(json.dumps(better))
    assert s.consider(teacher, native, entry, changed/'results.json', forward)['promoted']
    bad = copy.deepcopy(better); bad[0]['final_speed_m_s'] = 0.
    (changed/'results.json').write_text(json.dumps(bad))
    r = s.consider(teacher, native, entry, changed/'results.json', forward)
    assert not r['promoted'] and r['regressions']


def test_real_trace_reader_distinguishes_request_and_filtered_command(tmp_path):
    import json
    from analyze_m20_flat_trace import analyze
    rows = []
    for i in range(3):
        obs = np.zeros(63); obs[5] = -1.; obs[6] = .3
        target = np.zeros((16, 5)); target[0, 1] = .05*i if i < 2 else 10.
        rows.append(dict(event='policy_inference', observation_built=True, inferred=True,
            obs63=obs.tolist(), mono_s=[0., .02, 10.][i], policy_episode=0 if i < 2 else 1,
            policy_index_for_robot=[0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 14, 9, 10, 11, 15],
            default_position_robot=[0.]*16, intended_command_16x5=target.flatten().tolist(),
            requested_velocity3=[1., 0., 0.]))
    trace = tmp_path/'trace.jsonl'
    trace.write_text('\n'.join(json.dumps(r) for r in rows+[dict(event='capture_end', complete=True)]))
    result = analyze(trace)
    assert result['requested_forward_max_m_s'] == 1.
    assert result['commanded_forward_max_m_s'] == .3
    assert result['policy_interval_p99_ms'] == pytest.approx(20.)
    assert result['front_target_jump_peak_deg'] == pytest.approx(np.degrees(.05))
    trace.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='incomplete'): analyze(trace)
