"""PPO segments with independent behavior reports; no simulator imports."""
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / 'scripts/repair_workstation_limits.json'
TASK = 'M20-Stair-V11.1-AscentRecovery-v0'
REFINE_TASK = 'M20-Stair-V11.1-RecoveryRefine-v0'
TARGETED_TASK = 'M20-Stair-V11.1-TargetedCorrection-v0'
TARGETED_V2_TASK = 'M20-Stair-V11.1-TargetedCorrectionV2-v0'
STEP_SUPPORT_TASK = 'M20-Stair-V11.1-StepSupport-v0'
STEP_RETENTION_TASK = 'M20-Stair-V11.1-StepRetention-v0'
TRACTION_TRANSITION_TASK = 'M20-Stair-V11.1-TractionTransition-v0'
ENTRY_TRANSFER_TASK = 'M20-Stair-V11.1-EntryTransfer-v0'
ENTRY_SEQUENCE_TASK = 'M20-Stair-V11.1-EntrySequenceRepair-v0'
FLAT_POSTURE_TASK = 'M20-Stair-V11.1-FlatPosture-v0'
ROUTE_HOLD_TASK = 'M20-Stair-V11.1-RouteHold-v0'


class TrainingInterrupted(KeyboardInterrupt):
    """Interrupt training without entering Isaac Kit's native signal teardown."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'User requested {signal.Signals(signum).name}')


def install_interrupt_handlers(cleanup_seconds=20):
    """Call after AppLauncher installs its own handlers; headless repair only.

    The first signal unwinds Python so status/logs and evaluation children can
    be finalized. A second signal, or the deadline, terminates native cleanup
    even if Kit stops returning to Python. No partial candidate is promoted.
    """
    def interrupt(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(cleanup_seconds)
        raise TrainingInterrupted(signum)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


def run_evaluation(command, *, env, stdout, timeout=600):
    """Keep the evaluator and its benchmark children in one cancellable group."""
    process = subprocess.Popen(command, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A benchmark descendant may survive even after its parent exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        raise
    return subprocess.CompletedProcess(command, returncode)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_session(task, total_updates, segment_updates, baseline_path, output_path):
    limits = json.loads(LEDGER.read_text())
    if task not in (TASK, REFINE_TASK, TARGETED_TASK, TARGETED_V2_TASK, STEP_SUPPORT_TASK,
                    STEP_RETENTION_TASK, TRACTION_TRANSITION_TASK, ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
        raise ValueError('Guarded training is restricted to recovery tasks')
    if type(total_updates) is not int or not 1 <= total_updates <= limits['max_guarded_total_updates']:
        raise ValueError('Invalid guarded total update count')
    if type(segment_updates) is not int or not 1 <= segment_updates <= limits['max_guard_interval']:
        raise ValueError('Guard interval exceeds the workstation ledger')
    if not baseline_path or not Path(baseline_path).is_file():
        raise ValueError('Guarded training requires original-model baseline results')
    baseline = Path(baseline_path)
    rows = json.loads(baseline.read_text())
    gate_spec = importlib.util.spec_from_file_location('m20_session_acceptance', ROOT/'scripts/tools/m20_repair_acceptance.py')
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    if not isinstance(rows, list):
        raise ValueError('Baseline must be a list of physical benchmark cases')
    failures = gate.validate_rows(rows, json.loads(gate.CRITERIA.read_text()))
    if failures:
        raise ValueError('Invalid original-model baseline: ' + '; '.join(failures))
    source = (hashlib.sha256(Path(os.environ.get('M20_RECOVERY_TEACHER', ROOT/'interview_prep/model_49998.pt')).read_bytes()).hexdigest()
              if task == ENTRY_SEQUENCE_TASK else
              json.loads((ROOT/'interview_prep/CHECKPOINT.json').read_text())['sha256'])
    if task in (FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
        teacher = Path(os.environ.get('M20_RECOVERY_TEACHER', ''))
        if not teacher.is_file():
            raise ValueError('FlatPosture requires its measured stair teacher')
        source = hashlib.sha256(teacher.read_bytes()).hexdigest()
    for profile in ('standing', 'keyboard'):
        manifest = json.loads((baseline.parent/profile/'manifest.json').read_text())
        if manifest.get('checkpoint_sha256') != source:
            raise ValueError('Retention baseline must come from the selected task seed')
    if not output_path:
        raise ValueError('Guarded training requires a session output directory')
    output = Path(output_path)
    if (output / 'status.json').exists():
        raise ValueError('Refusing to overwrite a guarded session')
    return limits


def resource_check(path, limits):
    free_gib = shutil.disk_usage(path).free / 2**30
    available_gib = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available_gib = int(line.split()[1]) / 2**20
    if free_gib < limits['minimum_free_disk_gib']:
        raise RuntimeError(f'Disk reserve reached: {free_gib:.2f} GiB free')
    if available_gib is None or available_gib < limits['minimum_available_ram_gib']:
        raise RuntimeError(f'Insufficient available RAM: {available_gib} GiB')
    return dict(free_disk_gib=free_gib, available_ram_gib=available_gib)


def acoustic_comparison(row, baseline, speed, baseline_speed):
    current = (row.get('acoustic_proxies') or {}).get('phases', {}).get('moving')
    original = (baseline.get('acoustic_proxies') or {}).get('phases', {}).get('moving')
    available = current is not None and original is not None
    matched = (speed is not None and baseline_speed is not None and baseline_speed > 0
               and abs(speed/baseline_speed-1) <= .05)
    ratios = {}
    if available and matched:
        for key in ('touchdown_approach_rms_m_s', 'touchdown_load_rate_rms_N_s',
                    'leg_action_reversal_rms', 'wheel_action_reversal_rms', 'torque_rate_rms_Nm_s'):
            a, b = current.get(key), original.get(key)
            if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and math.isfinite(a) and math.isfinite(b) and b > 0):
                ratios[key] = a/b
    return dict(baseline_proxy_available=available, actual_speed_matched=bool(matched),
                speed_tolerance_fraction=.05, moving_proxy_ratios=ratios)


def evaluation_summary(report, baseline_path):
    """Validate results and retain case-level trends without stopping for behavior.

    Every behavior failure is advisory by user instruction. Report new falls,
    bypass, startup failures, stalls and lost baseline-complete ascents with
    greater visibility so a long run cannot hide them behind total reward.
    """
    candidates = report.get('candidates') or []
    if len(candidates) != 1:
        raise RuntimeError('Behavior report must contain one candidate for guarded training')
    candidate = candidates[0]
    results_path = Path(candidate.get('results', ''))
    baseline_path = Path(baseline_path)
    if not results_path.is_file() or not baseline_path.is_file():
        raise RuntimeError('Behavior report is missing candidate or baseline results')
    rows = json.loads(results_path.read_text())
    baseline_rows = json.loads(baseline_path.read_text())
    gate_spec = importlib.util.spec_from_file_location('m20_summary_acceptance', ROOT/'scripts/tools/m20_repair_acceptance.py')
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    criteria = json.loads(gate.CRITERIA.read_text())
    invalid = gate.validate_rows(rows, criteria) + gate.validate_rows(baseline_rows, criteria)
    if invalid:
        raise RuntimeError('Invalid behavior result data: ' + '; '.join(invalid))
    baseline_by_key = {gate.case_key(row): row for row in baseline_rows}
    failures, cases = [], []
    for row in rows:
        key = gate.case_key(row)
        name = '/'.join(map(str, key))
        original = baseline_by_key[key]
        if row['fell'] or row['observed_bypass'] or not row['policy_entered']:
            failures.append(name + ': fall, bypass or startup failure')
        if row['stall_detected'] and not original['stall_detected']:
            failures.append(name + ': new post-entry route stall')
        if row['terrain'] == 'ascent':
            if original.get('whole_body_top_supported') and not row['whole_body_top_supported']:
                failures.append(name + ': lost complete ascent')
            if original.get('all_treads_supported') and not row.get('all_treads_supported'):
                failures.append(name + ': lost complete tread support')
        speed_field = 'later_stairs_route_velocity_x' if row['terrain'] == 'ascent' else 'route_mean_velocity_x'
        speed, baseline_speed = row.get(speed_field), original.get(speed_field)
        cases.append(dict(case=name, speed=speed, baseline_speed=baseline_speed,
                          speed_to_command=speed/row['command_speed'] if speed is not None else None,
                          speed_to_baseline=speed/baseline_speed
                              if speed is not None and baseline_speed is not None and baseline_speed > 0 else None,
                          reach_p95=row['p95_max_abs_reach'],
                          idle_yaw_deg=row['after_c_idle_yaw_max_change_deg'],
                          moving_yaw_deg=row['forward_yaw_max_change_deg'],
                          lateral_m=row['forward_lateral_max_abs'],
                          yaw_min_deg=row.get('forward_yaw_min_change_deg'),
                          yaw_max_deg=row.get('forward_yaw_max_signed_change_deg'),
                          lateral_min_m=row.get('forward_lateral_min_m'),
                          lateral_max_m=row.get('forward_lateral_max_m'),
                          fell=row['fell'], stalled=row['stall_detected'],
                          whole_body_top_supported=row['whole_body_top_supported'],
                          all_treads_supported=row.get('all_treads_supported'),
                          skipped_treads=row.get('skipped_treads'),
                          acoustic_proxies=row.get('acoustic_proxies'),
                          acoustic_comparison=acoustic_comparison(row, original, speed, baseline_speed)))
    return dict(capability_regressions=failures, cases=cases)


def evaluation_scalars(summary):
    """Fixed physical cases on the additional-update axis, never train reward."""
    values = {'MuJoCo/retention_passed': int(bool(summary['retention_passed'])),
              'MuJoCo/repair_passed': int(bool(summary['repair_passed']))}
    stairs = [r for r in summary['cases'] if '/ascent/' in r['case']]
    values['MuJoCo/ascent_top_cases'] = sum(bool(r['whole_body_top_supported']) for r in stairs)
    values['MuJoCo/all_treads_cases'] = sum(bool(r['all_treads_supported']) for r in stairs)
    # A whole-flight pass is sparse. Keep measured gap trends visible while
    # every case still fails the strict criterion; never alter acceptance.
    gaps = [[t for side in (r.get('skipped_treads') or []) for t in side] for r in stairs]
    for row in summary.get('traction_cases', []):
        label = '/'.join(str(row[k]) for k in ('terrain', 'tread', 'speed', 'friction'))
        for key in ('early_ascent_slip_p95', 'early_ascent_slip_mean',
                    'early_loaded_contact_samples', 'steady_tracking_ratio', 'rear_support_fraction'):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[f'Traction/{label}/{key}'] = value
    values['MuJoCo/b1_skip_cases'] = sum(1 in row for row in gaps)
    values['MuJoCo/later_skip_cases'] = sum(any(t >= 2 for t in row) for row in gaps)
    values['MuJoCo/front_skipped_treads'] = sum(len(row) for row in gaps)
    if 'flat_start_cases' in summary:
        values['FlatStart/passed'] = int(summary['flat_start_passed'])
        for row in summary['flat_start_cases']:
            for key in ('pitch_peak_deg', 'pitch_rate_peak_deg_s', 'fr_inward_y_change_m',
                        'yaw_peak_deg', 'lateral_peak_m',
                        'front_width_min_m', 'final_speed_m_s', 'hipx_target_jump_peak_deg',
                        'sagittal_target_jump_peak_deg', 'wheel_target_jump_peak_rad_s'):
                values[f'FlatStart/{row["case"]["name"]}/{key}'] = row[key]
            values[f'FlatStart/{row["case"]["name"]}/front_hipx_peak_deg'] = max(row['front_hipx_angle_peak_deg'])
        values['Production/passed'] = int(summary['production_passed'])
        for row in summary['production_cases']:
            name = f'{row["profile"]}/{row["terrain"]}/{row["tread"]}/{row["command_speed"]}'
            for key in ('forward_yaw_max_change_deg', 'forward_lateral_max_abs', 'route_mean_velocity_x',
                        'later_stairs_route_velocity_x', 'p95_max_abs_reach', 'all_treads_supported'):
                if isinstance(row.get(key), (float, int)):
                    values[f'Production/{name}/{key}'] = row[key]
            for key, value in row.get('entry_transfer', {}).items():
                if isinstance(value, (float, int)):
                    values[f'Production/{name}/entry/{key}'] = value
    if 'behavior_selection' in summary:
        selection = summary['behavior_selection']
        values['MuJoCo/best_candidate_promoted'] = int(selection['promoted'])
        values['MuJoCo/acquired_behavior_regressions'] = len(selection['regressions'])
        if 'route_hold_passed' in selection:
            values['RouteHold/passed'] = int(selection['route_hold_passed'])
            values['RouteHold/best_score'] = selection['best_rank'][1]
            values['RouteHold/unresolved_targets'] = len(selection['route_unresolved'])
            values['RouteHold/steering_weight'] = summary['behavior_feedback']['route_steering_weight']
            values['RouteHold/flat_preservation_multiplier'] = summary['behavior_feedback']['route_flat_multiplier']
            values['RouteHold/early_traction_multiplier'] = summary['behavior_feedback']['early_traction_multiplier']
            values['RouteHold/early_traction_passed'] = int(selection['traction_passed'])
            values['RouteHold/early_traction_repair_passed'] = int(selection['traction_repair_passed'])
            values['RouteHold/early_traction_slip_score'] = selection['early_traction_score']
            values['RouteHold/best_early_traction_slip_score'] = selection['best_early_traction_score']
            for key in ('descent_error_ratio', 'low_mu_error_ratio', 'combined_error', 'descent_repair_passed',
                        'turn_error_ratio', 'turn_repair_passed'):
                if key in selection:
                    values['DescentTractionV2/'+key] = selection[key]
            feedback = summary['behavior_feedback']
            for key, value in feedback.get('v2_keep_multipliers', {}).items():
                values['DescentTractionV2/'+key+'_keep_multiplier'] = value
            for key in ('v2_descent_reward_multiplier', 'v2_slip_reward_multiplier', 'v2_turn_reward_multiplier'):
                if key in feedback:
                    values['DescentTractionV2/'+key] = feedback[key]
        else:
            values['MuJoCo/best_candidate_all_treads_cases'] = selection['best_rank'][0]
        values['Retention/success_anchor_multiplier'] = summary['behavior_feedback']['success_anchor_multiplier']
    for row in summary['cases']:
        for key in ('speed', 'speed_to_command', 'reach_p95', 'idle_yaw_deg', 'moving_yaw_deg',
                    'lateral_m', 'yaw_min_deg', 'yaw_max_deg', 'lateral_min_m', 'lateral_max_m',
                    'fell', 'stalled', 'whole_body_top_supported', 'all_treads_supported'):
            value = row.get(key)
            if isinstance(value, (float, int)) and math.isfinite(value):
                values[f'MuJoCo/{row["case"]}/{key}'] = value
        comparison = row.get('acoustic_comparison', {})
        if comparison.get('baseline_proxy_available'):
            values[f'AcousticComparison/{row["case"]}/actual_speed_matched'] = int(comparison['actual_speed_matched'])
            for key, value in comparison['moving_proxy_ratios'].items():
                values[f'AcousticComparison/{row["case"]}/ratio_to_baseline/{key}'] = value
    for row in summary.get('front_turn_cases', []):
        name = str((row['forward'], row['yaw'], row['friction']))
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[f'FrontTurn/{name}/{key}'] = value
    rows = summary['cases']+summary.get('supplementary', [])
    for row in rows:
        for phase, metrics in (row.get('acoustic_proxies') or {}).get('phases', {}).items():
            for key, value in metrics.items():
                if isinstance(value, (float, int)) and math.isfinite(value):
                    values[f'AcousticProxy/{row["case"]}/{phase}/{key}'] = value
    for row in summary.get('supplementary', []):
        for key in ('goal_reached', 'fell', 'observed_bypass'):
            values[f'MuJoCoSupplement/{row["case"]}/{key}'] = int(row[key])
        for key, value in row.get('retention_pose', {}).items():
            values[f'DescentRetention/{row["case"]}/{key}'] = value
    if 'descent_retained' in summary:
        values['DescentRetention/passed'] = int(summary['descent_retained'])
    return values


def write_evaluation_tensorboard(summary, log_dir):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(Path(log_dir)/'mujoco_eval')) as writer:
        for tag, value in evaluation_scalars(summary).items():
            writer.add_scalar(tag, value, summary['completed_updates'])


def learn_guarded(runner, *, total_updates, segment_updates, baseline_path, output_path, log_dir,
                  task=TASK, evaluation_every_updates=None, evaluation_profile='full'):
    """Preserve optimizer and continue learning through valid behavior failures.

    RSL 5 records the last update index, not the next index. We explicitly
    advance it between calls so segment boundaries never repeat an iteration.
    Only segment-final weights are written; baseline/user assets are untouched.
    """
    limits = validate_session(task, total_updates, segment_updates, baseline_path, output_path)
    if evaluation_every_updates is None:
        evaluation_every_updates = segment_updates
    if (type(evaluation_every_updates) is not int or evaluation_every_updates < segment_updates
            or evaluation_every_updates % segment_updates):
        raise ValueError('Evaluation interval must be a multiple of the guarded segment size')
    if task in (ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK) and evaluation_profile != 'entry':
        raise ValueError('EntrySequence uses the entry profile with full-stick and flat probes')
    if evaluation_profile not in ('full', 'entry'):
        raise ValueError('Unknown guarded evaluation profile')
    if evaluation_profile == 'entry' and task not in (ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
        raise ValueError('Entry evaluation profile is restricted to EntryTransfer')
    flat_repair = hasattr(getattr(runner, 'alg', None), 'flat_reference_results')
    early_58098 = hasattr(getattr(runner, 'alg', None), 'descent_reference_results')
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir).resolve()
    baseline_path = Path(baseline_path).resolve()
    selector = None
    if task in (STEP_RETENTION_TASK, TRACTION_TRANSITION_TASK, ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
        from behavior_selection import BehaviorSelector
        selector_class, selection_kwargs = BehaviorSelector, {}
        if task in (ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
            from entry_selection import EntryBehaviorSelector
            selector_class = EntryBehaviorSelector
            selection_kwargs['entry_results'] = runner.alg.entry_reference_results
            if evaluation_profile == 'full':
                selection_kwargs['traction_results'] = runner.alg.traction_reference_results
            if flat_repair:
                from flat_start_selection import FlatEntryBehaviorSelector
                selector_class = FlatEntryBehaviorSelector
                selection_kwargs.update(flat_reference_results=runner.alg.flat_reference_results,
                    flat_reference_checkpoint=runner.alg.flat_reference_checkpoint,
                    forward_reference_results=runner.alg.forward_reference_results)
        elif task == TRACTION_TRANSITION_TASK and evaluation_profile == 'full':
            from traction_selection import TractionBehaviorSelector
            selector_class = TractionBehaviorSelector
            selection_kwargs['traction_results'] = runner.alg.traction_reference_results
        if task == FLAT_POSTURE_TASK:
            from flat_posture_selection import FlatPostureBehaviorSelector
            selector_class = FlatPostureBehaviorSelector
        if task == ROUTE_HOLD_TASK:
            from route_hold_selection import RouteHoldBehaviorSelector
            selector_class = RouteHoldBehaviorSelector
            selection_kwargs['traction_results'] = runner.alg.traction_reference_results
            if early_58098:
                from early_traction_selection import EarlyTractionBehaviorSelector
                selector_class = EarlyTractionBehaviorSelector
                selection_kwargs['descent_results'] = runner.alg.descent_reference_results
                if getattr(runner.alg, 'repair_protocol', '') == 'm20_descent_low_mu_58098_v2':
                    from descent_traction_selection import DescentTractionBehaviorSelector
                    selector_class = DescentTractionBehaviorSelector
                    selection_kwargs['turn_reference_results'] = os.environ['M20_FRONT_TURN_RESULTS']
        selector = selector_class(output, runner.alg.behavior_teacher_checkpoint,
                                  runner.alg.behavior_reference_results,
                                  previous_best=runner.alg.behavior_selection_state, **selection_kwargs)
        runner.alg.behavior_selection_state = selector.export_state()
    source_iteration = int(runner.current_learning_iteration)
    initial_iteration = source_iteration + 1
    status = dict(status='starting', task=task, pid=os.getpid(), started=timestamp(), updated=timestamp(),
                  total_updates=total_updates, completed_updates=0, segment_updates=segment_updates,
                  evaluation_every_updates=evaluation_every_updates,
                  evaluation_profile=evaluation_profile,
                  initial_iteration=initial_iteration, source_iteration=source_iteration, log_dir=str(log_dir),
                  baseline_results=str(baseline_path),
                  baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  last_retained_checkpoint=None, last_checkpoint=None, repair_passed=False,
                  behavior_failure_policy='record_and_continue',
                  retention_failures_seen=False, retention_history=[])
    write_json(output / 'status.json', status)
    # Built-in interval saves are redundant with our segment-final save.
    runner.cfg['save_interval'] = 10**12
    completed = 0
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.monotonic()
    try:
        while completed < total_updates:
            resources = resource_check(output, limits)
            count = min(segment_updates, total_updates - completed)
            runner.current_learning_iteration = initial_iteration + completed
            status.update(status='training', updated=timestamp(), segment_start=completed,
                          segment_target=completed + count, resources=resources)
            write_json(output / 'status.json', status)
            print(f'[RECOVERY SESSION] additional updates {completed}/{total_updates}; segment {count}', flush=True)
            runner.learn(num_learning_iterations=count, init_at_random_ep_len=(completed == 0))
            writer = getattr(runner.logger, 'writer', None)
            if writer is not None:
                writer.flush()
                writer.close()
                runner.logger.writer = None
            expected_iteration = initial_iteration + completed + count - 1
            if runner.current_learning_iteration != expected_iteration:
                raise RuntimeError('Unexpected RSL iteration accounting at segment boundary')
            checkpoint = log_dir / f'model_{expected_iteration}.pt'
            if not checkpoint.is_file():
                raise RuntimeError('Segment did not save its final checkpoint')
            completed += count
            evaluate = completed == total_updates or completed % evaluation_every_updates == 0
            if not evaluate:
                status.update(status='training', completed_updates=completed,
                              last_checkpoint=str(checkpoint), updated=timestamp(),
                              elapsed_seconds=time.monotonic() - started)
                write_json(output / 'status.json', status)
                if task in (ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
                    # Only this new run's unevaluated intermediate files are
                    # pruned. Evaluated checkpoints and best archives survive.
                    for saved in log_dir.glob('model_*.pt'):
                        iteration = int(saved.stem.split('_')[-1])
                        additional = iteration-initial_iteration+1
                        if (0 < additional < completed-segment_updates
                                and additional % evaluation_every_updates):
                            saved.unlink()
                print(f'[RECOVERY SESSION] acceptance deferred at {completed}/{total_updates}', flush=True)
                continue
            evaluation = output / f'eval_{completed:06d}'
            if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != status['baseline_sha256']:
                raise RuntimeError('Original-model baseline changed during training')
            status.update(status='evaluating', completed_updates=completed, updated=timestamp(),
                          candidate_checkpoint=str(checkpoint), evaluation_dir=str(evaluation),
                          last_checkpoint=str(checkpoint),
                          elapsed_seconds=time.monotonic() - started)
            write_json(output / 'status.json', status)
            command = [sys.executable, str(ROOT / 'scripts/tools/m20_repair_acceptance.py'),
                       '--checkpoint', str(checkpoint), '--output', str(evaluation),
                       '--baseline-results', str(baseline_path), '--gate', 'retention']
            # Keep periodic full evidence; every interval retains metrics and sequences.
            if completed % 250 or task in (ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
                command.append('--no-trajectories')
            with (output / f'eval_{completed:06d}.log').open('w') as stream:
                result = run_evaluation(command, env=cpu_env, stdout=stream)
            report_path = evaluation / 'acceptance.json'
            if not report_path.is_file():
                raise RuntimeError(f'Evaluation produced no report (exit {result.returncode})')
            report = json.loads(report_path.read_text())
            write_json(output / 'latest_acceptance.json', report)
            if report.get('validation_failures') or report.get('valid_baseline') is not True:
                raise RuntimeError(f'Behavior evaluation is invalid; inspect {report_path}')
            if report.get('retention_passed') is True and result.returncode != 0:
                raise RuntimeError(f'Evaluation exit {result.returncode} disagrees with its passing report: {report_path}')
            summary = evaluation_summary(report, baseline_path)
            summary.update(completed_updates=completed, checkpoint=str(checkpoint), report=str(report_path),
                           retention_passed=report.get('retention_passed'), repair_passed=report.get('repair_passed'))
            if ((evaluation_profile == 'full' or early_58098) and task in
                    (REFINE_TASK, TARGETED_TASK, TARGETED_V2_TASK, STEP_SUPPORT_TASK,
                     STEP_RETENTION_TASK, TRACTION_TRANSITION_TASK, ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK)):
                supplement = evaluation/'quiet_descent'
                command = [sys.executable, str(ROOT/'scripts/tools/m20_quiet_supplement.py'),
                           '--checkpoint', str(checkpoint), '--output', str(supplement)]
                if early_58098:
                    command.append('--retention-metrics')
                    if getattr(runner.alg, 'repair_protocol', '') == 'm20_descent_low_mu_58098_v2':
                        command.append('--retention-v2')
                with (evaluation/'quiet_descent.log').open('w') as stream:
                    extra = run_evaluation(command, env=cpu_env, stdout=stream)
                if extra.returncode != 0:
                    raise RuntimeError('Supplementary evaluator runtime error; inspect quiet_descent.log')
                summary['supplementary'] = json.loads((supplement/'results.json').read_text())
            if selector is not None:
                selection_kwargs = {}
                if early_58098:
                    selection_kwargs['descent_results'] = supplement/'results.json'
                    if getattr(runner.alg, 'repair_protocol', '') == 'm20_descent_low_mu_58098_v2':
                        target = evaluation/'front_turn'
                        command = [sys.executable, str(ROOT/'scripts/tools/m20_front_turn_benchmark.py'),
                                   '--checkpoint', str(checkpoint), '--output', str(target)]
                        with (evaluation/'front_turn.log').open('w') as stream:
                            extra = run_evaluation(command, env=cpu_env, stdout=stream)
                        if extra.returncode != 0:
                            raise RuntimeError('Turn evaluator runtime error; inspect front_turn.log')
                        selection_kwargs['turn_results'] = target/'results.json'
                        summary['front_turn_cases'] = json.loads((target/'results.json').read_text())
                if task in (ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
                    target = evaluation/'entry'
                    command = [sys.executable, str(ROOT/'scripts/tools/m20_entry_flat_benchmark.py'),
                        '--checkpoint', str(checkpoint), '--startup', 'z', '--keyboard-profile',
                        '--keyboard-idle-seconds', '0', '--speeds', '.7', '--output', str(target),
                        '--no-trajectories']
                    with (evaluation/'entry.log').open('w') as stream:
                        extra = run_evaluation(command, env=cpu_env, stdout=stream)
                    if extra.returncode != 0:
                        raise RuntimeError('Entry evaluator runtime error; inspect entry.log')
                    selection_kwargs['entry_results'] = target/'results.json'
                if ((task in (TRACTION_TRANSITION_TASK, ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK)
                     and evaluation_profile == 'full') or task == ROUTE_HOLD_TASK):
                    traction = evaluation/'traction'
                    command = [sys.executable, str(ROOT/'scripts/tools/m20_traction_benchmark.py'),
                               '--checkpoint', str(checkpoint), '--output', str(traction)]
                    with (evaluation/'traction.log').open('w') as stream:
                        extra = run_evaluation(command, env=cpu_env, stdout=stream)
                    if extra.returncode != 0:
                        raise RuntimeError('Traction evaluator runtime error; inspect traction.log')
                    selection_kwargs['traction_results'] = traction/'results.json'
                    summary['traction_results'] = str(traction/'results.json')
                    summary['traction_cases'] = json.loads((traction/'results.json').read_text())
                if flat_repair:
                    target = evaluation/'flat_start'
                    command = [sys.executable, str(ROOT/'scripts/tools/m20_flat_command_probe.py'),
                        '--checkpoint', str(checkpoint), '--output', str(target),
                        '--benchmark', '--no-trajectories']
                    with (evaluation/'flat_start.log').open('w') as stream:
                        extra = run_evaluation(command, env=cpu_env, stdout=stream)
                    if extra.returncode != 0:
                        raise RuntimeError('Flat-start evaluator runtime error; inspect flat_start.log')
                    selection_kwargs['flat_results'] = target/'results.json'
                    summary['flat_start_cases'] = json.loads((target/'results.json').read_text())
                    target = evaluation/'production'
                    command = [sys.executable, str(ROOT/'scripts/tools/m20_production_transition_benchmark.py'),
                        '--checkpoint', str(checkpoint),
                        '--output', str(target), '--no-trajectories']
                    with (evaluation/'production.log').open('w') as stream:
                        extra = run_evaluation(command, env=cpu_env, stdout=stream)
                    if extra.returncode != 0:
                        raise RuntimeError('Production-command evaluator runtime error; inspect production.log')
                    selection_kwargs['forward_results'] = target/'results.json'
                    summary['production_cases'] = json.loads((target/'results.json').read_text())
                selection = selector.consider(checkpoint, report['candidates'][0]['results'], **selection_kwargs)
                if early_58098:
                    summary['descent_retained'] = selection['descent_retained']
                if task == ROUTE_HOLD_TASK:
                    status['route_hold_passed'] = summary['route_hold_passed'] = selection['route_hold_passed']
                if 'flat_start_passed' in selection:
                    status['flat_start_passed'] = summary['flat_start_passed'] = selection['flat_start_passed']
                    status['production_passed'] = summary['production_passed'] = selection['production_passed']
                if task in (ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
                    status['entry_passed'] = selection['entry_passed']
                    summary['entry_passed'] = selection['entry_passed']
                feedback = runner.alg.behavior_feedback(selection.get('feedback_failures', selection['regressions']))
                runner.alg.behavior_selection_state = selector.export_state()
                # Feedback follows evaluation, so it is absent from the
                # evaluated candidate. Save an atomic continuation checkpoint
                # without changing that candidate's measured file hash.
                continuation = log_dir/'resume_after_eval.pt'
                temporary = log_dir/'resume_after_eval.pt.tmp'
                runner.save(str(temporary))
                temporary.replace(continuation)
                summary.update(behavior_selection=selection, behavior_feedback=feedback)
                if task in (TRACTION_TRANSITION_TASK, ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
                    status.update(traction_passed=selection['traction_passed'],
                                  traction_repair_passed=selection['traction_repair_passed'])
                    summary.update(traction_passed=selection['traction_passed'],
                                   traction_repair_passed=selection['traction_repair_passed'],
                                   native_repair_passed=bool(report.get('repair_passed')))
                    summary['repair_passed'] = bool(report.get('repair_passed')) and (
                        evaluation_profile != 'full' or selection['traction_repair_passed'] is True)
                    if selection['traction_passed'] is False:
                        status['retention_failures_seen'] = True
                        status['last_traction_failures'] = selection['regressions']
                    else:
                        status['last_traction_failures'] = []
                status.update(best_candidate_checkpoint=selection['best_checkpoint'],
                              best_candidate_manifest=str(output/'best_candidate.json'),
                              last_resume_checkpoint=str(continuation),
                              behavior_feedback=feedback)
                write_json(evaluation/'behavior_selection.json', dict(selection=selection, feedback=feedback))
            stage_repaired = bool(report.get('repair_passed')) and (
                evaluation_profile != 'full' or task not in (TRACTION_TRANSITION_TASK, ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK)
                or status.get('traction_repair_passed') is True)
            if task in (ENTRY_TRANSFER_TASK, ENTRY_SEQUENCE_TASK, FLAT_POSTURE_TASK, ROUTE_HOLD_TASK):
                stage_repaired &= status.get('entry_passed') is True
            if flat_repair:
                stage_repaired &= status.get('flat_start_passed') is True and status.get('production_passed') is True
            if task == FLAT_POSTURE_TASK:
                # This phase targets flat posture; strict stair sequencing is
                # still reported separately, not relabeled as repaired.
                stage_repaired = (status.get('flat_start_passed') is True
                    and status.get('production_passed') is True
                    and not selection['regressions'])
            if task == ROUTE_HOLD_TASK:
                stage_repaired = status.get('route_hold_passed') is True
            # Persist the complete stage verdict in every output, including
            # TensorBoard; the native suite alone cannot certify deployment.
            summary['repair_passed'] = stage_repaired
            write_evaluation_tensorboard(summary, log_dir)
            write_json(output/'latest_metrics.json', summary)
            with (output/'metrics_history.jsonl').open('a') as stream:
                stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + '\n')
            if report.get('retention_passed') is not True:
                failures = report.get('retention_failures')
                if result.returncode != 2 or not failures:
                    raise RuntimeError(f'Behavior evaluation failed without a valid rejection; inspect {report_path}')
                capability_failures = summary['capability_regressions']
                status['retention_failures_seen'] = True
                status['retention_history'].append(dict(
                    completed_updates=completed, checkpoint=str(checkpoint),
                    report=str(report_path), failures=failures, capability_regressions=capability_failures))
                status.update(last_retention_failures=failures, capability_regressions=capability_failures,
                              updated=timestamp())
                status['status'] = 'training_after_retention_failure'
                write_json(output / 'status.json', status)
                if capability_failures:
                    print('[RECOVERY CAPABILITY REGRESSION] continuing PPO as requested; '
                          'candidate is not accepted:\n'
                          + '\n'.join(f'  - {failure}' for failure in capability_failures), flush=True)
                print('[RECOVERY WARNING] retention failed; continuing PPO with recorded failures:\n'
                      + '\n'.join(f'  - {failure}' for failure in failures), flush=True)
            else:
                status.update(last_retained_checkpoint=str(checkpoint),
                              last_retention_failures=[], capability_regressions=[], updated=timestamp())
            if stage_repaired:
                status['last_repair_passed_checkpoint'] = str(checkpoint)
            status['repair_passed'] = stage_repaired
            status['retention_passed'] = bool(report.get('retention_passed'))
            status['last_checkpoint'] = str(checkpoint)
            write_json(output / 'status.json', status)
            print(f'[RECOVERY SEGMENT COMPLETE] {completed}/{total_updates}; '
                  f'retention_passed={report.get("retention_passed")}; repair_passed={status["repair_passed"]}', flush=True)
        status.update(status=('completed_with_retention_failures' if status['retention_failures_seen'] else 'completed'),
                      updated=timestamp(), elapsed_seconds=time.monotonic() - started)
        write_json(output / 'status.json', status)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            status.update(status='stopped_by_user', stop_reason=str(error) or 'KeyboardInterrupt',
                          signal=getattr(error, 'signum', signal.SIGINT), updated=timestamp())
            write_json(output / 'status.json', status)
        elif not str(status.get('status', '')).startswith('stopped_'):
            status.update(status='stopped_error', error=f'{type(error).__name__}: {error}', updated=timestamp())
            write_json(output / 'status.json', status)
        raise
