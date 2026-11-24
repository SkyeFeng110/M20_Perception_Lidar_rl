"""Measured acceleration safeguards for the entry-first model archive."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from entry_selection import EntryBehaviorSelector, entry_rank, extra_results, diagnostics, gait_regressions
from behavior_selection import regressions as native_regressions

EXPECTED = {prefix+name for prefix in ('raw_', 'filtered_') for name in ('light', 'full', 'partial_full')}


def validated_results(path, checkpoint):
    path = Path(path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    rows = json.loads(path.read_text())
    if (manifest.get('protocol') != 'flat_start_v1' or not manifest.get('physics_enabled')
            or manifest.get('policy_updated') is not False or not manifest.get('deployment_idle_stop')
            or manifest.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            or manifest.get('forward_acceleration_m_s2') != 1.5
            or manifest.get('forward_deceleration_m_s2') != 10./3.
            or manifest.get('duration') != 6. or manifest.get('physics_dt') != .001):
        raise ValueError('Flat-start evidence identity/protocol mismatch')
    if len(rows) != 6 or {r['case']['name'] for r in rows} != EXPECTED:
        raise ValueError('Flat-start evidence requires the fixed six cases')
    for r in rows:
        for key in ('pitch_peak_deg', 'pitch_rate_peak_deg_s', 'fr_inward_y_change_m',
                    'front_width_min_m', 'final_speed_m_s', 'base_min_height_m',
                    'yaw_peak_deg', 'lateral_peak_m',
                    'hipx_target_jump_peak_deg', 'sagittal_target_jump_peak_deg',
                    'wheel_target_jump_peak_rad_s'):
            if not isinstance(r.get(key), (float, int)) or not math.isfinite(r[key]):
                raise ValueError('Missing/nonfinite flat-start metric: '+key)
        if len(r.get('front_hipx_angle_peak_deg', [])) != 2 or not all(math.isfinite(v) for v in r['front_hipx_angle_peak_deg']):
            raise ValueError('Missing flat-start hipx measurement')
    return rows


def passed(rows):
    return all(r['pitch_peak_deg'] <= 3. and r['pitch_rate_peak_deg_s'] <= 20.
               and max(r['front_hipx_angle_peak_deg']) <= 10.
               and r['fr_inward_y_change_m'] <= .04 and r['front_width_min_m'] >= .36
               and r['yaw_peak_deg'] <= 3. and r['lateral_peak_m'] <= .15
               and r['base_min_height_m'] >= .40
               and r['final_speed_m_s'] >= .8*r['case']['speed'] for r in rows)


def regressions(rows, reference):
    before = {r['case']['name']: r for r in reference}
    failures = []
    tolerances = dict(pitch_peak_deg=.5, pitch_rate_peak_deg_s=5., fr_inward_y_change_m=.02,
                      yaw_peak_deg=.5, lateral_peak_m=.03,
                      hipx_target_jump_peak_deg=2., sagittal_target_jump_peak_deg=5.,
                      wheel_target_jump_peak_rad_s=1.)
    for r in rows:
        b, name = before[r['case']['name']], r['case']['name']
        for key, tolerance in tolerances.items():
            if r[key] > b[key]*1.10+tolerance:
                failures.append('flat/'+name+': '+key)
        if max(r['front_hipx_angle_peak_deg']) > max(b['front_hipx_angle_peak_deg'])*1.1+2.:
            failures.append('flat/'+name+': hipx angle')
        if r['front_width_min_m'] < b['front_width_min_m']-.025:
            failures.append('flat/'+name+': support width')
        if r['final_speed_m_s'] < .85*b['final_speed_m_s'] or r['base_min_height_m'] < .40:
            failures.append('flat/'+name+': speed or collapse')
    return failures


def score(rows):
    return -sum(r['pitch_peak_deg']/3.+r['pitch_rate_peak_deg_s']/20.
        +r['yaw_peak_deg']/3.+r['lateral_peak_m']/.15
        +max(r['front_hipx_angle_peak_deg'])/10.+max(0., r['fr_inward_y_change_m'])/.04
        for r in rows)


def validated_forward(path, checkpoint):
    path = Path(path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    rows = json.loads(path.read_text())
    if (manifest.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            or not manifest.get('physics_enabled') or manifest.get('protocol') != 'production_transition_v1'
            or manifest.get('candidate_forward_acceleration_m_s2') != 1.5
            or not manifest.get('zero_command_wheel_stop')
            or manifest.get('startup') != 'Z->C->immediate drive'):
        raise ValueError('Production speed/acceleration evidence is missing or incompatible')
    from behavior_selection import identity
    expected = {(profile+'_z_c_8', terrain, tread, speed)
        for profile, speeds in (('current', [1.]), ('flat-start', [.7, 1.]))
        for terrain, tread in (('flat', None), ('ascent', .26), ('ascent', .30))
        for speed in speeds}
    if len(rows) != 9 or {identity(r) for r in rows} != expected:
        raise ValueError('Production suite must include current raw full 1m/s and the candidate controller on flat and both stairs')
    sources = manifest.get('sources', [])
    if len(sources) != 2 or {s.get('profile') for s in sources} != {'current', 'flat-start'}:
        raise ValueError('Missing measured production subprotocols')
    for source in sources:
        m = source.get('manifest', {})
        direct = source['profile'] == 'current'
        if (m.get('checkpoint_sha256') != manifest['checkpoint_sha256']
                or not m.get('physics_enabled') or not m.get('production_profile')
                or m.get('production_direct') != direct or m.get('keyboard_idle_seconds') != 0
                or m.get('forward_acceleration') != 1.5
                or m.get('speeds') != ([1.] if direct else [.7, 1.])):
            raise ValueError('Production subprotocol speed/filter mismatch')
    diagnostics(rows)
    return rows


def forward_passed(rows):
    return all(not r['fell'] and not r['stall_detected'] and not r['observed_bypass']
        and (r['terrain'] == 'flat' or (r['all_treads_supported'] and r['whole_body_top_supported']
             and r['entry_transfer']['safe_entry'])) for r in rows)


class FlatEntryBehaviorSelector(EntryBehaviorSelector):
    def __init__(self, *args, flat_reference_results, flat_reference_checkpoint, forward_reference_results, previous_best=None, **kwargs):
        self.flat_seed = validated_results(flat_reference_results, flat_reference_checkpoint)
        self.flat_rows, self.flat_path = self.flat_seed, str(Path(flat_reference_results).resolve())
        self.forward_seed = validated_forward(forward_reference_results, flat_reference_checkpoint)
        self.forward_rows, self.forward_path = self.forward_seed, str(Path(forward_reference_results).resolve())
        if previous_best and previous_best.get('flat_results'):
            self.flat_rows = validated_results(previous_best['flat_results'], previous_best['checkpoint'])
            if regressions(self.flat_rows, self.flat_seed):
                raise ValueError('Restored best violates flat-start seed protection')
            self.flat_path = previous_best['flat_results']
            self.forward_rows = validated_forward(previous_best['forward_results'], previous_best['checkpoint'])
            self.forward_path = previous_best['forward_results']
            if native_regressions(self.forward_rows, self.forward_seed)+gait_regressions(self.forward_rows, self.forward_seed):
                raise ValueError('Restored best violates deployed-command stair protection')
        super().__init__(*args, previous_best=previous_best, **kwargs)

    def _archive(self, *args, **kwargs):
        super()._archive(*args, **kwargs)
        target = Path(self.best_checkpoint).parent/'flat_start'
        target.mkdir(exist_ok=True)
        for name in ('results.json', 'manifest.json'):
            src = Path(self.flat_path).parent/name
            dst = target/name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        self.flat_path = str(target/'results.json')
        forward_target = target.parent/'production'
        forward_target.mkdir(exist_ok=True)
        for name in ('results.json', 'manifest.json'):
            src, dst = Path(self.forward_path).parent/name, forward_target/name
            if src.resolve() != dst.resolve(): shutil.copy2(src, dst)
        self.forward_path = str(forward_target/'results.json')
        path = self.output/'best_candidate.json'
        record = json.loads(path.read_text())
        record.update(flat_start_passed=passed(self.flat_rows), flat_start_score=score(self.flat_rows),
                      flat_start_results=self.flat_path, production_results=self.forward_path,
                      production_passed=forward_passed(self.forward_rows))
        record['fully_accepted'] &= record['flat_start_passed'] and record['production_passed']
        path.write_text(json.dumps(record, indent=2)+'\n')

    def export_state(self):
        return dict(super().export_state(), flat_results=self.flat_path, forward_results=self.forward_path)

    def consider(self, checkpoint, results, entry_results, flat_results, forward_results, traction_results=None):
        rows = validated_results(flat_results, checkpoint)
        failures = sorted(set(regressions(rows, self.flat_seed)+regressions(rows, self.flat_rows)))
        forward = validated_forward(forward_results, checkpoint)
        for reference in (self.forward_seed, self.forward_rows):
            failures += native_regressions(forward, reference)+gait_regressions(forward, reference)
        previous_rows, previous_path = self.flat_rows, self.flat_path
        previous_forward, previous_forward_path = self.forward_rows, self.forward_path
        old_entry_rank = entry_rank(self.rows+self.entry_rows)
        old_production_rank = entry_rank(self.forward_rows)
        if failures:
            # Still validate the native/entry evidence even when flat failed.
            native = self._validated_results(checkpoint, results)
            entry = extra_results(entry_results, checkpoint)
            for candidate, reference in ((native, self.seed_rows), (native, self.rows),
                                          (entry, self.entry_seed), (entry, self.entry_rows)):
                failures += native_regressions(candidate, reference)+gait_regressions(candidate, reference)
            return dict(promoted=False, regressions=failures, candidate_rank=old_entry_rank,
                best_rank=old_entry_rank, best_checkpoint=self.best_checkpoint,
                entry_passed=all(r['entry_transfer']['safe_entry'] for r in diagnostics(native+entry)),
                traction_passed=None, traction_repair_passed=None,
                flat_start_passed=passed(rows), flat_start_score=score(rows),
                production_passed=forward_passed(forward))
        self.flat_rows, self.flat_path = rows, str(flat_results)
        self.forward_rows, self.forward_path = forward, str(forward_results)
        outcome = super().consider(checkpoint, results, entry_results, traction_results)
        if (not outcome['promoted'] and not outcome['regressions']
                and tuple(outcome['candidate_rank']) >= tuple(old_entry_rank)
                and (score(rows) > score(previous_rows)+.01 or entry_rank(forward) > old_production_rank)):
            self.rows = self._validated_results(checkpoint, results)
            self.entry_rows = extra_results(entry_results, checkpoint)
            self.entry_path = str(entry_results)
            self._archive(checkpoint, results)
            outcome.update(promoted=True, best_checkpoint=self.best_checkpoint)
        if not outcome['promoted']:
            self.flat_rows, self.flat_path = previous_rows, previous_path
            self.forward_rows, self.forward_path = previous_forward, previous_forward_path
        if self.traction_seed is None:
            outcome.update(traction_passed=None, traction_repair_passed=None)
        return dict(outcome, flat_start_passed=passed(rows), flat_start_score=score(rows),
                    production_passed=forward_passed(forward))
