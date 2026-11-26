"""58098 retention, including descent, gates every early-traction promotion."""
import hashlib
import json
import math
import statistics
from pathlib import Path
import shutil
from behavior_selection import identity
from route_hold_selection import RouteHoldBehaviorSelector

PROTOCOL = 'm20_early_handoff_58098_v1'


def validated_descent(path, checkpoint):
    path = Path(path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (manifest.get('protocol') != PROTOCOL or not manifest.get('physics_enabled')
            or manifest.get('checkpoint_sha256') != sha(checkpoint)
            or manifest.get('results_sha256') != sha(path)
            or manifest.get('height') != .16 or manifest.get('duration') != 25.):
        raise ValueError('Invalid descent retention protocol or checkpoint')
    rows = json.loads(path.read_text())
    if len(rows) != 4 or {(r['tread'], r['command_speed']) for r in rows} != {(t, v) for t in (.26, .30) for v in (.5, .7)}:
        raise ValueError('Descent retention needs all four measured cases')
    for row in rows:
        if not isinstance(row.get('goal_reached'), bool) or not isinstance(row.get('fell'), bool):
            raise ValueError('Invalid descent physical result')
        pose = row.get('retention_pose', {})
        for key in ('pitch_peak_deg', 'roll_peak_deg', 'reach_p95_m', 'mean_speed_m_s',
                    'yaw_peak_deg', 'lateral_peak_m', 'moving_seconds'):
            if not isinstance(pose.get(key), (int, float)) or not math.isfinite(pose[key]):
                raise ValueError('Invalid descent pose: '+key)
    return rows


def descent_regressions(rows, reference):
    before = {(r['tread'], r['command_speed']): r for r in reference}
    failures = []
    for row in rows:
        old = before[row['tread'], row['command_speed']]
        reasons = []
        for key in ('fell', 'observed_bypass'):
            if row.get(key) and not old.get(key):
                reasons.append(key)
        if old['goal_reached'] and not row['goal_reached']:
            reasons.append('lost goal_reached')
        a, b = row['retention_pose'], old['retention_pose']
        for key, margin in (('pitch_peak_deg', 3.), ('roll_peak_deg', 3.), ('reach_p95_m', .02),
                            ('yaw_peak_deg', 1.), ('lateral_peak_m', .05)):
            if a[key] > b[key]+margin:
                reasons.append(key)
        if a['mean_speed_m_s'] < .9*min(b['mean_speed_m_s'], row['command_speed']):
            reasons.append('route speed')
        if reasons:
            failures.append(row['case']+': '+', '.join(reasons))
    return failures


class EarlyTractionBehaviorSelector(RouteHoldBehaviorSelector):
    @staticmethod
    def _early_slip_score(rows):
        selected = [r['early_ascent_slip_p95'] for r in rows if r['terrain'] == 'ascent' and (
            (r['tread'] == .26 and r['friction'] == 1. and r['speed'] >= .5)
            or (r['friction'] == .6 and r['speed'] == 1.))]
        if len(selected) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in selected):
            raise ValueError('Need the 26cm nominal-friction and both low-friction early-traction cases')
        return sum(selected)/5.

    def _traction_repaired(self, rows):
        nominal = lambda rs: [r['early_ascent_slip_p95'] for r in rs if r['terrain'] == 'ascent'
            and r['tread'] == .26 and r['friction'] == 1. and r['speed'] >= .5]
        return (super()._traction_repaired(rows)
                and statistics.median(nominal(rows)) <= .85*statistics.median(nominal(self.traction_seed)))

    def _traction_regressions(self, rows):
        from traction_selection import identity as traction_identity
        failures = super()._traction_regressions(rows)
        for reference in (self.traction_seed, self.traction_rows):
            before = {traction_identity(r): r for r in reference}
            for row in rows:
                if row['rear_support_fraction'] < .95*before[traction_identity(row)]['rear_support_fraction']:
                    failures.append('traction/'+str(traction_identity(row))+': rear support reduced')
        return failures

    def __init__(self, *args, descent_results, **kwargs):
        self._descent_ready = False
        self._pending_descent = None
        self.descent_seed_path = str(Path(descent_results).resolve())
        self.descent_seed = self.descent_rows = validated_descent(descent_results, args[1])
        self.descent_path = self.descent_seed_path
        super().__init__(*args, **kwargs)
        previous = kwargs.get('previous_best')
        if previous and previous.get('descent_results'):
            self.descent_path = previous['descent_results']
            self.descent_rows = validated_descent(self.descent_path, self.best_checkpoint)
            if descent_regressions(self.descent_rows, self.descent_seed):
                raise ValueError('Restored candidate regresses the descent reference')
        self._descent_ready = True
        self._archive(self.best_checkpoint, self.best_results,
                      seed=hashlib.sha256(Path(self.best_checkpoint).read_bytes()).hexdigest() == self.seed_sha256)

    def _regressions(self, native, entry, flat, forward):
        failures = super()._regressions(native, entry, flat, forward)
        # Stricter relative flat gates prevent buying traction by slowing the
        # robot or undoing its acquired full-stick pitch/reach repair.
        for reference in (self.flat_seed, self.flat_rows):
            before = {r['case']['name']: r for r in reference}
            for row in flat:
                old = before[row['case']['name']]
                if row['final_speed_m_s'] < .95*min(old['final_speed_m_s'], row['case']['speed']):
                    failures.append('flat/'+row['case']['name']+': speed traded for traction')
                if row['pitch_peak_deg'] > old['pitch_peak_deg']+.5:
                    failures.append('flat/'+row['case']['name']+': pitch_peak_deg')
        for rows, reference in ((native, self.seed_rows), (forward, self.forward_seed)):
            before = {identity(r): r for r in reference}
            for row in rows:
                old = before[identity(row)]
                if row['terrain'] == 'flat':
                    if (row.get('route_mean_velocity_x') or 0.) < .95*min(old.get('route_mean_velocity_x') or 0., row['command_speed']):
                        failures.append('flat/'+str(identity(row))+': speed traded for traction')
                a, b = row.get('retention_pose', {}), old.get('retention_pose', {})
                for key, margin in (('pitch_peak_deg', .5 if row['terrain'] == 'flat' else 2.),
                                    ('roll_peak_deg', 2.), ('reach_p95_m', .02)):
                    if isinstance(b.get(key), (int, float)) and (
                            not isinstance(a.get(key), (int, float)) or not math.isfinite(a[key])
                            or a[key] > b[key]+margin):
                        failures.append(row['terrain']+'/'+str(identity(row))+': retained '+key)
                if row['terrain'] == 'ascent':
                    for key in ('skipped_treads', 'rear_skipped_treads'):
                        for side, old_skips in enumerate(old.get(key) or []):
                            new_skips = (row.get(key) or [[], []])[side]
                            if set(new_skips)-set(old_skips):
                                failures.append('ascent/'+str(identity(row))+': new '+key)
        if self._descent_ready:
            rows = self._pending_descent[0] if self._pending_descent else self.descent_rows
            failures += descent_regressions(rows, self.descent_seed)+descent_regressions(rows, self.descent_rows)
        return sorted(set(failures))

    def _archive(self, *args, **kwargs):
        if self._descent_ready and self._pending_descent:
            self.descent_rows, self.descent_path = self._pending_descent
        super()._archive(*args, **kwargs)
        if not self._descent_ready:
            return
        target = Path(self.best_checkpoint).parent/'descent'
        target.mkdir(exist_ok=True)
        for name in ('results.json', 'manifest.json'):
            source = Path(self.descent_path).parent/name
            if source.resolve() != (target/name).resolve():
                shutil.copy2(source, target/name)
        self.descent_path = str(target/'results.json')
        path = self.output/'best_candidate.json'
        record = json.loads(path.read_text())
        record.update(selection_focus='B1-B3 traction; measured 58098 flat/ascent/descent retained',
                      descent_results=self.descent_path, descent_reference_results=self.descent_seed_path,
                      descent_retained=not descent_regressions(self.descent_rows, self.descent_seed))
        path.write_text(json.dumps(record, indent=2)+'\n')

    def export_state(self):
        return dict(super().export_state(), descent_results=self.descent_path)

    def consider(self, *args, descent_results, **kwargs):
        rows = validated_descent(descent_results, args[0])
        failures = descent_regressions(rows, self.descent_seed)+descent_regressions(rows, self.descent_rows)
        self._pending_descent = (rows, str(Path(descent_results).resolve()))
        try:
            result = super().consider(*args, **kwargs)
        finally:
            self._pending_descent = None
        result['descent_retained'] = not failures
        return result
