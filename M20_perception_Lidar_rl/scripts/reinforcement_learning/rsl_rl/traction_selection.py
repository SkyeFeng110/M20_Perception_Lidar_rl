"""Additional traction gate, composed with the unchanged native capability gate."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from behavior_selection import BehaviorSelector, rank


def identity(row):
    return tuple(row[k] for k in ('terrain', 'tread', 'profile', 'speed', 'friction'))


def validated_results(path, checkpoint=None):
    path = Path(path)
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if manifest.get('suite') != 'traction_transition_v1' or not manifest.get('physics_enabled'):
        raise ValueError('Missing physical traction evidence')
    expected = {('flat', None, p, 1., f) for p in ('step_release', 'partial_full', 'ramp') for f in (1., .6)}
    expected |= {('ascent', w, 'hold', v, f) for w in (.26, .30) for v in (.3, .5, .7, 1.) for f in (1., .6)}
    rows = json.loads(path.read_text())
    if len(rows) != len(expected) or {identity(r) for r in rows} != expected:
        raise ValueError('Traction evidence must contain the complete 22-case suite')
    if checkpoint and hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
        raise ValueError('Traction evidence checkpoint hash mismatch')
    for row in rows:
        for key in ('pitch_peak_deg', 'pitch_p95_deg', 'pitch_rate_peak_deg_s', 'rear_support_fraction',
                    'progress_m', 'lateral_max_abs_m', 'lateral_signed_final_m', 'yaw_max_abs_deg'):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                raise ValueError('Nonfinite/missing traction metric: '+key)
        for key in ('rear_slip_p95', 'rear_slip_mean', 'steady_tracking_ratio'):
            if row.get(key) is not None and not math.isfinite(row[key]):
                raise ValueError('Nonfinite traction metric: '+key)
        if row['terrain'] == 'ascent' and row.get('early_ascent_slip_p95') is not None:
            for key in ('early_ascent_slip_p95', 'early_ascent_slip_mean'):
                if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                    raise ValueError('Invalid early-tread traction metric: '+key)
            if row.get('early_loaded_contact_samples', 0) < 1:
                raise ValueError('No loaded contact samples on the first three treads')
    return rows, manifest


def regressions(rows, reference):
    baseline = {identity(r): r for r in reference}
    if {identity(r) for r in rows} != set(baseline):
        raise ValueError('Traction case sets differ')
    failures = []
    for row in rows:
        old = baseline[identity(row)]
        reasons = []
        if row['fell'] and not old['fell']:
            reasons.append('fall')
        if row['bypass'] and not old['bypass']:
            reasons.append('bypass')
        if old['termination'] == 'flight_complete' and row['termination'] != 'flight_complete':
            reasons.append('lost whole-body arrival')
        if row.get('entry_front_scope') == 'from_lower' and row.get('entry_front_simultaneous', False) \
                and old.get('entry_front_scope') == 'from_lower' and not old.get('entry_front_simultaneous', False):
            reasons.append('simultaneous first-tread front support')
        if old['steady_tracking_ratio'] is not None and (row['steady_tracking_ratio'] is None
                or row['steady_tracking_ratio'] < .9*min(1., old['steady_tracking_ratio'])):
            reasons.append('route speed')
        # Loaded support must remain measurable; unloading cannot fake low slip.
        if row['rear_support_fraction'] < max(.1, .8*old['rear_support_fraction']):
            reasons.append('rear support')
        for key, absolute, relative in (('rear_slip_p95', .03, .15), ('lateral_max_abs_m', .08, 0.),
                                        ('yaw_max_abs_deg', 3., 0.)):
            a, b = row[key], old[key]
            if b is not None and (a is None or a > b*(1+relative)+absolute):
                reasons.append(key)
        if row['terrain'] == 'ascent' and row.get('early_ascent_slip_p95') is not None \
                and old.get('early_ascent_slip_p95') is not None:
            value, old_value = row['early_ascent_slip_p95'], old['early_ascent_slip_p95']
            if value > old_value*1.15+.03:
                reasons.append('early-tread slip regression')
        if row['terrain'] == 'flat':
            for key, margin in (('pitch_peak_deg', 2.), ('pitch_rate_peak_deg_s', 10.)):
                if row[key] > old[key]+margin:
                    reasons.append(key)
        if reasons:
            failures.append('traction/'+str(identity(row))+': '+', '.join(reasons))
    return failures


def repair_improved(rows, reference):
    """Provisional simulator targets; physical deployment still needs retest."""
    import statistics
    flat = [r for r in rows if r['terrain'] == 'flat']
    early = [r for r in rows if r['terrain'] == 'ascent' and r['speed'] >= .85 and r['friction'] == .6]
    old_early = [r for r in reference if r['terrain'] == 'ascent' and r['speed'] >= .85 and r['friction'] == .6]
    early_available = bool(early) and len(early) == len(old_early) and all(
        r.get('early_ascent_slip_p95') is not None for r in early+old_early)
    if not early_available:
        early = [r for r in rows if r['terrain'] == 'ascent' and r['speed'] < .85]
        old_early = [r for r in reference if r['terrain'] == 'ascent' and r['speed'] < .85]
        if any(r['rear_slip_p95'] is None for r in early+old_early):
            return False
    return (not regressions(rows, reference)
        and all(r['pitch_peak_deg'] <= 8. and r['pitch_rate_peak_deg_s'] <= 45. for r in flat)
        and statistics.median(r['early_ascent_slip_p95'] if early_available else r['rear_slip_p95'] for r in early)
            <= .85*statistics.median(r['early_ascent_slip_p95'] if early_available else r['rear_slip_p95']
                                     for r in old_early))


class TractionBehaviorSelector(BehaviorSelector):
    def __init__(self, output, checkpoint, results, traction_results, previous_best=None):
        self.traction_seed, self.traction_manifest = validated_results(traction_results)
        self.current_traction_repair_passed = False
        super().__init__(output, checkpoint, results, previous_best=previous_best)
        self.traction_rows = self.traction_seed
        self.traction_best = str(Path(traction_results).resolve())
        if previous_best and previous_best.get('traction_results'):
            self.traction_best = previous_best['traction_results']
            self.traction_rows, _ = validated_results(self.traction_best, self.best_checkpoint)

    def export_state(self):
        return dict(super().export_state(), traction_results=self.traction_best)

    def _archive(self, *args, **kwargs):
        super()._archive(*args, **kwargs)
        path = self.output/'best_candidate.json'
        record = json.loads(path.read_text())
        record['native_fully_accepted'] = record['fully_accepted']
        record['traction_repair_passed'] = self.current_traction_repair_passed
        record['fully_accepted'] &= self.current_traction_repair_passed
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(path)

    def consider(self, checkpoint, results, traction_results):
        rows, manifest = validated_results(traction_results, checkpoint)
        for key in ('duration', 'period', 'frictions', 'contact_force', 'horizontal_normal_min',
                    'actor_dimensions', 'action_scale', 'kp', 'kd', 'effort_limit', 'armature',
                    'feature_contract', 'slip_method', 'startup'):
            if manifest[key] != self.traction_manifest[key]:
                raise ValueError('Traction protocol/config changed: '+key)
        failures = sorted(set(regressions(rows, self.traction_seed)+regressions(rows, self.traction_rows)))
        if failures:
            native = self._validated_results(checkpoint, results)
            return dict(promoted=False, regressions=failures, candidate_rank=rank(native),
                        best_rank=rank(self.rows), best_checkpoint=self.best_checkpoint,
                        traction_passed=False, traction_repair_passed=False)
        self.current_traction_repair_passed = repair_improved(rows, self.traction_seed)
        selection = super().consider(checkpoint, results)
        if selection['promoted']:
            target = Path(self.best_checkpoint).parent/'traction'
            target.mkdir(exist_ok=True)
            for name in ('results.json', 'manifest.json'):
                shutil.copy2(Path(traction_results).parent/name, target/name)
            self.traction_rows, self.traction_best = rows, str(target/'results.json')
        selection.update(traction_passed=True, traction_repair_passed=self.current_traction_repair_passed)
        return selection
