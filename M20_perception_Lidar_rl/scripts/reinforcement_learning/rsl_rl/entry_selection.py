"""Entry-first best selection with per-case later four-leg gait protection."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from behavior_selection import BehaviorSelector, identity, regressions, rank
from traction_selection import validated_results, regressions as traction_regressions, repair_improved


def diagnostics(rows):
    stairs = [r for r in rows if r['terrain'] == 'ascent']
    for r in stairs:
        d = r.get('entry_transfer')
        if not isinstance(d, dict) or d.get('protocol') not in ('entry_transfer_v2', 'entry_transfer_v3'):
            raise ValueError('Missing entry contact diagnostics; recollect physical evidence')
        for key in ('blocked_body_advance_m', 'pitch_peak_abs_deg', 'pitch_rate_peak_deg_s',
                    'entry_elapsed_seconds', 'twin_air_seconds'):
            if not isinstance(d.get(key), (int, float)) or not math.isfinite(d[key]):
                raise ValueError('Invalid entry metric: '+key)
    return stairs


def entry_rank(rows):
    stairs = diagnostics(rows)
    return (sum(r['entry_transfer']['safe_entry'] for r in stairs),
        sum(r['entry_transfer']['alternating'] for r in stairs),
        -sum(r['entry_transfer']['rear_lift_before_riser'] for r in stairs),
        -sum(r['entry_transfer']['blocked_body_advance_m']
             +.01*r['entry_transfer']['pitch_peak_abs_deg'] for r in stairs), *rank(rows))


def gait_regressions(rows, reference):
    old = {identity(r): r for r in diagnostics(reference)}
    failures = []
    for r in diagnostics(rows):
        a, b = r['entry_transfer'], old[identity(r)]['entry_transfer']
        reasons = []
        for key in ('safe_entry', 'alternating'):
            if b[key] and not a[key]: reasons.append('lost '+key)
        if not b['rear_lift_before_riser'] and a['rear_lift_before_riser']:
            reasons.append('new early rear lift')
        # Compare measured ordering on each later tread. Absolute times may
        # shift when entry is corrected; only robust pairwise order matters.
        for tread, before in b['support_times'].items():
            if int(tread) < 2: continue
            after = a['support_times'].get(tread, [None]*4)
            for i in range(4):
                if before[i] is not None and after[i] is None:
                    reasons.append('lost wheel support B'+tread); continue
                for j in range(i+1, 4):
                    if any(v is None for v in (before[i], before[j], after[i], after[j])): continue
                    if abs(before[i]-before[j]) <= .06: continue
                    if (after[j]-after[i]) * (1 if before[j] > before[i] else -1) < -.04:
                        reasons.append('four-leg order B'+tread)
        if reasons: failures.append('entry/'+str(identity(r))+': '+', '.join(sorted(set(reasons))))
    return failures


def extra_results(path, checkpoint):
    path = Path(path)
    rows = json.loads(path.read_text())
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if (manifest.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            or not manifest.get('physics_enabled') or not manifest.get('keyboard_profile')
            or manifest.get('keyboard_idle_seconds') != 0):
        raise ValueError('Entry supplement must measure immediate production C-to-8 physics')
    expected = {('z_c_8_immediate', 'flat', None, .7),
                ('z_c_8_immediate', 'ascent', .26, .7), ('z_c_8_immediate', 'ascent', .30, .7)}
    if len(rows) != 3 or {identity(r) for r in rows} != expected:
        raise ValueError('Entry supplement needs its fixed three cases')
    diagnostics(rows)
    return rows


class EntryBehaviorSelector(BehaviorSelector):
    def __init__(self, output, checkpoint, results, entry_results, traction_results=None, previous_best=None):
        if traction_results is None:
            self.traction_seed, self.traction_manifest = None, None
        else:
            self.traction_seed, self.traction_manifest = validated_results(traction_results, checkpoint)
        self.entry_seed = extra_results(entry_results, checkpoint)
        diagnostics(json.loads(Path(results).read_text()))
        self.entry_path = str(Path(entry_results).resolve())
        self.traction_path = (str(Path(traction_results).resolve())
                              if traction_results is not None else None)
        super().__init__(output, checkpoint, results)
        self.traction_rows, self.entry_rows = self.traction_seed, self.entry_seed
        if previous_best:
            if previous_best['reference_sha256'] != self.seed_sha256:
                raise ValueError('Entry best belongs to a different seed')
            cp, evidence = previous_best['checkpoint'], previous_best['results']
            native = self._validated_results(cp, evidence)
            traction = None
            if self.traction_seed is not None:
                traction, _ = validated_results(previous_best['traction_results'], cp)
            entry = extra_results(previous_best['entry_results'], cp)
            failures = (regressions(native, self.seed_rows)+gait_regressions(native, self.seed_rows)
                +regressions(entry, self.entry_seed)+gait_regressions(entry, self.entry_seed)
                +(traction_regressions(traction, self.traction_seed)
                  if traction is not None else []))
            if failures or entry_rank(native+entry) < entry_rank(self.seed_rows+self.entry_seed):
                raise ValueError('Restored entry best violates protected capabilities')
            self.rows, self.traction_rows, self.entry_rows = native, traction, entry
            self.entry_path = previous_best['entry_results']
            self.traction_path = (previous_best.get('traction_results')
                                  if self.traction_seed is not None else None)
            self._archive(cp, evidence)

    def _archive(self, *args, **kwargs):
        super()._archive(*args, **kwargs)
        destination = Path(self.best_checkpoint).parent
        sources = [(self.entry_path, 'entry')]
        if self.traction_path is not None:
            sources.append((self.traction_path, 'traction'))
        for source, folder in sources:
            target = destination/folder; target.mkdir(exist_ok=True)
            for name in ('manifest.json', 'results.json'):
                src, dst = Path(source).parent/name, target/name
                if src.resolve() != dst.resolve(): shutil.copy2(src, dst)
        self.entry_path = str(destination/'entry/results.json')
        if self.traction_path is not None:
            self.traction_path = str(destination/'traction/results.json')
        path = self.output/'best_candidate.json'; record = json.loads(path.read_text())
        supplement = json.loads(Path(self.entry_path).read_text())
        entry_pass = all(r['entry_transfer']['safe_entry'] for r in diagnostics(self.rows+supplement))
        record.update(entry_passed=entry_pass, entry_rank=entry_rank(self.rows+supplement),
                      native_fully_accepted=record['fully_accepted'])
        record['fully_accepted'] &= entry_pass
        path.write_text(json.dumps(record, indent=2)+'\n')

    def export_state(self):
        state = dict(super().export_state(), entry_results=self.entry_path)
        if self.traction_path is not None:
            state['traction_results'] = self.traction_path
        return state

    def consider(self, checkpoint, results, entry_results, traction_results=None):
        native = self._validated_results(checkpoint, results)
        entry = extra_results(entry_results, checkpoint)
        traction, manifest = None, None
        if self.traction_seed is not None:
            if traction_results is None:
                raise ValueError('Entry selector requires traction evidence in full evaluation profile')
            traction, manifest = validated_results(traction_results, checkpoint)
            for key in ('duration', 'period', 'frictions', 'contact_force', 'horizontal_normal_min',
                        'actor_dimensions', 'action_scale', 'kp', 'kd', 'effort_limit', 'armature',
                        'feature_contract', 'slip_method', 'startup'):
                if manifest[key] != self.traction_manifest[key]: raise ValueError('Traction protocol changed: '+key)
        failures = []
        for a, b in ((native, self.seed_rows), (native, self.rows),
                     (entry, self.entry_seed), (entry, self.entry_rows)):
            failures += regressions(a, b)+gait_regressions(a, b)
        traction_fail = []
        if self.traction_seed is not None:
            traction_fail = (traction_regressions(traction, self.traction_seed)
                             +traction_regressions(traction, self.traction_rows))
        failures = sorted(set(failures+traction_fail))
        score, best = entry_rank(native+entry), entry_rank(self.rows+self.entry_rows)
        promoted = not failures and score > best
        if promoted:
            self.rows, self.entry_rows, self.traction_rows = native, entry, traction
            self.entry_path = str(entry_results)
            if traction_results is not None:
                self.traction_path = str(traction_results)
            self._archive(checkpoint, results)
        passed = all(r['entry_transfer']['safe_entry'] for r in diagnostics(native+entry))
        return dict(promoted=promoted, regressions=failures, candidate_rank=score, best_rank=best,
            best_checkpoint=self.best_checkpoint, entry_passed=passed,
            traction_passed=(not traction_fail),
            traction_repair_passed=(repair_improved(traction, self.traction_seed)
                                    if self.traction_seed is not None else True))
