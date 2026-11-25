"""Route-first selection with the measured 56898 posture and gait protected."""
import json
import hashlib
import math
from pathlib import Path
import shutil
from behavior_selection import identity
from entry_selection import extra_results
from flat_start_selection import validated_results, validated_forward, score
from flat_posture_selection import FlatPostureBehaviorSelector, flat_passed, production_passed
from traction_selection import (validated_results as validated_traction_results,
    regressions as traction_regressions, repair_improved)


def route_failures(native, entry, flat, forward):
    failures = []
    for row in native+entry+forward:
        label = '/'.join(map(str, identity(row)))
        for field, limit in (('forward_yaw_max_change_deg', 3.),
                             ('forward_lateral_max_abs', .15 if row['terrain'] == 'ascent' else .10)):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
                failures.append(label+': route target '+field)
    for row in flat:
        for field, limit in (('yaw_peak_deg', 3.), ('lateral_peak_m', .10)):
            if row[field] > limit:
                failures.append('flat/'+row['case']['name']+': route target '+field)
    return sorted(set(failures))


def route_score(native, entry, flat, forward):
    # Normalize each suite separately so the many native cases do not hide
    # the immediate raw full-speed command used on the real machine.
    error = 0.
    for rows in (native, entry, forward):
        error += sum((r.get('forward_yaw_max_change_deg') if isinstance(r.get('forward_yaw_max_change_deg'), (int, float)) else 180.)/3.
            +(r.get('forward_lateral_max_abs') if isinstance(r.get('forward_lateral_max_abs'), (int, float)) else 10.)/
            (.15 if r['terrain'] == 'ascent' else .10) for r in rows)/len(rows)
    error += sum(r['yaw_peak_deg']/3.+r['lateral_peak_m']/.10 for r in flat)/len(flat)
    return -error


class RouteHoldBehaviorSelector(FlatPostureBehaviorSelector):
    def __init__(self, output, checkpoint, results, entry_results, flat_reference_results,
                 flat_reference_checkpoint, forward_reference_results, traction_results,
                 previous_best=None):
        traction_seed, traction_manifest = validated_traction_results(traction_results)
        self._route_traction_ready = False
        self._route_traction_seed = traction_seed
        self._route_traction_manifest = traction_manifest
        self._route_traction_seed_path = str(Path(traction_results).resolve())
        self.traction_best_path = None
        self.traction_repair_passed = False
        super().__init__(output, checkpoint, results, entry_results, flat_reference_results,
                         flat_reference_checkpoint, forward_reference_results, previous_best)
        self.traction_seed, self.traction_manifest = traction_seed, traction_manifest
        self.traction_rows = traction_seed
        self.traction_seed_path = self._route_traction_seed_path
        self._route_traction_ready = True
        if previous_best and previous_best.get('traction_results'):
            path = previous_best['traction_results']
            rows, _ = validated_traction_results(path, self.best_checkpoint)
            if traction_regressions(rows, self.traction_seed):
                raise ValueError('Restored route best regresses the measured traction seed')
            self.traction_rows = rows
            self.traction_best_path = str(Path(path).resolve())
            self.traction_repair_passed = repair_improved(rows, self.traction_seed)
        self._archive(self.best_checkpoint, self.best_results,
            seed=hashlib.sha256(Path(self.best_checkpoint).read_bytes()).hexdigest() == self.seed_sha256)

    def export_state(self):
        state = super().export_state()
        if self.traction_best_path is not None:
            state['traction_results'] = self.traction_best_path
        return state

    @staticmethod
    def _early_slip_score(rows):
        selected = [r['early_ascent_slip_p95'] for r in rows
            if r['terrain'] == 'ascent' and r['speed'] >= .85 and r['friction'] == .6]
        if len(selected) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in selected):
            raise ValueError('Route traction ranking requires B1-B3 slip from both 1 m/s low-friction ascent cases')
        return sum(selected)/len(selected)

    def _regressions(self, native, entry, flat, forward):
        failures = super()._regressions(native, entry, flat, forward)
        # Retain gains in each measured case instead of trading a large
        # left drift on one staircase for a better average on other surfaces.
        for rows, seed, best in ((native, self.seed_rows, self.rows),
                                 (entry, self.entry_seed, self.entry_rows),
                                 (forward, self.forward_seed, self.forward_rows)):
            for reference in (seed, best):
                before = {identity(r): r for r in reference}
                for row in rows:
                    old = before[identity(row)]
                    for field, tolerance in (('forward_yaw_max_change_deg', 1.),
                                             ('forward_lateral_max_abs', .05)):
                        a, b = row.get(field), old.get(field)
                        if isinstance(b, (int, float)) and (a is None or a > b+tolerance):
                            failures.append('/'.join(map(str, identity(row)))+': '+field)
        return sorted(set(failures))

    def _archive(self, *args, **kwargs):
        super()._archive(*args, **kwargs)
        path = self.output/'best_candidate.json'
        record = json.loads(path.read_text())
        unresolved = route_failures(self.rows, self.entry_rows, self.flat_rows, self.forward_rows)
        retained = not self._regressions(self.rows, self.entry_rows, self.flat_rows, self.forward_rows)
        if not self._route_traction_ready:
            record.update(selection_focus='original route; 56898 flat posture and stair gait protected',
                route_score=route_score(self.rows, self.entry_rows, self.flat_rows, self.forward_rows),
                route_hold_passed=False, route_unresolved=unresolved,
                traction_reference_results=self._route_traction_seed_path)
            path.write_text(json.dumps(record, indent=2)+'\n')
            return
        traction_ok = not traction_regressions(self.traction_rows, self.traction_seed)
        self.traction_repair_passed = repair_improved(self.traction_rows, self.traction_seed)
        stage_passed = (not unresolved and retained and traction_ok and self.traction_repair_passed
                        and flat_passed(self.flat_rows)
                        and production_passed(self.forward_rows))
        destination = Path(self.best_checkpoint).parent
        if self.traction_best_path is not None:
            target = destination/'traction'
            target.mkdir(exist_ok=True)
            for name in ('results.json', 'manifest.json'):
                source, dest = Path(self.traction_best_path).parent/name, target/name
                if source.resolve() != dest.resolve():
                    shutil.copy2(source, dest)
            self.traction_best_path = str(target/'results.json')
        record.update(selection_focus='original route; 56898 flat posture and stair gait protected',
            route_score=route_score(self.rows, self.entry_rows, self.flat_rows, self.forward_rows),
            route_hold_passed=stage_passed, route_unresolved=unresolved,
            traction_score=self._early_slip_score(self.traction_rows),
            traction_passed=traction_ok, traction_repair_passed=self.traction_repair_passed,
            traction_reference_results=self.traction_seed_path,
            traction_results=self.traction_best_path)
        # Relative improvement never certifies a still-biased seed/candidate.
        record['fully_accepted'] &= stage_passed
        path.write_text(json.dumps(record, indent=2)+'\n')

    def consider(self, checkpoint, results, entry_results, flat_results, forward_results, traction_results=None):
        native = self._validated_results(checkpoint, results)
        entry = extra_results(entry_results, checkpoint)
        flat = validated_results(flat_results, checkpoint)
        forward = validated_forward(forward_results, checkpoint)
        candidate_traction, manifest = validated_traction_results(traction_results, checkpoint)
        for key in ('duration', 'period', 'frictions', 'contact_force', 'horizontal_normal_min',
                    'actor_dimensions', 'action_scale', 'kp', 'kd', 'effort_limit', 'armature',
                    'feature_contract', 'slip_method', 'startup', 'early_tread_window',
                    'early_slip_method'):
            if manifest.get(key) != self.traction_manifest.get(key):
                raise ValueError('Route traction evaluation protocol changed: '+key)
        failures = self._regressions(native, entry, flat, forward)
        traction_failures = (traction_regressions(candidate_traction, self.traction_seed)
                             +traction_regressions(candidate_traction, self.traction_rows))
        failures = sorted(set(failures+traction_failures))
        unresolved = route_failures(native, entry, flat, forward)
        candidate = route_score(native, entry, flat, forward)
        best = route_score(self.rows, self.entry_rows, self.flat_rows, self.forward_rows)
        slip_score = self._early_slip_score(candidate_traction)
        best_slip_score = self._early_slip_score(self.traction_rows)
        traction_repaired = repair_improved(candidate_traction, self.traction_seed)
        promoted = (not failures and (slip_score < best_slip_score-.005
                    or (abs(slip_score-best_slip_score) <= .005 and candidate > best+.01)))
        if promoted:
            self.rows, self.entry_rows, self.flat_rows, self.forward_rows = native, entry, flat, forward
            self.entry_path, self.flat_path, self.forward_path = map(str, (entry_results, flat_results, forward_results))
            self.traction_rows = candidate_traction
            self.traction_best_path = str(Path(traction_results).resolve())
            self._archive(checkpoint, results)
        feedback_failures = failures+unresolved
        if not traction_repaired:
            feedback_failures.append('traction: early-tread slip target unresolved')
        return dict(promoted=promoted, regressions=failures, route_unresolved=unresolved,
            feedback_failures=sorted(set(feedback_failures)),
            candidate_rank=(-slip_score, candidate), best_rank=(-self._early_slip_score(self.traction_rows),
                route_score(self.rows, self.entry_rows, self.flat_rows, self.forward_rows)),
            best_checkpoint=self.best_checkpoint,
            route_hold_passed=(not failures and not unresolved and flat_passed(flat)
                and production_passed(forward) and traction_repaired),
            traction_passed=not traction_failures,
            traction_repair_passed=traction_repaired,
            early_traction_score=slip_score,
            best_early_traction_score=self._early_slip_score(self.traction_rows),
            entry_passed=all(r['entry_transfer']['safe_entry'] for r in native+entry if r['terrain'] == 'ascent'),
            flat_start_passed=flat_passed(flat),
            production_passed=production_passed(forward), flat_start_score=score(flat))
