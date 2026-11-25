"""Route-first selection with the measured 56898 posture and gait protected."""
import json
import math
from behavior_selection import identity
from entry_selection import extra_results
from flat_start_selection import validated_results, validated_forward, score
from flat_posture_selection import FlatPostureBehaviorSelector, flat_passed, production_passed


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
        stage_passed = (not unresolved and retained and flat_passed(self.flat_rows)
                        and production_passed(self.forward_rows))
        record.update(selection_focus='original route; 56898 flat posture and stair gait protected',
            route_score=route_score(self.rows, self.entry_rows, self.flat_rows, self.forward_rows),
            route_hold_passed=stage_passed, route_unresolved=unresolved)
        # Relative improvement never certifies a still-biased seed/candidate.
        record['fully_accepted'] &= stage_passed
        path.write_text(json.dumps(record, indent=2)+'\n')

    def consider(self, checkpoint, results, entry_results, flat_results, forward_results, traction_results=None):
        native = self._validated_results(checkpoint, results)
        entry = extra_results(entry_results, checkpoint)
        flat = validated_results(flat_results, checkpoint)
        forward = validated_forward(forward_results, checkpoint)
        failures = self._regressions(native, entry, flat, forward)
        unresolved = route_failures(native, entry, flat, forward)
        candidate = route_score(native, entry, flat, forward)
        best = route_score(self.rows, self.entry_rows, self.flat_rows, self.forward_rows)
        promoted = not failures and candidate > best+.01
        if promoted:
            self.rows, self.entry_rows, self.flat_rows, self.forward_rows = native, entry, flat, forward
            self.entry_path, self.flat_path, self.forward_path = map(str, (entry_results, flat_results, forward_results))
            self._archive(checkpoint, results)
        return dict(promoted=promoted, regressions=failures, route_unresolved=unresolved,
            feedback_failures=sorted(set(failures+unresolved)),
            candidate_rank=(candidate,), best_rank=(candidate if promoted else best,),
            best_checkpoint=self.best_checkpoint,
            route_hold_passed=not failures and not unresolved and flat_passed(flat) and production_passed(forward),
            entry_passed=all(r['entry_transfer']['safe_entry'] for r in native+entry if r['terrain'] == 'ascent'),
            traction_passed=None, traction_repair_passed=None, flat_start_passed=flat_passed(flat),
            production_passed=production_passed(forward), flat_start_score=score(flat))
