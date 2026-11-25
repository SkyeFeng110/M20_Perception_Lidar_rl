"""Flat-first selection under measured current-entry and later-gait retention."""
import json
import math
from pathlib import Path
from behavior_selection import BehaviorSelector, regressions as native_regressions, identity
from entry_selection import extra_results, gait_regressions
from flat_start_selection import (FlatEntryBehaviorSelector, validated_results,
    validated_forward, regressions as flat_regressions, passed, score)


def entry_retention(rows, reference):
    failures = gait_regressions(rows, reference)
    before = {identity(r): r for r in reference}
    for row in rows:
        if row['terrain'] != 'ascent':
            continue
        old = before[identity(row)]
        for side, tread in enumerate(old['first_supported_tread']):
            if tread == 1 and row['first_supported_tread'][side] != 1:
                failures.append('entry/'+str(identity(row))+': lost first B1 wheel '+str(side))
        if (old['entry_transfer']['lead_side'] is not None
                and row['entry_transfer']['lead_side'] != old['entry_transfer']['lead_side']):
            failures.append('entry/'+str(identity(row))+': changed lead side')
    return failures


def posture_score(native, flat, production):
    raw = [r for r in flat if r['case']['name'].startswith('raw_')]
    stance = [r for r in native+production if r['terrain'] == 'flat']
    return score(raw)-sum(r['sagittal_target_jump_peak_deg']/20.
        +r['wheel_target_jump_peak_rad_s']/5.
        +10.*max(0., .9-r['final_speed_m_s']/r['case']['speed']) for r in raw)-sum(2.*(r['p95_max_abs_reach'] if isinstance(r.get('p95_max_abs_reach'), (int, float)) else 1.)/.10
        + max(0., .9-(r['route_mean_velocity_x'] or 0.)/r['command_speed']) for r in stance)


def flat_passed(rows):
    return passed(rows) and all(r['final_speed_m_s'] >= .9*r['case']['speed']
        and r['sagittal_target_jump_peak_deg'] <= 20.
        and r['wheel_target_jump_peak_rad_s'] <= 5. for r in rows)


def production_passed(rows):
    return all(r['policy_entered'] and not r['fell'] and not r['stall_detected'] and not r['observed_bypass']
        and (r['whole_body_top_supported'] if r['terrain'] == 'ascent' else
             (isinstance(r.get('p95_max_abs_reach'), (int, float))
              and isinstance(r.get('route_mean_velocity_x'), (int, float))
              and r['p95_max_abs_reach'] <= .12 and r['route_mean_velocity_x'] >= .9*r['command_speed']
              and r['forward_yaw_max_change_deg'] <= 3. and r['forward_lateral_max_abs'] <= .15))
        for r in rows)


class FlatPostureBehaviorSelector(FlatEntryBehaviorSelector):
    def __init__(self, output, checkpoint, results, entry_results, flat_reference_results,
                 flat_reference_checkpoint, forward_reference_results, previous_best=None):
        self.flat_seed = self.flat_rows = validated_results(flat_reference_results, flat_reference_checkpoint)
        self.flat_path = str(flat_reference_results)
        self.forward_seed = self.forward_rows = validated_forward(forward_reference_results, flat_reference_checkpoint)
        self.forward_path = str(forward_reference_results)
        self.entry_seed = self.entry_rows = extra_results(entry_results, checkpoint)
        self.entry_path = str(entry_results)
        self.traction_seed = self.traction_rows = self.traction_manifest = self.traction_path = None
        # Restore using flat-first ranking, not the former lexicographic B1
        # ranking. Otherwise a harmless entry timing change blocks resume.
        BehaviorSelector.__init__(self, output, checkpoint, results)
        if previous_best:
            if previous_best['reference_sha256'] != self.seed_sha256:
                raise ValueError('FlatPosture best belongs to another stair teacher')
            cp = previous_best['checkpoint']
            native = self._validated_results(cp, previous_best['results'])
            entry = extra_results(previous_best['entry_results'], cp)
            flat = validated_results(previous_best['flat_results'], cp)
            forward = validated_forward(previous_best['forward_results'], cp)
            failures = self._regressions(native, entry, flat, forward)
            if failures:
                raise ValueError('Restored flat best violates retention: '+'; '.join(failures))
            self.rows, self.entry_rows, self.flat_rows, self.forward_rows = native, entry, flat, forward
            self.entry_path, self.flat_path, self.forward_path = (previous_best[k] for k in
                ('entry_results', 'flat_results', 'forward_results'))
            self._archive(cp, previous_best['results'])

    def _regressions(self, native, entry, flat, forward):
        failures = []
        for rows, seed, best in ((native, self.seed_rows, self.rows),
                                 (entry, self.entry_seed, self.entry_rows),
                                 (forward, self.forward_seed, self.forward_rows)):
            for reference in (seed, best):
                failures += native_regressions(rows, reference)+entry_retention(rows, reference)
        failures += flat_regressions(flat, self.flat_seed)+flat_regressions(flat, self.flat_rows)
        # This phase must not win its posture score by reducing flat speed.
        for reference in (self.flat_seed, self.flat_rows):
            before = {r['case']['name']:r for r in reference}
            for r in flat:
                b = before[r['case']['name']]
                if r['final_speed_m_s'] < .98*min(b['final_speed_m_s'], .9*r['case']['speed']):
                    failures.append('flat/'+r['case']['name']+': speed traded for posture')
        for candidate, reference in ((native, self.seed_rows), (native, self.rows),
                                      (forward, self.forward_seed), (forward, self.forward_rows)):
            before = {identity(r):r for r in reference}
            for r in candidate:
                if r['terrain'] == 'flat':
                    b = before[identity(r)]
                    if (r['route_mean_velocity_x'] or 0.) < .98*min(b['route_mean_velocity_x'], .9*r['command_speed']):
                        failures.append('flat/'+str(identity(r))+': speed traded for posture')
        return sorted(set(failures))

    def _archive(self, *args, **kwargs):
        super()._archive(*args, **kwargs)
        path = self.output/'best_candidate.json'
        record = json.loads(path.read_text())
        record.update(selection_focus='flat posture; acquired entry/later gait protected',
            flat_posture_score=posture_score(self.rows, self.flat_rows, self.forward_rows),
            flat_start_passed=flat_passed(self.flat_rows), production_passed=production_passed(self.forward_rows),
            flat_posture_passed=flat_passed(self.flat_rows) and production_passed(self.forward_rows))
        path.write_text(json.dumps(record, indent=2)+'\n')

    def consider(self, checkpoint, results, entry_results, flat_results, forward_results, traction_results=None):
        native = self._validated_results(checkpoint, results)
        entry = extra_results(entry_results, checkpoint)
        flat = validated_results(flat_results, checkpoint)
        forward = validated_forward(forward_results, checkpoint)
        failures = self._regressions(native, entry, flat, forward)
        candidate = posture_score(native, flat, forward)
        best = posture_score(self.rows, self.flat_rows, self.forward_rows)
        promoted = not failures and candidate > best+.01
        if promoted:
            self.rows, self.entry_rows, self.flat_rows, self.forward_rows = native, entry, flat, forward
            self.entry_path, self.flat_path, self.forward_path = map(str, (entry_results, flat_results, forward_results))
            self._archive(checkpoint, results)
        return dict(promoted=promoted, regressions=failures, candidate_rank=(0, candidate),
            best_rank=(0, max(candidate, best) if promoted else best), best_checkpoint=self.best_checkpoint,
            entry_passed=all(r['entry_transfer']['safe_entry'] for r in native+entry if r['terrain'] == 'ascent'),
            traction_passed=None, traction_repair_passed=None, flat_start_passed=flat_passed(flat),
            production_passed=production_passed(forward), flat_start_score=score(flat))
