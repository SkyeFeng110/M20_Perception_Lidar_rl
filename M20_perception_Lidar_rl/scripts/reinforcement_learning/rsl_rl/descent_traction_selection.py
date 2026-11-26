"""V2 selection: descent route and every low-mu speed, with unchanged gait gates."""
import json
import math
import shutil
import hashlib
from pathlib import Path
from early_traction_selection import EarlyTractionBehaviorSelector
from traction_selection import identity
import front_turn_selection as turn

PROTOCOL = 'm20_descent_low_mu_58098_v2'


def descent_error(rows, seed):
    before = {r['case']: r for r in seed}
    terms = []
    for row in rows:
        a, b = row['retention_pose'], before[row['case']]['retention_pose']
        terms.append(.5*a['yaw_peak_deg']/max(1., b['yaw_peak_deg'])
                     +.5*a['lateral_peak_m']/max(.03, b['lateral_peak_m']))
    return sum(terms)/len(terms)


def low_mu_errors(rows, seed):
    before = {identity(r): r for r in seed}
    result = []
    for row in rows:
        if row['terrain'] != 'ascent' or row['friction'] != .6:
            continue
        a, b = row.get('early_ascent_slip_p95'), before[identity(row)].get('early_ascent_slip_p95')
        # A valid physical failure with no loaded B1 sample is a rejected
        # candidate, not an evaluator exception that terminates training.
        result.append(a/max(.05, b) if isinstance(a, (int, float)) and math.isfinite(a) else 4.)
    if len(result) != 8:
        raise ValueError('V2 low-mu evaluation must cover both treads and all four speeds')
    return result


def low_mu_error(rows, seed):
    values = low_mu_errors(rows, seed)
    # Include the worst speed explicitly: high-speed gains cannot mask the
    # .5/.7m/s regressions seen in V1.
    return .7*sum(values)/len(values)+.3*max(values)


class DescentTractionBehaviorSelector(EarlyTractionBehaviorSelector):
    def __init__(self, *args, turn_reference_results=None, **kwargs):
        self._turn_ready = False
        self._pending_turn = None
        self.turn_seed = turn.validate(turn_reference_results, args[1]) if turn_reference_results else []
        self.turn_rows = self.turn_seed
        self.turn_seed_path = str(Path(turn_reference_results).resolve()) if turn_reference_results else None
        self.turn_path = self.turn_seed_path
        previous = kwargs.get('previous_best')
        if previous and previous.get('repair_protocol') != PROTOCOL:
            raise ValueError('V2 selection state belongs to another repair objective')
        super().__init__(*args, **kwargs)
        if previous and previous.get('turn_results'):
            self.turn_path = previous['turn_results']
            self.turn_rows = turn.validate(self.turn_path, self.best_checkpoint)
            if turn.regressions(self.turn_rows, self.turn_seed):
                raise ValueError('Restored candidate regresses turn retention')
        self._turn_ready = True
        self._archive(self.best_checkpoint, self.best_results,
                      seed=hashlib.sha256(Path(self.best_checkpoint).read_bytes()).hexdigest() == self.seed_sha256)

    @staticmethod
    def _early_slip_score(rows):
        values = [r.get('early_ascent_slip_p95') for r in rows
                  if r['terrain'] == 'ascent' and r['friction'] == .6]
        if len(values) != 8:
            raise ValueError('V2 needs all eight low-mu ascent cases')
        return sum(v if isinstance(v, (int, float)) and math.isfinite(v) else 3. for v in values)/8.

    def _traction_regressions(self, rows):
        failures = super()._traction_regressions(rows)
        for reference in (self.traction_seed, self.traction_rows):
            before = {identity(r): r for r in reference}
            for row in rows:
                if row['terrain'] != 'ascent' or row['friction'] != .6:
                    continue
                value, old = row.get('early_ascent_slip_p95'), before[identity(row)].get('early_ascent_slip_p95')
                if value is None or value > old*1.05+.01:
                    failures.append('traction/'+str(identity(row))+': low-mu early-tread slip regression')
        return sorted(set(failures))

    def _traction_repaired(self, rows):
        return not self._traction_regressions(rows) and low_mu_error(rows, self.traction_seed) <= .85

    def _regressions(self, *args):
        failures = super()._regressions(*args)
        if self._descent_ready:
            rows = self._pending_descent[0] if self._pending_descent else self.descent_rows
            for reference in (self.descent_seed, self.descent_rows):
                before = {r['case']: r for r in reference}
                for row in rows:
                    a, b = row['retention_pose'], before[row['case']]['retention_pose']
                    for key, margin in (('yaw_peak_deg', .5), ('lateral_peak_m', .025)):
                        if a[key] > b[key]+margin:
                            failures.append(row['case']+': V2 '+key)
        if self._turn_ready and self.turn_seed:
            rows = self._pending_turn[0] if self._pending_turn else self.turn_rows
            failures += turn.regressions(rows, self.turn_seed)+turn.regressions(rows, self.turn_rows)
        return sorted(set(failures))

    def _promotion_rank(self, rows, route_value, *, candidate):
        descent = self._pending_descent[0] if candidate and self._pending_descent else self.descent_rows
        d = descent_error(descent, self.descent_seed)
        t = low_mu_error(rows, self.traction_seed)
        turns = self._pending_turn[0] if candidate and self._pending_turn else self.turn_rows
        u = turn.error(turns, self.turn_seed) if self.turn_seed else 1.
        return (-.4*d-.35*t-.25*u, route_value)

    def _rank_improved(self, candidate, best):
        d = descent_error(self._pending_descent[0], self.descent_seed)
        old = descent_error(self.descent_rows, self.descent_seed)
        return d <= min(1., old)+.02 and candidate[0] > best[0]+.005

    def _archive(self, *args, **kwargs):
        if self._turn_ready and self._pending_turn:
            self.turn_rows, self.turn_path = self._pending_turn
        super()._archive(*args, **kwargs)
        if not self._descent_ready:
            return
        path = self.output/'best_candidate.json'
        record = json.loads(path.read_text())
        d = descent_error(self.descent_rows, self.descent_seed)
        t = low_mu_error(self.traction_rows, self.traction_seed)
        repaired = d <= .85
        unresolved = turn.targets(self.turn_rows, self.turn_seed) if self.turn_seed else ['turn/baseline missing']
        if self._turn_ready and self.turn_path:
            target = Path(self.best_checkpoint).parent/'front_turn'
            target.mkdir(exist_ok=True)
            for name in ('results.json', 'manifest.json'):
                source = Path(self.turn_path).parent/name
                if source.resolve() != (target/name).resolve():
                    shutil.copy2(source, target/name)
            self.turn_path = str(target/'results.json')
        u = turn.error(self.turn_rows, self.turn_seed) if self.turn_seed else 1.
        record.update(selection_focus='descent route, low-mu traction and grounded front-led turning; 58098 gait retained',
            repair_protocol=PROTOCOL, descent_error_ratio=d, low_mu_error_ratio=t,
            descent_repair_passed=repaired, combined_error=.4*d+.35*t+.25*u,
            turn_error_ratio=u, turn_repair_passed=not unresolved, turn_unresolved=unresolved,
            turn_results=self.turn_path, turn_reference_results=self.turn_seed_path)
        record['fully_accepted'] &= repaired and not unresolved
        record['route_hold_passed'] &= repaired and not unresolved
        path.write_text(json.dumps(record, indent=2)+'\n')

    def export_state(self):
        return dict(super().export_state(), repair_protocol=PROTOCOL, turn_results=self.turn_path)

    def consider(self, *args, **kwargs):
        # Parent validates and gates descent before any promotion.
        from early_traction_selection import validated_descent
        rows = validated_descent(kwargs['descent_results'], args[0])
        d = descent_error(rows, self.descent_seed)
        traction = json.loads(Path(kwargs['traction_results']).read_text())
        t = low_mu_error(traction, self.traction_seed)
        turn_path = kwargs.pop('turn_results', None)
        if self.turn_seed and turn_path is None:
            raise ValueError('V2 candidate requires all fixed front-turn cases')
        turns = turn.validate(turn_path, args[0]) if turn_path else []
        self._pending_turn = (turns, str(Path(turn_path).resolve())) if turn_path else None
        try:
            result = super().consider(*args, **kwargs)
        finally:
            self._pending_turn = None
        u = turn.error(turns, self.turn_seed) if self.turn_seed else 1.
        unresolved = turn.targets(turns, self.turn_seed) if self.turn_seed else ['turn/baseline missing']
        result.update(descent_error_ratio=d, low_mu_error_ratio=t,
            combined_error=.4*d+.35*t+.25*u, descent_repair_passed=d <= .85,
            turn_error_ratio=u, turn_repair_passed=not unresolved, turn_unresolved=unresolved)
        result['route_hold_passed'] &= d <= .85 and not unresolved
        result['feedback_failures'] += unresolved
        if d > .85:
            result['feedback_failures'].append('descent/route: original-route improvement target unresolved')
        return result
