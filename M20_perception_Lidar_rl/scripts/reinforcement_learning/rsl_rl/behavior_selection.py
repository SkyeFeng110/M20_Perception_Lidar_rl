"""Keep a separate, measured best candidate without stopping or rolling back PPO."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]


def identity(row):
    return (row['profile'], row['terrain'], row['tread'], row['command_speed'])


def regressions(rows, reference):
    """Protect acquired capabilities per case, never just the total pass count."""
    baseline = {identity(row): row for row in reference}
    if {identity(row) for row in rows} != set(baseline):
        raise ValueError('Behavior selection requires identical case sets')
    failures = []
    for row in rows:
        old = baseline[identity(row)]
        label = '/'.join(map(str, identity(row)))
        reasons = []
        for field in ('fell', 'observed_bypass', 'stall_detected'):
            if row[field] and not old[field]:
                reasons.append(field)
        for field in ('policy_entered', 'whole_body_top_supported', 'all_treads_supported'):
            if old.get(field) and not row.get(field):
                reasons.append('lost '+field)
        for field, tolerance in (('forward_yaw_max_change_deg', 3.),
                                 ('after_c_idle_yaw_max_change_deg', 1.),
                                 ('forward_lateral_max_abs', .10)):
            a, b = row.get(field), old.get(field)
            if isinstance(b, (int, float)) and (a is None or not math.isfinite(a) or a > b+tolerance):
                reasons.append(field)
        speed_field = 'later_stairs_route_velocity_x' if row['terrain'] == 'ascent' else 'route_mean_velocity_x'
        a, b = row.get(speed_field), old.get(speed_field)
        # Reducing prior overspeed toward the command is a valid correction.
        if isinstance(b, (int, float)) and (a is None or a < .90*min(b, row['command_speed'])):
            reasons.append('route speed')
        if row['terrain'] == 'flat':
            a, b = row.get('p95_max_abs_reach'), old.get('p95_max_abs_reach')
            if isinstance(b, (int, float)) and (a is None or a > b+.01):
                reasons.append('flat workspace')
        if reasons:
            failures.append(label+': '+', '.join(reasons))
    return failures


def rank(rows):
    """Sequential support first; speed, yaw, lateral and flat stance break ties."""
    stairs = [row for row in rows if row['terrain'] == 'ascent']
    full = sum(bool(row.get('all_treads_supported')) for row in stairs)
    gaps = sum(len(side) for row in stairs for side in (row.get('skipped_treads') or []))
    error = 0.
    for row in rows:
        ascent = row['terrain'] == 'ascent'
        field = 'later_stairs_route_velocity_x' if ascent else 'route_mean_velocity_x'
        speed = row.get(field) or 0.
        error += max(0., .9-speed/row['command_speed'])
        yaw = row.get('forward_yaw_max_change_deg')
        error += (180. if yaw is None else yaw)/(8. if ascent else 3.)
        error += (row.get('forward_lateral_max_abs') or 0.)/(.35 if ascent else .15)
        if not ascent:
            error += (row.get('p95_max_abs_reach') or 0.)/.10
    return (full, -gaps, -error)


class BehaviorSelector:
    def __init__(self, output, checkpoint, results, previous_best=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.seed_rows = self.rows = self._validated_results(checkpoint, results)
        seed = True
        if previous_best:
            if previous_best['reference_sha256'] != self.seed_sha256:
                raise ValueError('Saved best candidate belongs to a different teacher')
            checkpoint, results = previous_best['checkpoint'], previous_best['results']
            rows = self._validated_results(checkpoint, results)
            if regressions(rows, self.seed_rows) or rank(rows) < rank(self.seed_rows):
                raise ValueError('Saved best candidate violates the seed capability protection')
            self.rows = rows
            seed = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == self.seed_sha256
        self._archive(checkpoint, results, seed=seed)

    def export_state(self):
        """Carry the measured best across interrupted training sessions."""
        return dict(reference_sha256=self.seed_sha256, checkpoint=self.best_checkpoint,
                    results=self.best_results)

    @staticmethod
    def _validated_results(checkpoint, results):
        results = Path(results)
        rows = json.loads(results.read_text())
        spec = importlib.util.spec_from_file_location('m20_selection_gate', ROOT/'scripts/tools/m20_repair_acceptance.py')
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        failures = gate.validate_rows(rows, json.loads(gate.CRITERIA.read_text()))
        if failures:
            raise ValueError('Invalid selection evidence: '+'; '.join(failures))
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        for profile in ('standing', 'keyboard'):
            manifest = json.loads((results.parent/profile/'manifest.json').read_text())
            if manifest.get('checkpoint_sha256') != digest or not manifest.get('physics_enabled'):
                raise ValueError('Best candidate evidence does not match checkpoint physics manifest')
        return rows

    def _archive(self, checkpoint, results, *, seed=False):
        checkpoint = Path(checkpoint).resolve()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        destination = self.output/'best_candidates'/digest[:16]
        destination.mkdir(parents=True, exist_ok=True)
        snapshot = destination/checkpoint.name
        if not snapshot.exists():
            shutil.copy2(checkpoint, snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError('Archived best candidate checksum mismatch')
        if not (destination/'params').exists():
            params = checkpoint.parent/'params'
            if not params.is_dir():
                import sys
                sys.path.insert(0, str(ROOT.parent/'mujoco_validation/deploy/mujoco_v135'))
                from checkpoint_config import resolve_configs
                params, _, _ = resolve_configs(checkpoint)
            shutil.copytree(params, destination/'params')
            association = checkpoint.with_suffix('.config.json')
            if association.is_file():
                shutil.copy2(association, destination/'execution_config_origin.json')
        archived_results = destination/'results.json'
        if Path(results).resolve() != archived_results.resolve():
            shutil.copy2(results, archived_results)
        for profile in ('standing', 'keyboard'):
            target = destination/profile/'manifest.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            source = Path(results).parent/profile/'manifest.json'
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        self.best_checkpoint = str(snapshot)
        self.best_results = str(archived_results)
        record = dict(source_checkpoint=str(checkpoint), checkpoint=str(snapshot),
                      checkpoint_sha256=digest, results=self.best_results,
                      source_results=str(Path(results).resolve()),
                      rank=rank(self.rows), seed=seed, fully_accepted=False,
                      scope='Best measured candidate; consult strict acceptance separately')
        # Strict acceptance remains a separate result and is never inferred
        # from this relative ranking or from the seed's candidate status.
        acceptance = Path(results).parent/'acceptance.json'
        if acceptance.exists():
            record['fully_accepted'] = json.loads(acceptance.read_text()).get('repair_passed') is True
            if acceptance.resolve() != (destination/'acceptance.json').resolve():
                shutil.copy2(acceptance, destination/'acceptance.json')
        temp = self.output/'best_candidate.json.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(self.output/'best_candidate.json')

    def consider(self, checkpoint, results):
        rows = self._validated_results(checkpoint, results)
        failures = sorted(set(regressions(rows, self.seed_rows)+regressions(rows, self.rows)))
        promoted = not failures and rank(rows) > rank(self.rows)
        if promoted:
            self.rows = rows
            self._archive(checkpoint, results)
        return dict(promoted=promoted, regressions=failures,
                    candidate_rank=rank(rows), best_rank=rank(self.rows),
                    best_checkpoint=self.best_checkpoint)
