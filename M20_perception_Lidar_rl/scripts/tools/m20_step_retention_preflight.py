"""Verify best-teacher behavior evidence and its replay; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
from behavior_selection import BehaviorSelector


def validate(teacher, replay, results, resume_checkpoint=None):
    rows = BehaviorSelector._validated_results(teacher, results)
    metadata = json.loads(Path(replay).with_suffix('.json').read_text())
    if (metadata['source_sha256'] != hashlib.sha256(Path(teacher).read_bytes()).hexdigest()
            or metadata['replay_sha256'] != hashlib.sha256(Path(replay).read_bytes()).hexdigest()):
        raise ValueError('Teacher and replay hash mismatch')
    suites = {str((Path(results).parent/p/'manifest.json').resolve()) for p in ('standing', 'keyboard')}
    for case in metadata['cases']:
        if case['manifest'] not in suites:
            raise ValueError('Replay and behavior evidence refer to different benchmark suites')
        if hashlib.sha256(Path(case['manifest']).read_bytes()).hexdigest() != case['manifest_sha256']:
            raise ValueError('Replay source manifest changed')
    success = sum(r['terrain'] == 'ascent' and r['all_treads_supported']
                  and r['whole_body_top_supported'] and not r['fell']
                  and not r['stall_detected'] and not r['observed_bypass'] for r in rows)
    if not success:
        raise ValueError('Selected teacher has no successful whole-flight support cases')
    if resume_checkpoint:
        import torch
        state = torch.load(resume_checkpoint, map_location='cpu', weights_only=True).get('m20_step_retention')
        if not state or (state['reference_sha256'] != metadata['source_sha256']
                         or state['replay_sha256'] != metadata['replay_sha256']):
            raise ValueError('Continuation requires saved StepRetention protection state and matching assets')
        if not state.get('best_candidate'):
            raise ValueError('Continuation checkpoint is missing the measured best candidate')
        best = state['best_candidate']
        if best['reference_sha256'] != metadata['source_sha256']:
            raise ValueError('Continuation best candidate belongs to a different teacher')
        best_rows = BehaviorSelector._validated_results(best['checkpoint'], best['results'])
        from behavior_selection import regressions, rank
        if regressions(best_rows, rows) or rank(best_rows) < rank(rows):
            raise ValueError('Continuation best candidate violates seed capability protection')
    return dict(teacher=str(Path(teacher).resolve()), successful_ascent_cases=success,
                replay_samples=metadata['samples'], no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--resume-checkpoint', type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.teacher, a.replay, a.results, a.resume_checkpoint), indent=2))
