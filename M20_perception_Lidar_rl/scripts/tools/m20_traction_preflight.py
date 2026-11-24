"""Validate next-stage baselines without Isaac or any optimizer update."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
from traction_selection import validated_results
from m20_step_retention_preflight import validate as retention_validate


def validate(checkpoint, teacher, replay, retention_results, traction_results,
             resume_checkpoint=None, allow_initial_refinement=False):
    retained = retention_validate(teacher, replay, retention_results)
    # A fresh stage must start from the exact teacher used to collect the
    # physical reference suite.  A continuation checkpoint is allowed to
    # differ, but its saved retention state must still point to that same
    # teacher/replay pair.
    seed_checkpoint = teacher if resume_checkpoint or allow_initial_refinement else checkpoint
    rows, manifest = validated_results(traction_results, seed_checkpoint)
    if manifest['duration'] != 14. or sorted(manifest['frictions']) != [.6, 1.]:
        raise ValueError('Launch baseline must use the fixed 14-second, two-friction suite')
    # The inherited seed selector/replay uses one physical teacher. Preparing
    # a later selected teacher requires regenerating matching replay/evidence.
    if resume_checkpoint:
        import torch
        state = torch.load(resume_checkpoint, map_location='cpu', weights_only=True).get('m20_step_retention')
        if not state or state.get('reference_sha256') != hashlib.sha256(Path(teacher).read_bytes()).hexdigest():
            raise ValueError('Continuation checkpoint has incompatible retention teacher')
        if state.get('replay_sha256') != hashlib.sha256(Path(replay).read_bytes()).hexdigest():
            raise ValueError('Continuation checkpoint has incompatible retention replay')
    elif not allow_initial_refinement and hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != hashlib.sha256(Path(teacher).read_bytes()).hexdigest():
        raise ValueError('This stage requires matching initial checkpoint, retention teacher and replay')
    return dict(retained, traction_cases=len(rows), initial_checkpoint=str(Path(checkpoint).resolve()),
                no_training_started=True,
                fresh_optimizer_required=not bool(resume_checkpoint),
                continuation_checkpoint=(str(Path(resume_checkpoint).resolve())
                                         if resume_checkpoint else None))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'teacher', 'replay', 'retention-results', 'traction-results'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--allow-initial-refinement', action='store_true', help='Fresh FlatStart stage; retain the existing stair teacher/replay with a separately validated actor checkpoint')
    a = p.parse_args()
    print(json.dumps(validate(a.checkpoint, a.teacher, a.replay, a.retention_results,
                              a.traction_results, a.resume_checkpoint, a.allow_initial_refinement), indent=2))
