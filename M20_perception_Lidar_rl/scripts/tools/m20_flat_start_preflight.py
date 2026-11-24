"""Check measured flat protocol and optional resume without starting Isaac."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
from flat_start_selection import validated_results, validated_forward


def validate(reference, teacher, forward_results, resume=None):
    rows = validated_results(reference, teacher)
    forward = validated_forward(forward_results, teacher)
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True).get('m20_flat_start')
        if not state or state.get('protocol') != 'flat_start_v1':
            raise ValueError('Resume requires FlatStart checkpoint; use old models as a fresh initial checkpoint')
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if (sha(state['flat_reference_results']) != sha(reference)
                or sha(state['flat_reference_checkpoint']) != sha(teacher)
                or sha(state['forward_reference_results']) != sha(forward_results)):
            raise ValueError('FlatStart continuation baseline changed')
    return dict(status='flat_start_preflight_passed', flat_cases=len(rows), production_cases=len(forward), no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--forward-results', type=Path, required=True)
    p.add_argument('--resume-checkpoint', type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.results, a.teacher, a.forward_results, a.resume_checkpoint), indent=2))
