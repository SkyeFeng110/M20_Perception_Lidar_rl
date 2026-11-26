"""Portable recovery assets resolved from the repository and explicit overrides."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'data/m20_recovery'


def resolve_assets():
    teacher = os.environ.get('M20_RECOVERY_TEACHER') or os.environ.get('M20_BASE_CHECKPOINT')
    replay = os.environ.get('M20_RECOVERY_REPLAY')
    # Explicit stage assets are self-contained. The historical 35300 replay
    # is needed only to fill defaults, not to initialize every later stage.
    if not teacher or not replay:
        metadata_path = DATA/'reference_observations_v2.json'
        try:
            metadata = json.loads(metadata_path.read_text())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'Recovery default metadata is missing: {metadata_path}. '
                'Set both M20_RECOVERY_TEACHER and M20_RECOVERY_REPLAY '
                'to the selected stage assets, or restore the original default metadata.'
            ) from error
        teacher = teacher or metadata['source_checkpoint']
        replay = replay or metadata['replay_path']
    def path(value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
    return path(teacher), path(replay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    teacher, replay = resolve_assets()
    if args.teacher:
        print(teacher)
    elif args.replay:
        print(replay)
    else:
        print(json.dumps(dict(teacher=str(teacher), replay=str(replay))))
