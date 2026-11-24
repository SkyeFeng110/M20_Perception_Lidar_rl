"""Real command mapping: current raw full-stick + candidate slew into B1/B2+."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(checkpoint, output, no_trajectories=False):
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    for profile, speeds in (('current', ['1.']), ('flat-start', ['.7', '1.'])):
        target = output/profile
        command = [sys.executable, str(ROOT/'scripts/tools/m20_entry_flat_benchmark.py'),
            '--checkpoint', str(checkpoint), '--output', str(target), '--startup', 'z',
            '--keyboard-profile', '--keyboard-idle-seconds', '0', '--production-profile',
            '--speeds', *speeds]
        if profile == 'current': command.append('--production-direct')
        if no_trajectories: command.append('--no-trajectories')
        with (output/(profile+'.log')).open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        for row in json.loads((target/'results.json').read_text()):
            row['profile'] = profile+'_z_c_8'
            rows.append(row)
        sources.append(dict(profile=profile,
            manifest_sha256=hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest(),
            manifest=json.loads((target/'manifest.json').read_text())))
    manifest = dict(protocol='production_transition_v1', physics_enabled=True,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        current_robot='direct full forward 1.0m/s',
        candidate_forward_acceleration_m_s2=1.5, candidate_forward_deceleration_m_s2=10./3.,
        zero_command_wheel_stop=True, startup='Z->C->immediate drive',
        actor_dimensions=[63, 16], sources=sources)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    return dict(cases=len(rows), physics_enabled=True, training_started=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-trajectories', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.checkpoint.resolve(), a.output.resolve(), a.no_trajectories)))
