"""Build current-gait protection and physically compact moving flat targets."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from m20_build_recovery_replay import build, sha256

PROTOCOL = 'flat_posture_current_stairs_compact_flat_v1'
ROOT = Path(__file__).resolve().parents[2]


def build_posture(native, teacher, flat_trajectory, flat_checkpoint, output, preserve_current_flat=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay = output/'current_gait.npz'
    meta = build(Path(native).parent, teacher, replay)
    xml = ROOT.parent/'mujoco_validation/deploy/mujoco_v135/terrains/stairs_ascent_walls.xml'
    start = float(ET.parse(xml).find(".//numeric[@name='start_x']").attrib['data'])
    with np.load(replay, allow_pickle=False) as d:
        packed = {key: d[key].copy() for key in d.files}
    masks = []
    for case in meta['cases']:
        with np.load(case['trajectory'], allow_pickle=False) as d:
            keep = d['policy_active'] & np.isin(d['phase'], [1, 2, 3])
            onset = d['wheel_position'][:, :2, 0].max(1) >= start-.50
            mask = np.maximum.accumulate(onset) if case['terrain'] == 'ascent' else np.zeros(len(keep), bool)
            masks.append(mask[keep])
    packed['stair_keep'] = np.concatenate(masks)
    np.savez_compressed(replay, **packed)
    meta.update(posture_protocol=PROTOCOL, stair_start=start,
                replay_sha256=sha256(replay), stair_keep_samples=int(packed['stair_keep'].sum()))
    replay.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    production = Path(native).resolve().parent.parent/'production/current'
    deployed = {key: [] for key in ('observation', 'action')}
    sources = []
    for row in json.loads((production/'results.json').read_text()):
        if row['terrain'] != 'ascent':
            continue
        path = production/f"ascent_{row['tread']}_{row['command_speed']}.npz"
        manifest_path = production/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('checkpoint_sha256') != sha256(teacher) or not manifest.get('production_direct'):
            raise ValueError('Stair retention requires current raw full-speed physics')
        with np.load(path, allow_pickle=False) as d:
            keep = ((d['phase'] == 2)
                    & np.maximum.accumulate(d['wheel_position'][:, :2, 0].max(1) >= start-.5))
            for key in deployed:
                deployed[key].append(d[key][keep])
            sources.append(dict(trajectory=str(path), trajectory_sha256=sha256(path),
                manifest=str(manifest_path), manifest_sha256=sha256(manifest_path)))
    deployed_path = output/'deployed_stairs.npz'
    np.savez_compressed(deployed_path, **{key: np.concatenate(value) for key, value in deployed.items()})
    deployed_path.with_suffix('.json').write_text(json.dumps(dict(protocol=PROTOCOL,
        sha256=sha256(deployed_path), source_sha256=sha256(teacher), sources=sources), indent=2)+'\n')
    flat_trajectory, flat_checkpoint = Path(flat_trajectory).resolve(), Path(flat_checkpoint).resolve()
    manifest = flat_trajectory.parent/'manifest.json'
    contract = json.loads(manifest.read_text())
    if (contract.get('checkpoint_sha256') != sha256(flat_checkpoint)
            or not contract.get('physics_enabled') or not contract.get('production_direct')):
        raise ValueError('Compact reference needs measured raw production flat physics')
    with np.load(flat_trajectory, allow_pickle=False) as d:
        moving = d['phase'] == 2
        start_time = d['time'][moving][0]
        q = d['quaternion']
        pitch = np.arcsin(np.clip(2*(q[:, 0]*q[:, 2]-q[:, 3]*q[:, 1]), -1, 1))
        speed = np.gradient(d['position'][:, 0], d['time'])
        keep = (moving & (d['time'] >= start_time+2.)
            & (np.abs(d['reach']).max(1) <= .12) & (np.abs(pitch) <= np.radians(2.))
            & (np.abs(d['angular_velocity'][:, 1]) <= .35)
            & (speed >= .8*d['command'][:, 0]) & (d['command'][:, 0] > .1))
        if preserve_current_flat:
            # A retention phase preserves the selected teacher's measured
            # posture; it must not silently substitute an older compact gait.
            if sha256(flat_checkpoint) != sha256(teacher):
                raise ValueError('Current-flat preservation must use the selected stair teacher')
            keep = (moving & (d['time'] >= start_time+2.)
                    & (speed >= .8*d['command'][:, 0]) & (d['command'][:, 0] > .1))
        if keep.sum() < 50:
            raise ValueError('Insufficient physically compact moving reference')
        compact = output/'compact_flat.npz'
        np.savez_compressed(compact, observation=d['observation'][keep], action=d['action'][keep],
            source_index=np.flatnonzero(keep), reach=d['reach'][keep], speed=speed[keep])
    compact.with_suffix('.json').write_text(json.dumps(dict(protocol=PROTOCOL,
        sha256=sha256(compact), source_checkpoint=str(flat_checkpoint), source_sha256=sha256(flat_checkpoint),
        trajectory=str(flat_trajectory), trajectory_sha256=sha256(flat_trajectory),
        manifest=str(manifest), manifest_sha256=sha256(manifest), samples=int(keep.sum()),
        preserve_current_flat=bool(preserve_current_flat),
        scope=('Measured current moving flat posture' if preserve_current_flat else
               'Measured compact sagittal moving posture only; startup impulse and steering are excluded')), indent=2)+'\n')
    return dict(current_gait_samples=meta['samples'], stair_keep_samples=meta['stair_keep_samples'],
                compact_flat_samples=int(keep.sum()), no_training_started=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'teacher', 'flat-trajectory', 'flat-checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(build_posture(**vars(p.parse_args())), indent=2))
