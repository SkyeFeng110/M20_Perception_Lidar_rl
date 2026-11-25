"""Learn route stability while softly retaining 56898's flat and stair motion."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict

from .v111_flat_posture_runner import FlatPosturePPO
from .step_retention import successful_support_loss
from .targeted_reference import mirror_observation, mirror_action, straight_mask
from .route_hold import PROTOCOL, steering_equivariance


class RouteHoldPPO(FlatPosturePPO):
    def __init__(self, *args, route_flat_replay, route_steering_weight=.15, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(route_flat_replay)
        self.route_flat_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = json.loads(path.with_suffix('.json').read_text())
        if (meta.get('protocol') != PROTOCOL or meta.get('sha256') != self.route_flat_sha256
                or meta.get('teacher_sha256') != self.reference_sha256):
            raise ValueError('RouteHold requires measured flat motion from its stair teacher')
        with np.load(path, allow_pickle=False) as data:
            self.route_flat_obs = torch.as_tensor(data['observation'].copy(), device=self.device)
            self.route_flat_action = torch.as_tensor(data['action'].copy(), device=self.device)
        if (self.route_flat_obs.shape != (len(self.route_flat_action), 63)
                or self.route_flat_action.shape[1:] != (16,) or len(self.route_flat_obs) < 300
                or not torch.isfinite(self.route_flat_obs).all() or not torch.isfinite(self.route_flat_action).all()):
            raise ValueError('Invalid RouteHold flat preservation replay')
        if not .05 <= route_steering_weight <= .5:
            raise ValueError('Invalid steering equivariance weight')
        self.route_base_steering_weight = float(route_steering_weight)
        self.route_steering_weight = float(route_steering_weight)
        self.route_flat_multiplier = 1.
        self.route_hold_protocol = PROTOCOL
        self.route_flat_losses, self.route_steering_losses = [], []

    def extra_anchor_loss(self):
        inherited = super().extra_anchor_loss()
        ids = torch.randint(len(self.route_flat_obs), (self.anchor_batch_size,), device=self.device)
        x, target = self.route_flat_obs[ids], self.route_flat_action[ids]
        reflected = torch.rand(len(ids), device=self.device) < .5
        x = torch.where(reflected[:, None], mirror_observation(x), x)
        target = torch.where(reflected[:, None], mirror_action(target), target)
        direct = self.actor(TensorDict({'policy': x}, batch_size=[len(x)]))
        # Preserve smooth purposeful sagittal actions and mean drive. HipX
        # and wheel differentials are free; no left-turn teacher is imposed.
        preserve = .75*self.route_flat_multiplier*successful_support_loss(direct, target, tolerance=.08)
        ids = torch.randint(len(self.deployed_obs), (self.anchor_batch_size,), device=self.device)
        steering_obs = torch.cat((x, self.deployed_obs[ids]))
        steering_obs = steering_obs[straight_mask(steering_obs)]
        steering = preserve.new_zeros(())
        if len(steering_obs):
            direct = self.actor(TensorDict({'policy': steering_obs}, batch_size=[len(steering_obs)]))
            reflected = self.actor(TensorDict({'policy': mirror_observation(steering_obs)}, batch_size=[len(steering_obs)]))
            steering = self.route_steering_weight*steering_equivariance(direct, reflected)
        self.route_flat_losses.append(float(preserve.detach()))
        self.route_steering_losses.append(float(steering.detach()))
        return inherited+preserve+steering

    def update(self):
        self.route_flat_losses.clear()
        self.route_steering_losses.clear()
        result = super().update()
        result.update(route_flat_preservation=sum(self.route_flat_losses)/max(1, len(self.route_flat_losses)),
                      route_steering_equivariance=sum(self.route_steering_losses)/max(1, len(self.route_steering_losses)),
                      route_steering_weight=self.route_steering_weight,
                      route_flat_multiplier=self.route_flat_multiplier)
        return result

    def behavior_feedback(self, regressions):
        result = super().behavior_feedback(regressions)
        route = [r for r in regressions if any(k in r for k in ('yaw', 'lateral', 'heading', 'route drift'))]
        flat_pose = [r for r in regressions if r.startswith('flat/') and any(k in r for k in
            ('pitch', 'reach', 'workspace', 'target_jump', 'support width', 'speed traded'))]
        self.route_steering_weight = (min(.5, self.route_steering_weight*1.2) if route else
                                     max(self.route_base_steering_weight, self.route_steering_weight/1.2))
        self.route_flat_multiplier = (min(1.5, self.route_flat_multiplier*1.1) if flat_pose else
                                      max(1., self.route_flat_multiplier/1.1))
        result.update(route_regressions=route, route_steering_weight=self.route_steering_weight,
                      route_flat_multiplier=self.route_flat_multiplier)
        return result

    def save(self):
        result = super().save()
        result['m20_route_hold'] = dict(protocol=PROTOCOL, flat_replay_sha256=self.route_flat_sha256,
            steering_weight=self.route_steering_weight, flat_multiplier=self.route_flat_multiplier)
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        resume = load_cfg is None or load_cfg.get('optimizer')
        state = loaded_dict.get('m20_route_hold', {})
        if resume:
            if state.get('protocol') != PROTOCOL or state.get('flat_replay_sha256') != self.route_flat_sha256:
                raise ValueError('RouteHold needs an initial checkpoint and fresh optimizer for the new objective')
        result = super().load(loaded_dict, load_cfg, strict)
        if resume:
            self.route_steering_weight = float(state['steering_weight'])
            self.route_flat_multiplier = float(state['flat_multiplier'])
            if not self.route_base_steering_weight <= self.route_steering_weight <= .5 or not 1 <= self.route_flat_multiplier <= 1.5:
                raise ValueError('Invalid restored RouteHold feedback')
        return result
