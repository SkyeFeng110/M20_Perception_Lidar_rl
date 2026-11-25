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
from .route_hold import PROTOCOL, masked_steering_equivariance

RUN_PROTOCOL = 'm20_route_hold_early_traction_v1'


class RouteHoldPPO(FlatPosturePPO):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        # RSL-RL 5 passes the environment to this factory, not PPO.__init__.
        alg = FlatPosturePPO.construct_algorithm(obs, env, cfg, device)
        alg._reward_env = getattr(env, 'unwrapped', env)
        alg._sync_traction_feedback()
        return alg

    def __init__(self, *args, route_flat_replay, traction_reference_results,
                 route_steering_weight=.15, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(route_flat_replay)
        self.route_flat_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        traction_path = Path(traction_reference_results)
        self.traction_reference_results = str(traction_path.resolve())
        self.traction_reference_sha256 = hashlib.sha256(traction_path.read_bytes()).hexdigest()
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
        self.early_traction_multiplier = 1.
        self._reward_env = None

    def _sync_traction_feedback(self):
        if self._reward_env is not None:
            self._reward_env._m20_route_early_traction_multiplier = self.early_traction_multiplier

    def extra_anchor_loss(self):
        inherited = super().extra_anchor_loss()
        ids = torch.randint(len(self.route_flat_obs), (self.anchor_batch_size,), device=self.device)
        x, target = self.route_flat_obs[ids], self.route_flat_action[ids]
        reflected = torch.rand(len(ids), device=self.device) < .5
        x = torch.where(reflected[:, None], mirror_observation(x), x)
        target = torch.where(reflected[:, None], mirror_action(target), target)
        ids = torch.randint(len(self.deployed_obs), (self.anchor_batch_size,), device=self.device)
        steering_obs = torch.cat((x, self.deployed_obs[ids]))
        straight = straight_mask(steering_obs)
        direct = self.actor(TensorDict({'policy': steering_obs}, batch_size=[len(steering_obs)]))
        reflected = self.actor(TensorDict({'policy': mirror_observation(steering_obs)}, batch_size=[len(steering_obs)]))
        steering = self.route_steering_weight*masked_steering_equivariance(direct, reflected, straight)
        # Preserve smooth purposeful sagittal actions and mean drive. HipX
        # and wheel differentials are free; no left-turn teacher is imposed.
        preserve = .75*self.route_flat_multiplier*successful_support_loss(
            direct[:len(x)], target, tolerance=.08)
        self.route_flat_losses.append(preserve.detach())
        self.route_steering_losses.append(steering.detach())
        return inherited+preserve+steering

    def update(self):
        self.route_flat_losses.clear()
        self.route_steering_losses.clear()
        result = super().update()
        result.update(route_flat_preservation=torch.stack(self.route_flat_losses).mean() if self.route_flat_losses else torch.zeros((), device=self.device),
                      route_steering_equivariance=torch.stack(self.route_steering_losses).mean() if self.route_steering_losses else torch.zeros((), device=self.device),
                      route_steering_weight=self.route_steering_weight,
                      route_flat_multiplier=self.route_flat_multiplier)
        tensor_keys = [key for key, value in result.items() if isinstance(value, torch.Tensor)]
        if tensor_keys:
            values = torch.stack([result[key].detach().reshape(()) for key in tensor_keys]).cpu().tolist()
            result.update(zip(tensor_keys, values))
        return result

    def behavior_feedback(self, regressions):
        result = super().behavior_feedback(regressions)
        traction = [r for r in regressions if 'early-tread slip' in r or 'traction target' in r]
        route = [r for r in regressions if any(k in r for k in ('yaw', 'lateral', 'heading', 'route drift'))]
        flat_pose = [r for r in regressions if (r.startswith('flat/') or '/flat/' in r or 'flat workspace' in r)
            and any(k in r for k in ('pitch', 'reach', 'workspace', 'target_jump', 'support width',
                                    'speed traded', 'collapse', 'fr_inward', 'hipx angle'))]
        self.route_steering_weight = (min(.5, self.route_steering_weight*1.2) if route else
                                     max(self.route_base_steering_weight, self.route_steering_weight/1.2))
        self.route_flat_multiplier = (min(1.5, self.route_flat_multiplier*1.1) if flat_pose else
                                      max(1., self.route_flat_multiplier/1.1))
        self.early_traction_multiplier = (min(1.6, self.early_traction_multiplier*1.15) if traction else
                                          max(1., self.early_traction_multiplier/1.15))
        self._sync_traction_feedback()
        result.update(route_regressions=route, route_steering_weight=self.route_steering_weight,
                      route_flat_multiplier=self.route_flat_multiplier,
                      early_traction_multiplier=self.early_traction_multiplier)
        return result

    def save(self):
        result = super().save()
        result['m20_route_hold'] = dict(protocol=RUN_PROTOCOL, flat_replay_sha256=self.route_flat_sha256,
            traction_reference_sha256=self.traction_reference_sha256,
            steering_weight=self.route_steering_weight, flat_multiplier=self.route_flat_multiplier,
            early_traction_multiplier=self.early_traction_multiplier)
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        resume = load_cfg is None or load_cfg.get('optimizer')
        state = loaded_dict.get('m20_route_hold', {})
        if resume:
            if (state.get('protocol') != RUN_PROTOCOL
                    or state.get('flat_replay_sha256') != self.route_flat_sha256
                    or state.get('traction_reference_sha256') != self.traction_reference_sha256):
                raise ValueError('RouteHold needs an initial checkpoint and fresh optimizer for the new objective')
        result = super().load(loaded_dict, load_cfg, strict)
        if resume:
            self.route_steering_weight = float(state['steering_weight'])
            self.route_flat_multiplier = float(state['flat_multiplier'])
            self.early_traction_multiplier = float(state['early_traction_multiplier'])
            if not self.route_base_steering_weight <= self.route_steering_weight <= .5 or not 1 <= self.route_flat_multiplier <= 1.5:
                raise ValueError('Invalid restored RouteHold feedback')
            if not 1. <= self.early_traction_multiplier <= 1.6:
                raise ValueError('Invalid restored early-traction feedback')
            self._sync_traction_feedback()
        return result
