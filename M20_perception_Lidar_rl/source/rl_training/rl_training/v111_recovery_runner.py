"""Reference-anchored PPO without editing the installed RSL-RL implementation."""
import copy
import hashlib
import math
from itertools import chain
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from .v11_continuous_runner import V11ContinuousRunner
from .recovery_reference import load_labeled_replay, reference_anchor_loss, mean_reversal_loss, valid_quiet_triplets


class RecoveryPPO(PPO):
    def __init__(self, *args, teacher_checkpoint, replay_path, anchor_weight,
                 anchor_front_stair_weight, anchor_batch_size, anchor_flat_leg_weight=.04,
                 anchor_flat_wheel_differential_weight=.08, anchor_ascent_other_weight=.35,
                 anchor_tolerance=.05, anchor_physical_ascent=False,
                 anchor_ascent_wheel_differential_weight=1., quiet_mean_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        if not Path(teacher_checkpoint).is_file() or not Path(replay_path).is_file():
            raise ValueError('Recovery requires an explicit frozen teacher and deployment replay')
        if (not math.isfinite(anchor_weight) or anchor_weight <= 0
            or not 0 < anchor_front_stair_weight <= 1 or anchor_batch_size < 1):
            raise ValueError('Invalid reference anchor parameters')
        if (any(not math.isfinite(v) or not 0 < v <= 1 for v in (
            anchor_flat_leg_weight, anchor_flat_wheel_differential_weight, anchor_ascent_other_weight,
            anchor_ascent_wheel_differential_weight))
            or not math.isfinite(anchor_tolerance) or not 0 <= anchor_tolerance <= .5):
            raise ValueError('Invalid scene-specific reference anchor parameters')
        self.reference_sha256 = hashlib.sha256(Path(teacher_checkpoint).read_bytes()).hexdigest()
        self.replay_sha256 = hashlib.sha256(Path(replay_path).read_bytes()).hexdigest()
        checkpoint = torch.load(teacher_checkpoint, map_location=self.device, weights_only=True)
        self.teacher = copy.deepcopy(self.actor)
        self.teacher.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        self.teacher.requires_grad_(False).eval()
        obs, scene, phase, case_id = load_labeled_replay(replay_path, self.reference_sha256)
        self.replay = torch.from_numpy(obs).to(self.device)
        self.replay_scene = torch.from_numpy(scene).to(self.device)
        self.replay_phase = torch.from_numpy(phase).to(self.device)
        self.replay_case = torch.from_numpy(case_id).to(self.device)
        # Balanced physical cases and policy phases prevent long flat rollouts
        # from drowning out ascent retention or the C-to-motion transition.
        groups = np.unique(np.stack((case_id, phase), axis=1), axis=0)
        members = [np.flatnonzero((case_id == case) & (phase == ph)) for case, ph in groups]
        sizes = np.asarray([len(group) for group in members], dtype=np.int64)
        self.replay_group_members = torch.from_numpy(np.concatenate(members)).to(self.device)
        self.replay_group_sizes = torch.from_numpy(sizes).to(self.device)
        self.replay_group_starts = torch.from_numpy(np.concatenate(([0], np.cumsum(sizes)[:-1]))).to(self.device)
        self.anchor_weight = anchor_weight
        self.front_weight = anchor_front_stair_weight
        self.anchor_batch_size = anchor_batch_size
        if not math.isfinite(quiet_mean_weight) or not 0 <= quiet_mean_weight <= .02:
            raise ValueError('Mean-action quietness regularizer exceeds its .02 budget')
        self.quiet_mean_weight = quiet_mean_weight
        self.quiet_ramp_provider = lambda: 0.
        self.quiet_triplets = None
        self.quiet_losses = []
        self.anchor_options = dict(flat_leg_weight=anchor_flat_leg_weight,
            flat_wheel_differential_weight=anchor_flat_wheel_differential_weight,
            ascent_other_weight=anchor_ascent_other_weight, tolerance=anchor_tolerance,
            physical_ascent=anchor_physical_ascent,
            ascent_wheel_differential_weight=anchor_ascent_wheel_differential_weight)
        # Keep the early representation; later features and decoder can learn
        # the new landing policy, with the frozen reference protecting drive.
        for name, parameter in self.actor.named_parameters():
            if name.startswith(('mlp.0.', 'mlp.2.', 'distribution.')):
                parameter.requires_grad_(False)
        self.anchor_losses = []
        self._anchor_hook = self.optimizer.register_step_pre_hook(self._anchor_step)

    def _anchor_step(self, optimizer, args, kwargs):
        choices = torch.randint(len(self.replay_group_sizes), (self.anchor_batch_size,), device=self.device)
        offsets = (torch.rand(self.anchor_batch_size, device=self.device) * self.replay_group_sizes[choices]).long()
        indices = self.replay_group_members[self.replay_group_starts[choices] + offsets]
        x = self.replay[indices]
        obs = TensorDict({'policy': x}, batch_size=[len(x)])
        with torch.no_grad():
            target = self.teacher(obs)
        with torch.enable_grad():
            predicted = self.actor(obs)
            loss = reference_anchor_loss(predicted, target, x, self.replay_scene[indices],
                self.front_weight, self.anchor_weight, **self.anchor_options)
            quiet = self._mean_quietness()
            (loss+quiet).backward()
        torch.nn.utils.clip_grad_norm_([p for p in chain(self.actor.parameters(), self.critic.parameters())
            if p.requires_grad], self.max_grad_norm)
        self.anchor_losses.append(loss.detach())
        self.quiet_losses.append(quiet.detach())

    def _mean_quietness(self):
        ramp = self.quiet_ramp_provider()
        if not self.quiet_mean_weight or not ramp or self.quiet_triplets is None or not len(self.quiet_triplets):
            return torch.zeros((), device=self.device)
        selected = torch.randint(len(self.quiet_triplets), (self.anchor_batch_size,), device=self.device)
        time, env = self.quiet_triplets[selected].unbind(1)
        obs = self.storage.observations['policy']
        triplet = torch.stack((obs[time-2, env], obs[time-1, env], obs[time, env]))
        x = triplet.flatten(0, 1)
        means = self.actor(TensorDict({'policy': x}, batch_size=[len(x)])).reshape(3, -1, 16)
        stair = (triplet[..., [57, 60]] > .5).any(-1).any(0)
        return self.quiet_mean_weight*ramp*mean_reversal_loss(means, stair)

    def update(self):
        self.anchor_losses.clear()
        self.quiet_losses.clear()
        if self.quiet_mean_weight:
            self.quiet_triplets = valid_quiet_triplets(self.storage.dones)
        losses = super().update()
        losses['reference_anchor'] = torch.stack(self.anchor_losses).mean() if self.anchor_losses else torch.zeros((), device=self.device)
        if self.quiet_mean_weight:
            losses['quiet_mean_reversal'] = torch.stack(self.quiet_losses).mean() if self.quiet_losses else torch.zeros((), device=self.device)
            losses['quiet_ramp'] = self.quiet_ramp_provider()
        self.quiet_triplets = None
        return losses


class RecoveryRunner(V11ContinuousRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.env.unwrapped.cfg, 'v111_refine'):
            def ramp():
                env = self.env.unwrapped
                stage = int(env.common_step_counter)-getattr(env, '_m20_refine_start_step', int(env.common_step_counter))
                return max(0., min(1., (stage/96-200)/1000))
            self.alg.quiet_ramp_provider = ramp

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos['m20_common_step_counter'] = int(self.env.unwrapped.common_step_counter)
        if hasattr(self.env.unwrapped, '_m20_refine_start_step'):
            infos['m20_refine_stage_steps'] = (infos['m20_common_step_counter']
                                               - self.env.unwrapped._m20_refine_start_step)
        infos['reference_sha256'] = self.alg.reference_sha256
        infos['replay_sha256'] = self.alg.replay_sha256
        super().save(path, infos)
