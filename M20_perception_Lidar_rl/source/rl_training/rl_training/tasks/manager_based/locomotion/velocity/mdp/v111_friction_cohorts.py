"""Startup-only coherent wheel materials, using Isaac Lab's shape indexing."""
import torch
from isaaclab.envs.mdp.events import randomize_rigid_body_material


class WheelFrictionCohorts(randomize_rigid_body_material):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = len(self.material_buckets)
        if n < 2 or n % 2:
            raise ValueError('Wheel friction cohorts need an even number of material buckets')
        half = n//2
        # 40% low-mu, 40% nominal coherent ground, 20% inherited independent
        # per-shape randomization. All four wheels share a sampled material
        # in a coherent case; other robot body materials are never replaced.
        self.material_buckets[:half, 0].uniform_(.45, .70)
        self.material_buckets[half:, 0].uniform_(.85, 1.20)
        self.material_buckets[:, 1] = self.material_buckets[:, 0]
        self.material_buckets[:, 2] = 0.
        self._half = half

    def __call__(self, env, env_ids, static_friction_range, dynamic_friction_range,
                 restitution_range, num_buckets, asset_cfg, make_consistent=False):
        ids = torch.arange(env.scene.num_envs, device='cpu') if env_ids is None else env_ids.cpu()
        selected = ids[ids.remainder(10) < 8]
        low = selected.remainder(10) < 4
        bucket = torch.randint(self._half, (len(selected),), device='cpu')+torch.where(low, 0, self._half)
        materials = self.asset.root_physx_view.get_material_properties()
        samples = self.material_buckets[bucket, None, :]
        if self.num_shapes_per_body is None:
            raise ValueError('Wheel cohort event must select the four wheel bodies explicitly')
        for body in self.asset_cfg.body_ids:
            start = sum(self.num_shapes_per_body[:body])
            end = start+self.num_shapes_per_body[body]
            materials[selected, start:end] = samples.expand(-1, end-start, -1)
        self.asset.root_physx_view.set_material_properties(materials, selected)
        if not hasattr(env, '_m20_v2_low_mu'):
            env._m20_v2_low_mu = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)
        env._m20_v2_low_mu[ids.to(env.device)] = False
        env._m20_v2_low_mu[selected[low].to(env.device)] = True
