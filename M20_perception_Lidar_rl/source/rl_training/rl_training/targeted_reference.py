"""Native-ABI reflection and narrowly scoped takeover correction objectives.

Reflection is equivariance, not equality between the legs of an asymmetric
gait. BHD slots describe successive forward edges and are unchanged by y -> -y.
Units below are the saved contract: omega*.25, joint velocity*.05, q in rad.
"""
import math
import torch

JOINT_MIRROR = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14)
HIP = (0, 3, 6, 9)


def validate_execution_contract(nominal, action_scale, observation_scales):
    expected_nominal = [0., -.3, .6]*2+[0., .3, -.6]*2+[0.]*4
    expected_scale = [.125, .25, .25]*4+[5.]*4
    if (list(nominal) != expected_nominal or list(action_scale) != expected_scale
        or observation_scales != dict(base_ang_vel=.25, projected_gravity=1., velocity_commands=1.,
            joint_pos=1., joint_vel=.05, actions=1., height_scan=1.)):
        raise ValueError('Targeted correction requires the verified native nominal/action/observation scales')


def mirror_action(action):
    result = action[..., list(JOINT_MIRROR)].clone()
    result[..., list(HIP)] *= -1
    return result


def mirror_observation(obs):
    result = obs.clone()
    # Angular velocity is axial; gravity and velocity command are polar.
    result[..., [0, 2, 4, 7, 8]] *= -1
    for start in (9, 25, 41):
        result[..., start:start+16] = mirror_action(obs[..., start:start+16])
    return result


def straight_mask(obs):
    return (obs[:, 7].abs() < .05) & (obs[:, 8].abs() < .03)


def settled_idle_mask(obs, phase):
    # Missing BHD alone is never used as a physical flat-terrain label.
    # This is a local, near-nominal, quiet recorded takeover state only.
    return ((phase == 1) & (obs[:, 6:9].abs().amax(1) < 1.e-4)
        & (-obs[:, 5] > .998) & (obs[:, :3].abs().amax(1) < .04)
        & (obs[:, 9:21].abs().amax(1) < .18)
        & (obs[:, 25:37].abs().amax(1) < .025)
        & (obs[:, 57:63].abs().amax(1) < 1.e-4))


def correction_scales(obs, phase):
    legs = torch.ones_like(obs[:, :12])
    # Release defective ab/adduction bias, keep all other retention weights.
    legs[:, list(HIP)] = torch.where(straight_mask(obs)[:, None], .10, 1.)
    wheels = torch.where(settled_idle_mask(obs, phase)[:, None], .08, 1.)
    return legs, wheels


def equivariance_loss(action, mirrored_action):
    error = (action-mirror_action(mirrored_action)).square()
    return (error/(1+error)).mean()


def masked_equivariance_loss(action, mirrored_action, mask):
    """Equivariance over a fixed batch, without GPU-dependent indexing."""
    error = (action-mirror_action(mirrored_action)).square()
    per_sample = (error/(1+error)).mean(1)
    weights = mask.to(per_sample.dtype)
    return (per_sample*weights).sum()/weights.sum().clamp_min(1.)


def idle_wheel_loss(action, mask):
    # No leg-amplitude cost or moving common-drive cost is introduced.
    energy = action[:, 12:].square()
    per_sample = (energy/(1+energy)).mean(1)
    return (per_sample*mask).sum()/mask.sum().clamp_min(1)


def commanded_reach(action):
    """Exact body-x wheel-from-hipy FK for the saved two .25 m links.

    Hipx rotation leaves body-x unchanged. Both sagittal axes are (0,-1,0).
    Use only the fixed saved joint nominal/scale contract, verified in preflight.
    """
    hip = action[:, [1, 4, 7, 10]]*.25+action.new_tensor([-.3, -.3, .3, .3])
    knee = action[:, [2, 5, 8, 11]]*.25+action.new_tensor([.6, .6, -.6, -.6])
    return .25*(hip.sin()+(hip+knee).sin())


def flat_workspace_loss(action, obs, scene):
    mask = (scene == 0) & straight_mask(obs) & (obs[:, 6] > .1)
    # A free workspace instead of one joint pose, with no wheel-drive cost.
    energy = ((commanded_reach(action).abs()-.09).clamp_min(0)/.12).square()
    per_sample = (energy/(1+energy)).mean(1)
    return (per_sample*mask).sum()/mask.sum().clamp_min(1)


def coherent_triplets(case, phase):
    """Contiguous physical replay only; exclude case and phase boundaries."""
    valid = (case[2:] == case[1:-1]) & (case[1:-1] == case[:-2])
    valid &= (phase[2:] == phase[1:-1]) & (phase[1:-1] == phase[:-2])
    return valid.nonzero(as_tuple=True)[0]+2


def settlement_step(target, peak, expected, alignment, height_error, velocity,
                    force, active, reset, *, airborne_weight=.5, force_threshold=5.):
    """Finite descending/contact credit for each monotonically advancing tread.

    Credit cannot be consumed during upward lift without contact. The configured
    fraction is available in descent; actual loading reserves the rest. Reset,
    closed command, holding and revisiting never pay catch-up credit.
    """
    if (not math.isfinite(airborne_weight)
        or not 0. <= float(airborne_weight) <= 1.):
        raise ValueError('airborne_weight must be finite and within [0, 1]')
    if not math.isfinite(force_threshold) or float(force_threshold) < 0.:
        raise ValueError('force_threshold must be finite and non-negative')
    descending = ((-velocity-.02)/.20).clamp(0, 1)
    close = torch.exp(-(height_error/.08).square())
    # Airborne descent can provide only a bounded early hint. The remaining
    # credit is earned by real vertical loading once the wheel is near the
    # tread plane; this prevents a swing-through from paying the full landing
    # budget while preserving the old V1 defaults.
    loading = ((force-force_threshold)/30.).clamp(0, 1)*(height_error.abs() <= .035)
    score = alignment*close*(airborne_weight*descending
        +(1-airborne_weight*descending)*loading)
    # Changing targets begins a new finite ledger, but inactive observations
    # consume current credit. The physical target index never moves backward.
    prior = torch.where(target == expected, peak, 0.)
    prior = torch.where(reset[:, None], score, prior)
    credit = (score-prior).clamp_min(0)*active*~reset[:, None]
    return expected.clone(), torch.maximum(prior, score), credit


def ascent_speed_credit(path_speed, command_x, active, target_speed_fraction=.8):
    """Bounded opportunity bonus: forward travel pays, standing still pays zero."""
    if not math.isfinite(target_speed_fraction) or not 0. < target_speed_fraction <= 1.:
        raise ValueError('target_speed_fraction must be in (0, 1]')
    target = (command_x*target_speed_fraction).clamp_min(.001)
    return (path_speed/target).clamp(0, 1)*active*(command_x > .1)
