"""Isolated entry-only stage; the later four-leg gait retains its objectives."""
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import os
from pathlib import Path
from .v111_traction_transition_env_cfg import DeeproboticsM20V111TractionTransitionEnvCfg
from .v111_step_retention_env_cfg import DeeproboticsM20V111StepRetentionPPORunnerCfg, M20StepRetentionPPOCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_entry_transfer as entry


@configclass
class M20EntryTransferPPOCfg(M20StepRetentionPPOCfg):
    class_name: str = 'rl_training.v111_entry_transfer_runner:EntryTransferPPO'
    entry_reference_results: str = ''
    traction_reference_results: str = ''


@configclass
class M20FlatStartPPOCfg(M20EntryTransferPPOCfg):
    class_name: str = 'rl_training.v111_flat_start_runner:FlatStartPPO'
    flat_reference_results: str = ''
    flat_reference_checkpoint: str = ''
    forward_reference_results: str = ''


@configclass
class DeeproboticsM20V111EntryTransferEnvCfg(DeeproboticsM20V111TractionTransitionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Replace the height-only 40 ms ordering detector, the repeated lead
        # reward, and the action-delta cost opposing rapid purposeful lifts.
        # Traction/flat/noise/speed/later support terms retain their settings.
        self.rewards.v111_entry_simultaneous = None
        self.rewards.v111_entry_lead = None
        self.rewards.v111_entry_pitch = None
        self.rewards.v111_transition_action_impulse = None
        self.rewards.v111_entry_transfer_progress = RewTerm(func=entry.progress_reward, weight=1.)
        self.rewards.v111_entry_transfer_safety = RewTerm(func=entry.safety_cost, weight=1.)
        self.events.v111_refine_reset.func = entry.reset
        for key in ('eligible_episodes', 'clean_complete', 'early', 'twin', 'rear_early', 'credit_sum', 'cost_used'):
            setattr(self.curriculum, 'v111_entry_transfer_'+key,
                    CurrTerm(func=entry.metric, params={'key': key}))
        # Opt-in stage: old EntryTransfer experiments remain reproducible.
        if os.environ.get('M20_FLAT_START_REPAIR') == '1':
            from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_flat_start as flat
            self.commands.base_velocity.class_type = flat.FlatStartVelocityCommand
            self.commands.base_velocity.ranges.lin_vel_x = (-1., 1.)
            self.commands.base_velocity.ranges.lin_vel_y = (-.5, .5)
            self.commands.base_velocity.ranges.ang_vel_z = (-.6, .6)
            self.rewards.v111_flat_start_safety = RewTerm(func=flat.safety_cost, weight=1.)
            self.events.v111_refine_reset.func = flat.reset
            self.actions.joint_vel.class_type = flat.StoppedWheelVelocityAction
            self.observations.policy.actions.func = flat.executed_last_action
            self.observations.critic.actions.func = flat.executed_last_action
            for key in ('cost', 'lateral_cost', 'attitude_cost', 'impulse_cost', 'front_width',
                        'min_outward', 'hipx_abs', 'pitch_abs', 'pitch_rate_abs', 'transient'):
                setattr(self.curriculum, 'v111_flat_start_'+key,
                        CurrTerm(func=flat.metric, params={'key': key}))
            if os.environ.get('M20_ENTRY_DIRECT_REPAIR') == '1':
                # Direct 1 m/s first-riser samples are now a primary cohort.
                # Strengthen only the B1 sequence ledger; later stair gait
                # objectives and the successful replay remain unchanged.
                self.rewards.v111_entry_transfer_safety.weight = 1.5
                self.rewards.v111_entry_sequence = RewTerm(func=entry.sequence_reward, weight=.75)
                self.curriculum.v111_entry_transfer_sequence = CurrTerm(
                    func=entry.metric, params={'key': 'sequence'})
                self.curriculum.v111_entry_transfer_rear_hold = CurrTerm(
                    func=entry.metric, params={'key': 'rear_hold'})


@configclass
class DeeproboticsM20V111EntryTransferPPORunnerCfg(DeeproboticsM20V111StepRetentionPPORunnerCfg):
    experiment_name = 'M20-V11.1-EntryTransfer'
    algorithm = M20EntryTransferPPOCfg()

    def __post_init__(self):
        super().__post_init__()
        # Fresh optimizer for the changed objective. Never inherit the old
        # run's exhausted learning-rate/retention feedback state.
        self.algorithm.learning_rate = 1.e-5
        entry, traction = os.environ.get('M20_ENTRY_RESULTS', ''), os.environ.get('M20_TRACTION_RESULTS', '')
        if not entry or not Path(entry).is_file() or not traction or not Path(traction).is_file():
            raise ValueError('EntryTransfer requires matching physical entry and traction baselines')
        self.algorithm.entry_reference_results = str(Path(entry).resolve())
        self.algorithm.traction_reference_results = str(Path(traction).resolve())
        if os.environ.get('M20_FLAT_START_REPAIR') == '1':
            self.experiment_name = 'M20-V11.1-FlatStart'
            old = self.algorithm.to_dict()
            old.pop('class_name')
            self.algorithm = M20FlatStartPPOCfg(**old)
            for field, variable in (('flat_reference_results', 'M20_FLAT_RESULTS'),
                                    ('flat_reference_checkpoint', 'M20_FLAT_CHECKPOINT'),
                                    ('forward_reference_results', 'M20_FORWARD_RESULTS')):
                path = Path(os.environ.get(variable, ''))
                if not path.is_file():
                    raise ValueError('FlatStart requires '+variable)
                setattr(self.algorithm, field, str(path.resolve()))


@configclass
class DeeproboticsM20V111EntrySequenceEnvCfg(DeeproboticsM20V111EntryTransferEnvCfg):
    """B1-only sequence repair initialized from a physically good baseline.

    This branch removes the conflicting fixed-speed entry term.  Native
    velocity tracking remains active, while the sequence ledger supplies the
    ordered support signal and the rear wait is released by wheel distance.
    """

    v111_strict_entry_sequence: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Disable the 0.35 m/s approach-speed objective completely.  Keeping
        # the state metric is useful for diagnostics, but it is disconnected
        # from the reward graph in this task.
        self.rewards.v111_entry_overshoot_speed = None
        self.curriculum.v111_repair_entry_speed_cost = None
        self.v111_entry_flat.entry_speed_cost_scale = 0.
        # The new ledger owns B1 support/skip events. The historical detectors
        # remain diagnostics but must not pay several inconsistent B1 bonuses.
        self.rewards.v111_entry_support = None
        self.rewards.v111_entry_skip = None
        # The selected seed already has useful posture. Remove the preceding
        # stages' absolute flat-pose and traction shaping instead of trying to
        # reshape it again or rewarding four-wheel contact during a B1 swing.
        for name in ('v111_flat_workspace', 'v111_recovery_flat_quality',
                     'v111_transition_flat_quality', 'v111_transition_pitch',
                     'v111_traction_quality', 'v111_traction_cost',
                     'v111_flat_start_safety', 'v111_targeted_ascent_speed_floor'):
            setattr(self.rewards, name, None)
        for name in tuple(vars(self.curriculum)):
            if name.startswith(('v111_traction_', 'v111_flat_start_')):
                setattr(self.curriculum, name, None)
        self.rewards.v111_sequence_flat_motion_cost = RewTerm(func=entry.flat_motion_cost, weight=1.)
        self.rewards.v111_sequence_flat_motion_quality = RewTerm(func=entry.flat_motion_quality, weight=.75)
        for key in ('cost', 'quality', 'reach', 'yaw', 'lateral', 'speed'):
            setattr(self.curriculum, 'v111_sequence_flat_'+key,
                    CurrTerm(func=entry.flat_motion_metric, params={'key': key}))
        # Restore the native forward deficit now that no entry deceleration
        # term asks the policy to slow down.  This prevents a zero-speed local
        # optimum while retaining the regular progress and velocity rewards.
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_rear_posture
        self.rewards.world_forward_deficit.func = v111_rear_posture.v111_world_forward_deficit_with_descent_support
        self.rewards.world_forward_deficit.params = {'min_speed_fraction': .5, 'speed_scale': .3}
        self.rewards.track_lin_vel_xy_exp.func = v111_rear_posture.v111_track_lin_vel_with_descent_support
        self.rewards.world_forward_progress.func = v111_rear_posture.v111_world_forward_with_descent_support
        self.rewards.world_forward_progress.params = {'target_speed_fraction': .5}
        # Supplemental later-stair bonuses/costs must not entice the lead leg
        # to leave B1 while the other front leg is still transferring.
        for name in ('v111_adjacent_support', 'v111_adjacent_skip',
                     'v111_refine_landing', 'v111_recovery_swing',
                     'v111_recovery_progress', 'v111_retention_forward_route'):
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.params = {'reward_func': term.func, 'reward_params': dict(term.params)}
                term.func = entry.after_front_entry_reward
        mirror = self.rewards.joint_mirror
        mirror.params = {'reward_func': mirror.func, 'reward_params': dict(mirror.params)}
        mirror.func = entry.after_rear_approach_reward
        self.commands.base_velocity.class_type = entry.EntrySequenceVelocityCommand
        self.commands.base_velocity.ranges.lin_vel_x = (-1., 1.)
        self.commands.base_velocity.ranges.lin_vel_y = (-.5, .5)
        self.commands.base_velocity.ranges.ang_vel_z = (-.6, .6)
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_flat_start as flat
        self.actions.joint_vel.class_type = flat.StoppedWheelVelocityAction
        self.observations.policy.actions.func = flat.executed_last_action
        self.observations.critic.actions.func = flat.executed_last_action
        # Replace the loose event-only sequence credit with the strict
        # first-loaded -> second-loaded handoff ledger.
        self.rewards.v111_entry_sequence = RewTerm(func=entry.sequence_reward, weight=1.25)
        self.rewards.v111_entry_transfer_safety.weight = 1.25
        self.curriculum.v111_entry_transfer_sequence = CurrTerm(
            func=entry.metric, params={'key': 'sequence'})
        self.curriculum.v111_entry_transfer_rear_hold = CurrTerm(
            func=entry.metric, params={'key': 'rear_hold'})
        for key in ('front_cost_used', 'rear_cost_used', 'posture_cost_used',
                    'skip_cost_used', 'front_loss', 'skipped', 'valid_lift', 'sequence_sum',
                    'first_lift', 'first_support', 'second_legal_lift'):
            setattr(self.curriculum, 'v111_entry_transfer_'+key,
                    CurrTerm(func=entry.metric, params={'key': key}))


@configclass
class M20EntrySequencePPOCfg(M20StepRetentionPPOCfg):
    class_name: str = 'rl_training.v111_entry_sequence_runner:EntrySequencePPO'
    entry_reference_results: str = ''
    flat_reference_results: str = ''
    flat_reference_checkpoint: str = ''
    forward_reference_results: str = ''
    learning_rate: float = 1.e-5
    success_anchor_weight: float = 1.
    local_anchor_weight: float = .5


@configclass
class DeeproboticsM20V111EntrySequencePPORunnerCfg(DeeproboticsM20V111StepRetentionPPORunnerCfg):
    experiment_name = 'M20-V11.1-EntrySequenceRepair'
    algorithm = M20EntrySequencePPOCfg()
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.e-5
        path = Path(os.environ.get('M20_ENTRY_RESULTS', ''))
        if not path.is_file():
            raise ValueError('EntrySequence requires measured M20_ENTRY_RESULTS from the new seed')
        self.algorithm.entry_reference_results = str(path.resolve())
        for field, variable in (('flat_reference_results', 'M20_FLAT_RESULTS'),
                                ('flat_reference_checkpoint', 'M20_RECOVERY_TEACHER'),
                                ('forward_reference_results', 'M20_FORWARD_RESULTS')):
            path = Path(os.environ.get(variable, ''))
            if not path.is_file():
                raise ValueError('EntrySequence requires '+variable)
            setattr(self.algorithm, field, str(path.resolve()))
