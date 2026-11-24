"""Retain successful stairs while releasing all defective flat teacher targets."""
from .v111_entry_transfer_runner import EntryTransferPPO


class FlatStartPPO(EntryTransferPPO):
    def __init__(self, *args, flat_reference_results, flat_reference_checkpoint, forward_reference_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_reference_results = flat_reference_results
        self.flat_reference_checkpoint = flat_reference_checkpoint
        self.forward_reference_results = forward_reference_results

    def behavior_feedback(self, regressions):
        # An unresolved flat start must not tighten imitation of a teacher
        # with the same defect. Genuine stair regressions retain the existing
        # bounded retention feedback and never stop training.
        stair = [r for r in regressions if not r.startswith('flat/')]
        return dict(super().behavior_feedback(stair),
                    flat_regressions=sum(r.startswith('flat/') for r in regressions))

    def save(self):
        state = super().save()
        state['m20_flat_start'] = dict(protocol='flat_start_v1',
            flat_reference_results=self.flat_reference_results,
            flat_reference_checkpoint=self.flat_reference_checkpoint,
            forward_reference_results=self.forward_reference_results)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('optimizer'):
            state = loaded_dict.get('m20_flat_start')
            if not state or state.get('protocol') != 'flat_start_v1':
                raise ValueError('FlatStart continuation requires a FlatStart checkpoint; use initial-checkpoint for old models')
        return super().load(loaded_dict, load_cfg, strict)

    def reference_loss(self, predicted, target, x, phase, scene):
        # All direct teacher-action imitation is restricted to stair scenes.
        # The successful B2+ replay and bounded local update loss remain live.
        keep = scene == 1
        if not keep.any():
            return predicted.sum()*0.
        after = getattr(self, '_reference_after_entry', None)
        if after is not None:
            self._reference_after_entry = after[keep]
        try:
            return super().reference_loss(predicted[keep], target[keep], x[keep], phase[keep], scene[keep])
        finally:
            if after is not None:
                self._reference_after_entry = after
