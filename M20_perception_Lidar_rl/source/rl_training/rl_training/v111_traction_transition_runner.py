"""Preserve successful leg timing while allowing traction to retune common drive."""
from pathlib import Path
from .v111_step_retention_runner import StepRetentionPPO
from .traction_transition import drive_band_loss, support_legs_loss


class TractionTransitionPPO(StepRetentionPPO):
    def __init__(self, *args, traction_reference_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.traction_reference_results = str(Path(traction_reference_results).resolve())

    def reference_loss(self, predicted, target, x, phase, scene):
        return self.anchor_weight*drive_band_loss(predicted, target, x[:, 6])

    def support_reference_loss(self, predicted, target):
        command = predicted.new_ones(len(predicted))
        return support_legs_loss(predicted, target)+.5*drive_band_loss(predicted, target, command)
