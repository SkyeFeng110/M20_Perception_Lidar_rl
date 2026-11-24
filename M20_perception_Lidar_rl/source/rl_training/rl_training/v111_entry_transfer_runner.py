"""Release front-entry imitation, retain rear support and the later gait."""
import json
from pathlib import Path
import numpy as np
import torch
from .v111_traction_transition_runner import TractionTransitionPPO
from .traction_transition import drive_band_loss


class EntryTransferPPO(TractionTransitionPPO):
    def __init__(self, *args, entry_reference_results, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_reference_results = str(Path(entry_reference_results).resolve())
        path = Path(kwargs['replay_path'])
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata.get('entry_mask_protocol') != 'measured_bilateral_b1_plus_120ms_v1':
            raise ValueError('EntryTransfer requires physical post-B1 replay masks')
        with np.load(path, allow_pickle=False) as data:
            mask = data['post_entry']
            if mask.dtype != np.bool_ or mask.shape != (len(self.replay),):
                raise ValueError('Invalid post-entry replay mask')
            after = torch.from_numpy(mask).to(self.device)
        self.replay_post_entry = after
        groups = [members[after[members]] for members in self.success_members]
        if any(not len(members) for members in groups):
            raise ValueError('Every successful ascent needs post-entry protection')
        self.success_members = groups
        self.post_entry_samples = sum(len(members) for members in groups)

    def reference_batch_context(self, indices):
        self._reference_after_entry = self.replay_post_entry[indices]

    def reference_loss(self, predicted, target, x, phase, scene):
        # Front B1 targets are intentionally absent from the generic loss.
        # Rear sagittal targets preserve the existing support phase during
        # entry. They do not freeze a leg or prescribe a new four-leg cycle.
        moving_ascent = (scene == 1) & (phase == 2)
        error = ((predicted[:, [7, 8, 10, 11]]-target[:, [7, 8, 10, 11]].detach()).abs()-.08).clamp_min(0).square().mean(1)
        rear = (error*moving_ascent).sum()/moving_ascent.sum().clamp_min(1)
        protect_drive = (scene == 0) | getattr(self, '_reference_after_entry', torch.ones(len(scene), dtype=torch.bool, device=scene.device))
        # The defective entry includes delayed front-wheel drive followed by
        # a large burst. Do not retain that burst through a common-drive band.
        drive = (drive_band_loss(predicted[protect_drive], target[protect_drive], x[protect_drive, 6])
                 if protect_drive.any() else predicted.sum()*0.)
        return self.anchor_weight*drive+.25*rear
