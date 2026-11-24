"""CPU-only rejection checks; no network or robot dependencies."""
import copy
import unittest
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
from export_native_actor import NAMES, replace_weights
from release_utils import replace_once


class ExportGuards(unittest.TestCase):
    def setUp(self):
        self.state = {'mlp.'+name: torch.ones(2, dtype=torch.float32) for name in NAMES}
        self.model = helper.make_model(helper.make_graph([], 'guard', [], [],
            [numpy_helper.from_array(np.ones(2, dtype=np.float32), name) for name in NAMES]))

    def test_verified_replacement_preserves_baseline(self):
        candidate = {name: tensor*2 for name, tensor in self.state.items()}
        updated = replace_weights(self.model, self.state, candidate)
        self.assertTrue(all(np.array_equal(numpy_helper.to_array(t), np.ones(2)*2)
                            for t in updated.graph.initializer))
        self.assertTrue(all(np.array_equal(numpy_helper.to_array(t), np.ones(2))
                            for t in self.model.graph.initializer))

    def test_wrong_baseline_rejected(self):
        baseline = copy.deepcopy(self.state)
        baseline['mlp.0.weight'][0] = 3
        with self.assertRaisesRegex(ValueError, 'Baseline checkpoint/ONNX'):
            replace_weights(self.model, baseline, self.state)

    def test_invalid_candidate_rejected(self):
        for value in (torch.tensor([float('nan'), 1.]), torch.ones(3), torch.ones(2, dtype=torch.float64)):
            candidate = dict(self.state, **{'mlp.6.bias': value})
            with self.assertRaisesRegex(ValueError, 'Invalid candidate'):
                replace_weights(self.model, self.state, candidate)

    def test_unknown_graph_initializer_rejected(self):
        self.model.graph.initializer.append(numpy_helper.from_array(np.zeros(1), 'unknown'))
        with self.assertRaisesRegex(ValueError, 'Unexpected baseline'):
            replace_weights(self.model, self.state, self.state)

    def test_unknown_or_ambiguous_patch_rejected(self):
        for text in ('missing', 'anchor anchor'):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                replace_once(text, 'anchor', 'changed')


if __name__ == '__main__':
    unittest.main()
