import unittest

import numpy as np

from emonet import EmoNet, EmoNetConfig, TORCH_AVAILABLE


class EmoNetSmokeTests(unittest.TestCase):
    def test_simulation_produces_branch_artifacts(self) -> None:
        model = EmoNet(EmoNetConfig(seed=7))
        base_stim = model.run_until_converged("URGENT! critical alert now!!!")

        self.assertEqual(base_stim.shape, (4,))
        self.assertGreaterEqual(len(model.state.branch_log), 1)
        self.assertGreater(sum(1 for record in model.state.branch_log if record.active_nodes), 0)

        pruned = model.prune_to_survivors()
        branches = model.extract_topk_branches()
        dominant = model.build_dominant_branch()
        tensor = model.dominant_branch_to_tensor(dominant)

        self.assertIsInstance(pruned, list)
        self.assertIsInstance(branches, list)
        self.assertGreaterEqual(len(pruned), 1)
        self.assertGreaterEqual(len(branches), 1)
        self.assertGreaterEqual(len(dominant), 1)
        self.assertEqual(tuple(tensor.shape), (len(dominant), 6))
        self.assertTrue(np.all(base_stim >= 0.0))
        self.assertTrue(np.all(base_stim <= 1.0))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_forward_runs_end_to_end_when_torch_available(self) -> None:
        model = EmoNet(EmoNetConfig(seed=11))
        outputs = model.forward("please summarize the issue quickly but politely")

        self.assertIn("stim_vec", outputs)
        self.assertIn("dominant_branch", outputs)
        self.assertIn("z", outputs)
        self.assertIn("s", outputs)
        self.assertEqual(tuple(outputs["stim_vec"].shape), (4,))
        self.assertEqual(tuple(outputs["z"].shape), (64,))
        self.assertEqual(tuple(outputs["s"].shape), (32,))


if __name__ == "__main__":
    unittest.main()
