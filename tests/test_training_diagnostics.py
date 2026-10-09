import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import (
    find_nonfinite_gradients,
    get_expected_visible_cot,
    measure_and_clip_grad_norm,
)


class GradientDiagnosticsTest(unittest.TestCase):
    def _model_with_gradient(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[5.0]])
        return model

    def test_missing_threshold_measures_without_clipping(self):
        model = self._model_with_gradient()
        norm, clipping_enabled = measure_and_clip_grad_norm(model, "ddp", None)

        self.assertFalse(clipping_enabled)
        torch.testing.assert_close(norm, torch.tensor(5.0))
        torch.testing.assert_close(model.weight.grad, torch.tensor([[5.0]]))

    def test_positive_threshold_clips_and_returns_preclip_norm(self):
        model = self._model_with_gradient()
        norm, clipping_enabled = measure_and_clip_grad_norm(model, "ddp", 1.0)

        self.assertTrue(clipping_enabled)
        torch.testing.assert_close(norm, torch.tensor(5.0))
        self.assertLessEqual(model.weight.grad.norm().item(), 1.0)

    def test_nonpositive_threshold_preserves_original_behavior(self):
        model = self._model_with_gradient()
        _, clipping_enabled = measure_and_clip_grad_norm(model, "ddp", 0.0)

        self.assertFalse(clipping_enabled)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[5.0]]))

    def test_nonfinite_scan_preserves_and_identifies_original_gradient(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[float("nan"), 3.0]])

        findings = find_nonfinite_gradients(model)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["name"], "weight")
        self.assertEqual(findings[0]["nan"], 1)
        self.assertEqual(findings[0]["finite_abs_max"], 3.0)
        self.assertTrue(torch.isnan(model.weight.grad[0, 0]))
        self.assertEqual(model.weight.grad[0, 1].item(), 3.0)


class ValidationMetricsTest(unittest.TestCase):
    def test_visible_cot_tracks_curriculum_suffix(self):
        configs = SimpleNamespace(cot=False, no_cot=False, max_latent_stage=3)
        steps = ["first", "second", "third"]

        self.assertEqual(get_expected_visible_cot(steps, 0, configs), "first\nsecond\nthird")
        self.assertEqual(get_expected_visible_cot(steps, 2, configs), "third")
        self.assertEqual(get_expected_visible_cot(steps, 4, configs), "")

    def test_cot_and_no_cot_controls(self):
        steps = ["first", "second"]
        cot_configs = SimpleNamespace(cot=True, no_cot=False, max_latent_stage=3)
        no_cot_configs = SimpleNamespace(cot=False, no_cot=True, max_latent_stage=3)

        self.assertEqual(get_expected_visible_cot(steps, 2, cot_configs), "first\nsecond")
        self.assertEqual(get_expected_visible_cot(steps, 0, no_cot_configs), "")


if __name__ == "__main__":
    unittest.main()
