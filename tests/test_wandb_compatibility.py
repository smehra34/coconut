"""Compatibility checks for persistent W&B run IDs."""

import tempfile
import unittest

from run import get_wandb_run_id


class WandbCompatibilityTest(unittest.TestCase):
    def test_run_id_is_created_and_reused(self):
        with tempfile.TemporaryDirectory() as save_dir:
            first = get_wandb_run_id(save_dir)
            second = get_wandb_run_id(save_dir)

        self.assertTrue(first)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
