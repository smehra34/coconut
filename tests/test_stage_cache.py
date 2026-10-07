"""Equivalence checks for deterministic curriculum-stage caching."""

import tempfile
import unittest
from types import SimpleNamespace

from datasets import Dataset

from dataset import get_cot_latent_dataset, get_question_latent_dataset


class StageCacheTest(unittest.TestCase):
    def setUp(self):
        self.base = Dataset.from_dict(
            {
                "question_tokenized": [[1, 2], [3], [4, 5, 6]],
                "steps_tokenized": [[[7], [8, 9]], [[10]], [[11], [12], [13]]],
                "answer_tokenized": [[14, 0], [15, 0], [16, 0]],
                "idx": [0, 1, 2],
            }
        )
        self.config = SimpleNamespace(
            pad_latent_to_max=True,
            max_latent_stage=3,
            c_thought=1,
            uniform_prob=0.0,
            no_cot=False,
        )

    def test_cot_latent_cache_matches_uncached(self):
        expected = get_cot_latent_dataset(
            2, self.base, self.config, 20, 21, 22
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            first = get_cot_latent_dataset(
                2, self.base, self.config, 20, 21, 22, cache_dir=cache_dir
            )
            second = get_cot_latent_dataset(
                2, self.base, self.config, 20, 21, 22, cache_dir=cache_dir
            )
            self.assertEqual(first.to_dict(), expected.to_dict())
            self.assertEqual(second.to_dict(), expected.to_dict())

    def test_question_cache_matches_uncached(self):
        expected = get_question_latent_dataset(
            2, self.base, self.config, 20, 21, 22
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            actual = get_question_latent_dataset(
                2, self.base, self.config, 20, 21, 22, cache_dir=cache_dir
            )
            self.assertEqual(actual.to_dict(), expected.to_dict())


if __name__ == "__main__":
    unittest.main()
