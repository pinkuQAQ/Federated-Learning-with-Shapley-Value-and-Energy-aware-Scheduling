import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selection import softmax_score_selection


class SoftmaxScoreSelectionTests(unittest.TestCase):
    def test_samples_unique_available_clients(self):
        selected = softmax_score_selection(
            scores=np.array([0.0, 1.0, 2.0, 3.0]),
            num_selected=2,
            temperature=1.0,
            available_clients=[1, 2, 3],
            rng=np.random.RandomState(7),
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(set(selected).issubset({1, 2, 3}))

    def test_is_reproducible_with_fixed_rng(self):
        kwargs = {
            "scores": np.array([0.2, 0.4, 0.6, 0.8]),
            "num_selected": 3,
            "temperature": 0.7,
        }

        first = softmax_score_selection(rng=np.random.RandomState(42), **kwargs)
        second = softmax_score_selection(rng=np.random.RandomState(42), **kwargs)

        self.assertEqual(first, second)

    def test_small_temperature_concentrates_on_highest_score(self):
        selected = softmax_score_selection(
            scores=np.array([0.0, 10.0]),
            num_selected=1,
            temperature=0.01,
            rng=np.random.RandomState(1),
        )

        self.assertEqual(selected, [1])

    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            softmax_score_selection(np.array([0.0, 1.0]), 1, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
