import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedmsv import FedMSVSelector, aggregate_client_models


def scalar_state(value):
    return {"weight": torch.tensor([float(value)], dtype=torch.float32)}


class FedMSVTests(unittest.TestCase):
    def test_weighted_sampling_is_unique_and_reproducible(self):
        first = FedMSVSelector(4, 3, seed=7)
        second = FedMSVSelector(4, 3, seed=7)
        first.cumulative_msv[:] = [1.0, -2.0, 0.0, -1.0]
        second.cumulative_msv[:] = first.cumulative_msv
        first._refresh_sampling_weights()
        second._refresh_sampling_weights()

        selected_first = first.select([0, 1, 2, 3])
        selected_second = second.select([0, 1, 2, 3])

        self.assertEqual(selected_first, selected_second)
        self.assertEqual(len(selected_first), len(set(selected_first)))
        self.assertLess(first.raw_sampling_weights[1], first.raw_sampling_weights[0])

    def test_sign_msv_penalizes_consistently_harmful_client(self):
        selector = FedMSVSelector(
            num_clients=2,
            sample_size=2,
            guided_prefix=1,
            epsilon_a=0.0,
            epsilon_b=0.0,
            epsilon_c=0.1,
            seed=3,
        )
        client_models = {0: scalar_state(2.0), 1: scalar_state(-1.0)}
        current = aggregate_client_models(client_models, [0, 1], {0: 1, 1: 1})

        record = selector.update_from_round(
            selected_clients=[0, 1],
            previous_global=scalar_state(0.0),
            current_global=current,
            client_models=client_models,
            client_data_sizes={0: 1, 1: 1},
            utility_fn=lambda state: float(state["weight"].item()),
            round_index=0,
        )

        np.testing.assert_allclose(record["round_msv"], [1.0, -1.0])
        self.assertGreater(selector.raw_sampling_weights[0], selector.raw_sampling_weights[1])
        self.assertEqual(record["permutation_count"], 2)

    def test_round_truncation_skips_small_global_change(self):
        selector = FedMSVSelector(
            num_clients=2,
            sample_size=2,
            guided_prefix=1,
            epsilon_a=0.01,
            epsilon_b=0.01,
            epsilon_c=0.1,
            seed=5,
        )
        record = selector.update_from_round(
            selected_clients=[0, 1],
            previous_global=scalar_state(0.0),
            current_global=scalar_state(0.005),
            client_models={0: scalar_state(0.01), 1: scalar_state(0.0)},
            client_data_sizes={0: 1, 1: 1},
            utility_fn=lambda state: float(state["weight"].item()),
            round_index=0,
        )

        self.assertTrue(record["round_skipped"])
        self.assertEqual(record["permutation_count"], 0)
        self.assertEqual(record["utility_evaluations"], 2)
        np.testing.assert_allclose(selector.cumulative_msv, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
