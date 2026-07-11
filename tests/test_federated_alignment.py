import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from federated_main import (
    _add_central_dp_noise,
    _resolve_initial_rounds,
    update_shapley_values,
)
from options import args_parser


class RecordingShapleyCalculator:
    def __init__(self):
        self.current_global_model = None

    def compute_with_history(self, **kwargs):
        self.current_global_model = kwargs["current_global_model"]
        return [0.5]


class FailingShapleyCalculator:
    def compute_with_history(self, **kwargs):
        raise RuntimeError("synthetic failure")


class ValueShapleyCalculator:
    def __init__(self, value):
        self.value = value

    def compute_with_history(self, **kwargs):
        return [self.value]


class FederatedAlignmentTests(unittest.TestCase):
    def test_paper_operating_point_defaults(self):
        with patch.object(sys, "argv", ["test"]):
            args = args_parser()

        self.assertEqual(args.shapley_max_iter, 20)
        self.assertEqual(args.shapley_update_method, "mean")
        self.assertEqual(args.energy_budget, 5.0)
        self.assertEqual(args.dp_clip_ema, 0.8)
        self.assertEqual(args.dp_max_clip_norm, 1.0)

    def test_default_warmup_covers_all_clients_once(self):
        args = SimpleNamespace(initial_rounds=None, num_users=101, num_selected=5)

        self.assertEqual(_resolve_initial_rounds(args), 21)

    def test_noise_uses_fixed_inverse_selected_count(self):
        args = SimpleNamespace(
            dp_clip_scope="global",
            dp_clip_norm=1.0,
            _current_dp_clip_norm=1.0,
            _current_dp_noise_multiplier=0.0,
            dp_noise_multiplier=0.0,
            dp_channel_assisted=True,
            dp_channel_noise_multiplier=2.0,
            _current_selected_channel_gains=None,
            dp_channel_gain_cap=2.0,
            dp_channel_mode="channel_only",
        )
        state = {"weight": torch.zeros(2)}

        _, stats = _add_central_dp_noise(
            state,
            state,
            args,
            selected_count=5,
            aggregation_max_weight=0.3,
        )

        self.assertAlmostEqual(stats["channel_noise_std"], 0.4)
        self.assertAlmostEqual(stats["aggregation_max_weight"], 0.3)
        self.assertAlmostEqual(stats["noise_scale_weight"], 0.2)
        self.assertEqual(stats["noise_scaling"], "selected_count")

    def test_shapley_uses_pre_noise_full_coalition_model(self):
        calculator = RecordingShapleyCalculator()
        args = SimpleNamespace(
            use_shapley=True,
            shapley_estimator="complementary",
            shapley_allocation="neyman",
            shapley_update_method="mean",
            shapley_alpha=0.5,
            verbose=False,
        )
        round_data = {
            0: {
                "selected_clients": [0],
                "client_models": {0: "client-model"},
                "previous_global": "previous-global",
                "current_global": "pre-noise-global",
            }
        }
        shapley_values = np.zeros(1)
        participation_counts = np.ones(1)
        observation_counts = np.zeros(1, dtype=np.int64)

        update_shapley_values(
            args=args,
            epoch=0,
            shapley_values=shapley_values,
            shapley_calculator=calculator,
            round_client_models=round_data,
            val_data_loader=None,
            user_groups={0: list(range(10))},
            client_participation_counts=participation_counts,
            shapley_observation_counts=observation_counts,
        )

        self.assertEqual(calculator.current_global_model, "pre-noise-global")
        self.assertAlmostEqual(shapley_values[0], 0.5)
        self.assertEqual(observation_counts[0], 1)

    def test_shapley_failure_preserves_persistent_estimate(self):
        args = SimpleNamespace(
            use_shapley=True,
            shapley_estimator="complementary",
            shapley_allocation="neyman",
            shapley_update_method="mean",
            shapley_alpha=0.5,
            verbose=False,
        )
        round_data = {
            0: {
                "selected_clients": [0],
                "client_models": {0: "client-model"},
                "previous_global": "previous-global",
                "current_global": "pre-noise-global",
            }
        }
        shapley_values = np.array([0.25])
        participation_counts = np.ones(1)
        observation_counts = np.ones(1, dtype=np.int64)

        update_shapley_values(
            args=args,
            epoch=0,
            shapley_values=shapley_values,
            shapley_calculator=FailingShapleyCalculator(),
            round_client_models=round_data,
            val_data_loader=None,
            user_groups={0: list(range(10))},
            client_participation_counts=participation_counts,
            shapley_observation_counts=observation_counts,
        )

        self.assertAlmostEqual(shapley_values[0], 0.25)
        self.assertEqual(observation_counts[0], 1)

    def test_shapley_mean_uses_valid_observation_count(self):
        args = SimpleNamespace(
            use_shapley=True,
            shapley_estimator="complementary",
            shapley_allocation="neyman",
            shapley_update_method="mean",
            shapley_alpha=0.5,
            verbose=False,
        )
        round_data = {
            0: {
                "selected_clients": [0],
                "client_models": {0: "model-0"},
                "previous_global": "previous-0",
                "current_global": "current-0",
            },
            1: {
                "selected_clients": [0],
                "client_models": {0: "model-1"},
                "previous_global": "previous-1",
                "current_global": "current-1",
            },
        }
        shapley_values = np.zeros(1)
        participation_counts = np.array([2.0])
        observation_counts = np.zeros(1, dtype=np.int64)

        update_shapley_values(
            args, 0, shapley_values, ValueShapleyCalculator(0.4),
            round_data, None, {0: list(range(10))}, participation_counts,
            shapley_observation_counts=observation_counts,
        )
        update_shapley_values(
            args, 1, shapley_values, ValueShapleyCalculator(0.8),
            round_data, None, {0: list(range(10))}, participation_counts,
            shapley_observation_counts=observation_counts,
        )

        self.assertAlmostEqual(shapley_values[0], 0.6)
        self.assertEqual(observation_counts[0], 2)


if __name__ == "__main__":
    unittest.main()
