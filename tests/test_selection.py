import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selection import (OortSelector, build_oort_client_priors,
                       channel_feasible_clients, online_exploration_selection,
                       softmax_score_selection)


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


class OnlineColdStartSelectionTests(unittest.TestCase):
    def test_first_round_uses_non_shapley_scores(self):
        selected = online_exploration_selection(
            scores=np.array([0.0, 1.0, 10.0]),
            observation_counts=np.zeros(3), num_selected=1,
            exploration_slots=1, temperature=0.01, rng=np.random.RandomState(3))
        self.assertEqual(selected, [2])

    def test_reserves_one_slot_for_unobserved_client(self):
        selected = online_exploration_selection(
            scores=np.array([10.0, 1.0, 2.0, 3.0]),
            observation_counts=np.array([1, 0, 0, 0]), num_selected=2,
            exploration_slots=1, temperature=0.01, rng=np.random.RandomState(3))
        self.assertEqual(set(selected), {0, 3})


class ChannelFeasibilityTests(unittest.TestCase):
    def test_quantile_filter_removes_poor_channels(self):
        selected, metadata = channel_feasible_clients(
            np.array([0.1, 0.2, 1.0, 2.0]), num_selected=2, drop_quantile=0.5)
        self.assertEqual(set(selected), {2, 3})
        self.assertEqual(metadata['candidate_count_after'], 2)

    def test_fallback_restores_best_channels(self):
        selected, metadata = channel_feasible_clients(
            np.array([0.1, 0.2, 1.0, 2.0]), num_selected=2, min_gain=10.0)
        self.assertEqual(set(selected), {2, 3})
        self.assertEqual(metadata['fallback_count'], 2)


class OortSelectorTests(unittest.TestCase):
    def _selector(self, **overrides):
        kwargs = {
            'num_clients': 4,
            'sample_size': 2,
            'initial_rewards': np.ones(4),
            'initial_durations': np.ones(4),
            'seed': 7,
        }
        kwargs.update(overrides)
        return OortSelector(**kwargs)

    def test_temporal_uncertainty_rewards_older_client(self):
        selector = self._selector()
        selector.round = 10
        selector.explored[:] = True
        selector.statistical_utility[:] = 2.0
        selector.last_selected_round[:] = np.array([1, 9, 9, 9])
        utilities = selector._client_utilities([0, 1])
        expected_gap = np.sqrt(0.1 * np.log(10)) - np.sqrt(0.1 * np.log(10) / 9)
        self.assertGreater(utilities[0], utilities[1])
        self.assertAlmostEqual(utilities[0] - utilities[1], expected_gap)

    def test_blacklist_uses_total_count_and_caps_fraction(self):
        selector = self._selector(blacklist_rounds=10, blacklist_max_fraction=0.25)
        selector.participation_counts[:] = np.array([20, 15, 11, 1])
        selector._refresh_blacklist()
        self.assertEqual(np.flatnonzero(selector.blacklisted).tolist(), [0])
        self.assertEqual(selector.participation_counts.tolist(), [20, 15, 11, 1])

    def test_unexplored_sampling_uses_registered_reward_window(self):
        selector = self._selector(
            sample_size=1,
            epsilon=1.0,
            sample_window=1.0,
            initial_rewards=np.array([1.0, 10.0, 2.0, 3.0]),
        )
        self.assertEqual(selector.select(), [1])

    def test_flat_exploitation_utility_relaxes_pacer(self):
        selector = self._selector(pacer_window=2, pacer_delta=5.0, round_threshold=10.0)
        selector.round = 4
        selector.exploit_utility_history = [1.0, 1.0, 1.0, 1.0]
        selector._update_pacer()
        self.assertEqual(selector.round_threshold, 15.0)

    def test_exploration_stops_after_all_clients_are_observed(self):
        selector = self._selector(epsilon=0.2)
        selector.explored[:] = True
        selector.last_selected_round[:] = 1
        selector.select()
        self.assertEqual(selector.epsilon, 0.0)

    def test_registered_duration_sets_percentile_target(self):
        selector = self._selector(
            round_threshold=50.0,
            initial_durations=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        selector._update_target_duration()
        self.assertEqual(selector.target_duration, 3.0)


class OortClientPriorTests(unittest.TestCase):
    def test_profile_priors_are_deterministic_and_capped(self):
        sizes = np.array([50.0, 200.0, 800.0, 1200.0])
        first = build_oort_client_priors(
            sizes, duration_proxy='profile', reward_cap_samples=640,
            profile_sigma=0.5, profile_compute_weight=0.5, seed=42,
        )
        second = build_oort_client_priors(
            sizes, duration_proxy='profile', reward_cap_samples=640,
            profile_sigma=0.5, profile_compute_weight=0.5, seed=42,
        )
        np.testing.assert_allclose(first[0], [50.0, 200.0, 640.0, 640.0])
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertAlmostEqual(float(np.median(first[1])), 1.0)
        self.assertEqual(first[2]['profile_seed'], 104771)

    def test_equal_proxy_removes_system_duration_differences(self):
        _, durations, metadata = build_oort_client_priors(
            np.array([10.0, 100.0, 1000.0]), duration_proxy='equal'
        )
        np.testing.assert_allclose(durations, np.ones(3))
        self.assertEqual(metadata['duration_proxy'], 'equal')


if __name__ == "__main__":
    unittest.main()
