import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyapunov_optimizer import LyapunovTripleScheduler


class LyapunovTripleSchedulerTests(unittest.TestCase):
    def test_queue_updates_all_clients_each_round(self):
        scheduler = LyapunovTripleScheduler(num_clients=3, energy_budget=2.0)
        scheduler.energy_queue = np.array([5.0, 5.0, 5.0])

        scheduler.update_queue(
            energy_consumed=np.array([7.0]),
            selected_clients=[1],
            round_num=1,
        )

        np.testing.assert_allclose(scheduler.energy_queue, [3.0, 10.0, 3.0])

    def test_score_uses_physical_energy_scale(self):
        scheduler = LyapunovTripleScheduler(num_clients=2, V=0.0, energy_budget=1.0)
        scheduler.energy_queue = np.array([2.0, 2.0])

        scores = scheduler.compute_scores(
            shapley_values=np.zeros(2),
            energy_consumed=np.array([1.5, 3.0]),
        )

        np.testing.assert_allclose(scores, [-3.0, -6.0])


if __name__ == "__main__":
    unittest.main()
