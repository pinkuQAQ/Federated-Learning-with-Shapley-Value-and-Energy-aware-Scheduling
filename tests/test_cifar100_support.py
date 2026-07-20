import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import CNNCifar
from options import args_parser
from sampling import cifar_noniid_dirichlet


class FakeCifar100Dataset:
    def __init__(self, samples_per_class=2):
        self.targets = [
            label
            for label in range(100)
            for _ in range(samples_per_class)
        ]

    def __len__(self):
        return len(self.targets)


class Cifar100SupportTests(unittest.TestCase):
    def test_parser_sets_cifar100_shape(self):
        with patch.object(sys, "argv", ["test", "--dataset", "cifar100"]):
            args = args_parser()

        self.assertEqual(args.num_classes, 100)
        self.assertEqual(args.num_channels, 3)

    def test_cifar_model_outputs_one_hundred_logits(self):
        model = CNNCifar(SimpleNamespace(num_classes=100))
        output = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(tuple(output.shape), (2, 100))

    def test_dirichlet_partition_supports_one_hundred_classes(self):
        dataset = FakeCifar100Dataset(samples_per_class=2)
        groups = cifar_noniid_dirichlet(
            dataset,
            num_users=10,
            alpha=0.5,
            seed=7,
            dataset_name="CIFAR100",
        )

        assigned = set().union(*groups.values())
        self.assertEqual(len(assigned), len(dataset))
        self.assertEqual(len(groups), 10)


if __name__ == "__main__":
    unittest.main()
