from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from experiments.gsa.gsa import GradientProjector
from experiments.gsa.prepare_dataset import node_for_ship_type


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments" / "gsa"


class ExperimentContracts(unittest.TestCase):
    def test_ship_type_nodes(self) -> None:
        cases = {
            "Container ship": "container", "Oil tanker": "tanker", "Chemical tanker": "tanker",
            "Bulk carrier": "bulk", "General cargo ship": "general_cargo",
            "Passenger ship (Cruise Passenger ship)": "passenger",
        }
        self.assertEqual({name: node_for_ship_type(name) for name in cases}, cases)

    def test_protocol(self) -> None:
        config = json.loads((PACKAGE / "configs" / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(config["modalities"], ["ais", "weather", "registry", "emissions"])
        self.assertEqual(config["nodes"], ["container", "tanker", "bulk", "general_cargo", "passenger"])
        self.assertEqual(config["mia"]["shadow_models"], 50)
        self.assertEqual(set(config["ablations"]), {"A", "B", "C", "D", "E", "D_NONLINEAR"})
        self.assertEqual(
            set(config["robustness"]),
            {
                "ais_missing_fraction", "ais_drift_km",
                "weather_missing_fraction", "client_dropout_rate",
            },
        )

    def test_emissions_probe_excludes_emissions_tower(self) -> None:
        source = (PACKAGE / "downstream_emissions.py").read_text(encoding="utf-8")
        self.assertIn('outputs["ais"], outputs["weather"], outputs["registry"]', source)
        self.assertNotIn('outputs["emissions"]], dim=1', source)

    def test_teacher_priority_trajectory_metrics(self) -> None:
        source = (PACKAGE / "trajectory_prediction.py").read_text(encoding="utf-8")
        for metric in ("ade_5_km", "fde_5_km", "ade_10_km", "fde_10_km"):
            self.assertIn(metric, source)
        builder = (PACKAGE / "build_trajectory_sequences.py").read_text(encoding="utf-8")
        self.assertIn('default=10', builder)
        self.assertIn('max-gap-minutes', builder)

    def test_teacher_priority_robustness_scope(self) -> None:
        source = (PACKAGE / "robustness.py").read_text(encoding="utf-8")
        self.assertIn('"ais_missing_fraction"', source)
        self.assertIn('"ais_drift_km"', source)
        self.assertIn('"weather_missing_fraction"', source)
        self.assertIn('"client_dropout_rate"', source)
        self.assertNotIn('"gradient_packet_loss"', source)

    def test_privacy_is_risk_analysis_not_defense(self) -> None:
        source = (PACKAGE / "privacy_risk_analysis.py").read_text(encoding="utf-8")
        self.assertIn("GSA is not a privacy defense", source)
        self.assertIn("Do not claim that GSA prevents", source)

    def test_hash_sketch_uses_normalized_adjoint(self) -> None:
        projector = GradientProjector(
            1003, 16, structured_dim=128, structured_mode="hash"
        )
        x = torch.randn(1003)
        y = torch.randn(128)
        sketch = torch.zeros(128).scatter_add_(
            0, projector.hash_index, x * projector.hash_weight
        )
        back = y[projector.hash_index] * projector.hash_weight
        self.assertTrue(torch.allclose(torch.dot(sketch, y), torch.dot(x, back)))


if __name__ == "__main__":
    unittest.main()
