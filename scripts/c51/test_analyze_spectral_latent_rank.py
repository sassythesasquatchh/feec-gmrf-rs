#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("analyze_spectral_latent_rank.py")
SPEC = importlib.util.spec_from_file_location("c51_latent_rank_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(rank, error, tier="rank_discovery", status="measured"):
    return {
        "refinement": "8",
        "state_dimension": "3032",
        "alpha": "2",
        "design": "prior",
        "evidence_tier": tier,
        "rank": str(rank),
        "rank_fraction": str(rank / 3032),
        "status": status,
        "latent_mass_action_maximum_relative_error": str(error),
    }


class LatentRankAnalyzerTests(unittest.TestCase):
    def test_combines_primary_and_extension_without_reselecting_duplicate(self):
        rows = [
            row(1024, 0.08),
            row(2048, 0.03),
            row(2048, 0.03, "rank_extension"),
            row(2304, 0.015, "rank_extension"),
            row(3032, 1.0e-12, "rank_extension"),
        ]
        result = MODULE.analyze_group(rows, 0.02)
        self.assertEqual(result["status"], "attained")
        self.assertEqual(result["locked_rank"], "2304")
        self.assertEqual(result["decision_source_tier"], "rank_extension")
        self.assertEqual(result["evidence_span"], "rank_discovery+rank_extension")

    def test_duplicate_rank_must_reproduce_error(self):
        rows = [row(2048, 0.03), row(2048, 0.031, "rank_extension")]
        with self.assertRaisesRegex(ValueError, "inconsistent latent error"):
            MODULE.analyze_group(rows, 0.02)

    def test_small_duplicate_solver_delta_is_archived(self):
        rows = [row(2048, 0.03), row(2048, 0.0300000000017, "rank_extension")]
        result = MODULE.analyze_group(rows, 0.02)
        self.assertAlmostEqual(
            float(result["maximum_duplicate_rank_absolute_error_delta"]), 1.7e-12
        )
        self.assertEqual(
            float(result["duplicate_reproducibility_absolute_tolerance"]), 5.0e-12
        )

    def test_unattained_measured_curve_reports_actual_cap(self):
        rows = [row(2048, 0.08), row(4096, 0.021, "rank_extension")]
        result = MODULE.analyze_group(rows, 0.02)
        self.assertEqual(result["status"], "required_rank_gt_max_tested")
        self.assertEqual(result["maximum_tested_rank"], "4096")

    def test_censor_is_not_a_rank_lower_bound(self):
        rows = [row(2048, 0.08), row(4096, "", "rank_extension", "censored")]
        result = MODULE.analyze_group(rows, 0.02)
        self.assertEqual(result["status"], "computationally_unavailable")


if __name__ == "__main__":
    unittest.main()
