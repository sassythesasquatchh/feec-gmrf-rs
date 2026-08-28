#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("analyze_spectral_covariance_rank.py")
SPEC = importlib.util.spec_from_file_location("c51_rank_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(rank, selection, heldout, status="measured"):
    return {
        "refinement": "8",
        "alpha": "2",
        "design": "prior",
        "rank": str(rank),
        "status": status,
        "selection_e_primary": "" if selection is None else str(selection),
        "heldout_e_primary": "" if heldout is None else str(heldout),
    }


class RankAnalyzerTests(unittest.TestCase):
    def test_selects_smallest_sustained_prefix(self):
        rows = [
            row(64, 0.09, 0.09),
            row(128, 0.11, 0.08),
            row(256, 0.08, 0.07),
            row(512, 0.07, 0.06),
        ]
        result = MODULE.analyze_group(rows, 0.10)
        self.assertEqual(result["status"], "attained")
        self.assertEqual(result["locked_rank"], "256")

    def test_heldout_failure_does_not_retune(self):
        rows = [row(64, 0.08, 0.12), row(128, 0.07, 0.04)]
        result = MODULE.analyze_group(rows, 0.10)
        self.assertEqual(result["status"], "heldout_failed")
        self.assertEqual(result["locked_rank"], "64")

    def test_measured_failure_through_2048_means_greater_than_2048(self):
        rows = [row(1024, 0.11, 0.09), row(2048, 0.101, 0.08)]
        result = MODULE.analyze_group(rows, 0.10)
        self.assertEqual(result["status"], "required_rank_gt_2048")

    def test_censored_2048_is_computationally_unavailable(self):
        rows = [row(1024, 0.11, 0.09), row(2048, None, None, "censored")]
        result = MODULE.analyze_group(rows, 0.10)
        self.assertEqual(result["status"], "computationally_unavailable")
        self.assertNotEqual(result["status"], "required_rank_gt_2048")


if __name__ == "__main__":
    unittest.main()
