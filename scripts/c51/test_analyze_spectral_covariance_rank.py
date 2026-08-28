#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("analyze_spectral_covariance_rank.py")
SPEC = importlib.util.spec_from_file_location("c51_rank_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(rank, selection, heldout, status="measured", evidence_tier="rank_discovery"):
    return {
        "evidence_tier": evidence_tier,
        "refinement": "8",
        "alpha": "2",
        "design": "prior",
        "rank": str(rank),
        "status": status,
        "selection_e_primary": "" if selection is None else str(selection),
        "heldout_e_primary": "" if heldout is None else str(heldout),
    }


def decision(status, rank, evidence_tier, threshold="5.00000000000000028e-02"):
    return {
        "refinement": "8",
        "alpha": "2",
        "design": "density_0.10",
        "threshold": threshold,
        "evidence_tier": evidence_tier,
        "maximum_tested_rank": "2048" if evidence_tier == "rank_discovery" else "3032",
        "locked_rank": str(rank) if rank else "",
        "selection_error_at_locked_rank": "0.04" if rank else "",
        "heldout_error_at_locked_rank": "0.051" if rank else "",
        "status": status,
        "reason": f"{status} reason",
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

    def test_rank_extension_failure_is_greater_than_actual_maximum_tested(self):
        rows = [
            row(2048, 0.04, 0.04, evidence_tier="rank_extension"),
            row(2560, 0.03, 0.03, evidence_tier="rank_extension"),
            row(4096, 0.021, 0.019, evidence_tier="rank_extension"),
        ]
        result = MODULE.analyze_group(rows, 0.02)
        self.assertEqual(result["evidence_tier"], "rank_extension")
        self.assertEqual(result["status"], "required_rank_gt_max_tested")
        self.assertEqual(result["maximum_tested_rank"], "4096")
        self.assertIn("4096", result["reason"])

    def test_rank_curve_rejects_mixed_primary_and_extension_rows(self):
        rows = [
            row(2048, 0.04, 0.04),
            row(2304, 0.03, 0.03, evidence_tier="rank_extension"),
        ]
        with self.assertRaisesRegex(ValueError, "mix evidence tiers"):
            MODULE.analyze_group(rows, 0.05)

    def test_extension_does_not_retune_primary_heldout_failure(self):
        primary = decision("heldout_failed", 2000, "rank_discovery")
        extension = decision("attained", 2048, "rank_extension")
        merged = MODULE.merge_primary_extension([primary], [extension])[0]
        self.assertEqual(merged["status"], "heldout_failed")
        self.assertEqual(merged["locked_rank"], "2000")
        self.assertEqual(merged["decision_source_tier"], "rank_discovery")
        self.assertEqual(merged["post_primary_candidate_rank"], "2048")
        self.assertIn("does not retune", merged["reason"])

    def test_extension_resolves_only_primary_rank_censor(self):
        primary = decision("required_rank_gt_2048", None, "rank_discovery")
        extension = decision("attained", 2560, "rank_extension")
        merged = MODULE.merge_primary_extension([primary], [extension])[0]
        self.assertEqual(merged["status"], "attained")
        self.assertEqual(merged["locked_rank"], "2560")
        self.assertEqual(merged["decision_source_tier"], "rank_extension")
        self.assertEqual(merged["primary_status"], "required_rank_gt_2048")

    def test_extension_preserves_already_attained_primary_rank(self):
        primary = decision("attained", 1280, "rank_discovery")
        extension = decision("attained", 2048, "rank_extension")
        merged = MODULE.merge_primary_extension([primary], [extension])[0]
        self.assertEqual(merged["locked_rank"], "1280")
        self.assertEqual(merged["decision_source_tier"], "rank_discovery")


if __name__ == "__main__":
    unittest.main()
