#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "analyze_juq_performance.py"
SPEC = importlib.util.spec_from_file_location("analyze_juq_performance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class AnalyzeJuqPerformanceTests(unittest.TestCase):
    def points(self, compact_time=60.0, compact_rss=70.0):
        points = []
        for refinement in analysis.REFINEMENTS:
            for data_count in analysis.DATA_COUNTS:
                for replicate in range(3):
                    points.extend(
                        [
                            analysis.ResourcePoint(
                                refinement,
                                data_count,
                                replicate,
                                "whitney_compact",
                                compact_time,
                                compact_rss,
                                50.0,
                            ),
                            analysis.ResourcePoint(
                                refinement,
                                data_count,
                                replicate,
                                "exact_observation_gp",
                                100.0,
                                100.0,
                                100.0,
                            ),
                            analysis.ResourcePoint(
                                refinement,
                                data_count,
                                replicate,
                                "spectral_low_rank",
                                100.0,
                                100.0,
                                100.0,
                            ),
                        ]
                    )
        return points

    def test_frozen_rule_accepts_25_percent_time_win_without_memory_regression(self):
        comparisons = analysis.compare(self.points())
        self.assertEqual(len(comparisons), 8)
        self.assertTrue(all(item.practical_advantage for item in comparisons))

    def test_frozen_rule_rejects_advantage_with_other_resource_regression(self):
        comparisons = analysis.compare(self.points(compact_time=70.0, compact_rss=120.0))
        self.assertFalse(any(item.practical_advantage for item in comparisons))

    def test_exact_same_law_rule_reports_memory_tradeoff_but_adjudicates_time(self):
        points = []
        for refinement in analysis.REFINEMENTS:
            for data_count in analysis.DATA_COUNTS:
                for replicate in range(3):
                    points.extend(
                        [
                            analysis.ResourcePoint(
                                refinement, data_count, replicate, "fm_aug", 50.0, 200.0, 80.0
                            ),
                            analysis.ResourcePoint(
                                refinement,
                                data_count,
                                replicate,
                                "exact_observation_gp",
                                100.0,
                                100.0,
                                100.0,
                            ),
                            analysis.ResourcePoint(
                                refinement,
                                data_count,
                                replicate,
                                "spectral_low_rank",
                                100.0,
                                100.0,
                                100.0,
                            ),
                        ]
                    )
        comparisons = analysis.compare(
            points,
            baseline="fm_aug",
            comparators=analysis.COMPARATORS,
            memory_noninferiority_required=False,
        )
        self.assertTrue(all(item.practical_advantage for item in comparisons))
        self.assertTrue(all(item.peak_rss_ratio_median == 2.0 for item in comparisons))

    def test_three_replicates_are_required_for_paired_interval(self):
        with self.assertRaisesRegex(analysis.AnalysisError, "at least three"):
            analysis.exact_paired_bootstrap_median_ci([0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
