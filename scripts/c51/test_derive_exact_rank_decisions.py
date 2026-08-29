#!/usr/bin/env python3

import csv
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "derive_exact_rank_decisions.py"
SPEC = importlib.util.spec_from_file_location("derive_exact_rank_decisions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
decision = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = decision
SPEC.loader.exec_module(decision)


class DeriveExactRankDecisionTests(unittest.TestCase):
    def write_perf0(self, root: pathlib.Path) -> None:
        (root / "manifest.yaml").write_text("profile: perf0\n", encoding="utf-8")
        metrics = (
            "physical_output_covariance_action_relative_error",
            "physical_variance_relative_error",
            "physical_transformed_variance_relative_error",
            "physical_mean_relative_error",
        )
        fieldnames = [
            "method",
            "same_discrete_law_eligible",
            "observation_design_id",
            "physical_qoi_design_id",
            "physical_qoi_output_count",
            *[f"{metric}_{suffix}" for metric in metrics for suffix in ("status", "value")],
        ]
        with (root / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for method in ("fm_aug", "exact_observation_gp"):
                row = {
                    "method": method,
                    "same_discrete_law_eligible": "true",
                    "observation_design_id": decision.SENSOR_DESIGN,
                    "physical_qoi_design_id": decision.OUTPUT_DESIGN,
                    "physical_qoi_output_count": "18",
                }
                for metric in metrics:
                    row[f"{metric}_status"] = "measured"
                    row[f"{metric}_value"] = "0"
                writer.writerow(row)

    def write_attempt(self, root: pathlib.Path, dirty: bool = False) -> None:
        study = root / "study"
        study.mkdir(parents=True)
        for name in decision.ARCHIVE_FILES:
            (study / name).write_text("", encoding="utf-8")
        fieldnames = [
            "refinement",
            "state_dimension",
            "alpha",
            "design",
            "observation_count",
            "rank",
            "status",
            "selection_e_primary",
            "heldout_e_primary",
            "method_performance_timing_eligible",
        ]
        with (study / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "refinement": 8,
                    "state_dimension": 1000,
                    "alpha": 2,
                    "design": "high_m4096",
                    "observation_count": 4096,
                    "rank": 128,
                    "status": "measured",
                    "selection_e_primary": 0.03,
                    "heldout_e_primary": 0.03,
                    "method_performance_timing_eligible": "false",
                }
            )
            writer.writerow(
                {
                    "refinement": 8,
                    "state_dimension": 1000,
                    "alpha": 2,
                    "design": "high_m4096",
                    "observation_count": 4096,
                    "rank": 256,
                    "status": "measured",
                    "selection_e_primary": 0.01,
                    "heldout_e_primary": 0.015,
                    "method_performance_timing_eligible": "false",
                }
            )
        compact_fields = [
            "method",
            "method_performance_timing_eligible",
            "refinement",
            "alpha",
            "design",
            "observation_count",
            "status",
            "selection_e_primary",
            "heldout_e_primary",
            "selection_e_corr",
            "heldout_e_corr",
        ]
        with (study / "compact_accuracy.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=compact_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "method": "whitney_compact_projected",
                    "method_performance_timing_eligible": "false",
                    "refinement": 8,
                    "alpha": 2,
                    "design": "high_m4096",
                    "observation_count": 4096,
                    "status": "measured",
                    "selection_e_primary": 0.13,
                    "heldout_e_primary": 0.19,
                    "selection_e_corr": 0.02,
                    "heldout_e_corr": 0.01,
                }
            )
        manifest = {
            "outcome": "completed",
            "raw_return_code": 0,
            "evidence_tier": "high_observation_replay_accuracy_only",
            "profile": "high-observation-m4096-replay-v1-n8-r3032",
            "method_performance_timing_eligible": False,
            "source": {
                "repository": {"head": "a" * 40, "dirty": dirty},
                "submodules": {
                    "feec": {"head": "b" * 40, "dirty": False},
                    "gmrf": {"head": "c" * 40, "dirty": False},
                },
            },
        }
        (root / "attempt_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_derives_preregistered_sensitivity_rank_and_binds_perf0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            perf0 = root / "perf0"
            attempt = root / "attempt"
            perf0.mkdir()
            self.write_perf0(perf0)
            self.write_attempt(attempt)
            payload = decision.derive([attempt], perf0)
        selected = payload["decisions"][0]
        self.assertEqual(selected["spectral_low_rank"]["rank"], 256)
        self.assertEqual(selected["spectral_low_rank"]["lower_failing_rank"], 128)
        self.assertEqual(
            selected["spectral_low_rank"]["qualification_rule"],
            "pre_registered_2pct_sensitivity",
        )
        self.assertEqual(
            selected["exact_constrained"]["fixture_artifact_sha256"],
            selected["exact_observation_gp"]["fixture_artifact_sha256"],
        )

    def test_rejects_dirty_replay_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            perf0 = root / "perf0"
            attempt = root / "attempt"
            perf0.mkdir()
            self.write_perf0(perf0)
            self.write_attempt(attempt, dirty=True)
            with self.assertRaisesRegex(ValueError, "dirty"):
                decision.derive([attempt], perf0)

    def test_projected_profile_archives_fixed_tau_accuracy_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            perf0 = root / "perf0"
            attempt = root / "attempt"
            perf0.mkdir()
            self.write_perf0(perf0)
            self.write_attempt(attempt)
            payload = decision.derive(
                [attempt], perf0, decision.PROJECTED_PROFILE
            )
        compact = payload["decisions"][0]["whitney_compact"]
        self.assertEqual(payload["profile"], decision.PROJECTED_PROFILE)
        self.assertEqual(compact["validation_status"], "measured_accuracy_context")
        self.assertTrue(compact["fixed_tau_unnormalized"])
        self.assertFalse(compact["passes_primary_5pct_envelope"])
        self.assertEqual(compact["selection_relative_error"], 0.13)
        self.assertEqual(compact["heldout_relative_error"], 0.19)


if __name__ == "__main__":
    unittest.main()
