#!/usr/bin/env python3
"""Synthetic tests for the C51 archive decision gate."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_archive as c51  # noqa: E402


def set_measurement(row: dict[str, str], name: str, value: float) -> None:
    row[f"{name}_status"] = "measured"
    row[f"{name}_value"] = str(value)
    row[f"{name}_reason"] = ""


def base_row(
    method: str,
    refinement: int,
    observation_class: str,
    *,
    law: str,
    ambient: int = 100,
) -> dict[str, str]:
    row = {
        "schema_version": c51.SCHEMA_VERSION,
        "study_id": c51.STUDY_ID,
        "method": method,
        "available": "true",
        "unavailable_reason": "",
        "discrete_law_id": law,
        "same_discrete_law_eligible": "true" if method == "fm_direct" else "false",
        "matched_physical_qoi_eligible": "true",
        "full_uq_workload_eligible": "true",
        "method_performance_timing_eligible": "true",
        "refinement": str(refinement),
        "spatial_dimension": "3",
        "form_degree": "1",
        "alpha": "2",
        "kappa": "4",
        "tau": "1",
        "observation_class": observation_class,
        "sensor_count": "20",
        "sensor_density": "0.02",
        "repeated_data_count": "10",
        "posterior_sample_count": "100",
        "seed": str(1000 + refinement),
        "latent_dimension": str(ambient),
        "ambient_dimension": str(ambient),
        "constraint_dimension": "0",
        "observation_nnz": "60",
        "solver": "test",
        "ordering": "amd",
    }
    for name in c51.MEASUREMENT_FIELDS:
        row[f"{name}_status"] = "not_recorded"
        row[f"{name}_value"] = ""
        row[f"{name}_reason"] = ""
    return row


def add_resources(row: dict[str, str], scale: float) -> None:
    for name in c51.COLD_TIME_FIELDS:
        set_measurement(row, name, 1_000_000.0 * scale)
    set_measurement(row, "factor_external_peak_rss_bytes", 100_000_000.0 * scale)
    set_measurement(row, "query_external_peak_rss_bytes", 80_000_000.0 * scale)
    set_measurement(row, "factor_storage_nnz", 1_000.0 * scale)
    set_measurement(row, "posterior_precision_nnz", 2_000.0 * scale)
    set_measurement(row, "kkt_input_nnz", 2_500.0 * scale)
    if "spectral" in row["method"]:
        set_measurement(row, "spectral_operator_recovery_ns", 400_000.0 * scale)
        set_measurement(row, "spectral_eigensolve_ns", 500_000.0 * scale)
        set_measurement(row, "spectral_gp_build_ns", 100_000.0 * scale)
        set_measurement(row, "spectral_basis_storage_values", 1.0)
        set_measurement(row, "spectral_gp_storage_values", 900.0 * scale)
        set_measurement(row, "spectral_prepared_storage_values", 800.0 * scale)
        set_measurement(row, "spectral_dense_factor_values", 100.0 * scale)
        set_measurement(row, "spectral_state_feature_values", 500.0 * scale)
        set_measurement(row, "spectral_observation_feature_values", 200.0 * scale)
        set_measurement(row, "spectral_sampling_normal_dimension", 50.0 * scale)
        row["factor_storage_nnz_status"] = "unavailable"
        row["factor_storage_nnz_value"] = ""
        row["factor_storage_nnz_reason"] = (
            "spectral conditioning reports its dense factor separately"
        )


def add_accuracy(row: dict[str, str], value: float) -> None:
    for name in c51.EQUALITY_FIELDS:
        set_measurement(row, name, value)


def synthetic_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for refinement in (8, 12):
        ambient = 100 * refinement
        for observation_class in ("local_physical_b", "local_selectors"):
            exact = base_row(
                "fm_direct",
                refinement,
                observation_class,
                law="exact_full_metric",
                ambient=ambient,
            )
            add_resources(exact, 8.0)
            rows.append(exact)

            compact = base_row(
                "whitney_compact",
                refinement,
                observation_class,
                law="projected_whitney",
                ambient=ambient,
            )
            add_resources(compact, 1.0)
            add_accuracy(compact, 0.05)
            rows.append(compact)

            under_resolved = base_row(
                "spectral_r",
                refinement,
                observation_class,
                law="exact_full_metric",
                ambient=ambient,
            )
            add_resources(under_resolved, 3.0)
            add_accuracy(under_resolved, 0.20)
            set_measurement(under_resolved, "spectral_rank", ambient // 8)
            set_measurement(
                under_resolved,
                "spectral_basis_storage_values",
                ambient * ambient // 8,
            )
            rows.append(under_resolved)

            full = base_row(
                "spectral_r",
                refinement,
                observation_class,
                law="exact_full_metric",
                ambient=ambient,
            )
            full["same_discrete_law_eligible"] = "true"
            add_resources(full, 10.0)
            add_accuracy(full, 1.0e-12)
            set_measurement(full, "spectral_rank", ambient)
            set_measurement(full, "spectral_basis_storage_values", ambient * ambient)
            rows.append(full)

            reduced = base_row(
                "spectral_r",
                refinement,
                observation_class,
                law="exact_full_metric",
                ambient=ambient,
            )
            add_resources(reduced, 4.0)
            add_accuracy(reduced, 0.05)
            set_measurement(reduced, "spectral_rank", ambient // 4)
            set_measurement(reduced, "spectral_basis_storage_values", ambient * ambient // 4)
            rows.append(reduced)
    return rows


def manifest(profile: str, row_count: int) -> dict[str, object]:
    return {
        "schema_version": c51.SCHEMA_VERSION,
        "study_id": c51.STUDY_ID,
        "profile": profile,
        "compiler_profile": "release",
        "root_sha": "a" * 40,
        "feec_sha": "b" * 40,
        "gmrf_sha": "c" * 40,
        "root_dirty": False,
        "feec_dirty": False,
        "gmrf_dirty": False,
        "comparison_frames": [
            "same_discrete_law_representation",
            "matched_physical_qoi_method",
        ],
        "hardware": {
            "host": "test-host",
            "cpu_model": "test-cpu",
            "logical_cores": 8,
            "total_memory_bytes": 16_000_000_000,
            "operating_system": "Linux",
            "kernel": "test-kernel",
            "thread_affinity": "0-3",
        },
        "rows_csv": "rows.csv",
        "row_count": row_count,
    }


class AnalyzeArchiveTests(unittest.TestCase):
    def test_csv_loader_accepts_complete_v3_triplets(self) -> None:
        rows = synthetic_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            loaded = c51.load_rows(path)
        self.assertEqual(len(loaded), len(rows))
        self.assertEqual(loaded[0]["schema_version"], c51.SCHEMA_VERSION)

    def test_publication_fixture_can_pass(self) -> None:
        rows = synthetic_rows()
        result = c51.analyze(manifest("juq-v1", len(rows)), rows, c51.GateConfig())
        self.assertEqual(result.decision, "PASS", result.blockers)
        self.assertTrue(any(":spectral_full_rank" in key for key in result.availability_counts))
        self.assertTrue(any(":spectral_low_rank" in key for key in result.availability_counts))
        self.assertTrue(result.crossovers)

    def test_smoke_and_reference_ladder_are_discovery_only(self) -> None:
        for profile in ("smoke", "reference_ladder"):
            rows = synthetic_rows()
            result = c51.analyze(manifest(profile, len(rows)), rows, c51.GateConfig())
            self.assertEqual(result.evidence_tier, "discovery")
            self.assertEqual(result.decision, "FAIL")
            self.assertTrue(any("discovery evidence" in value for value in result.blockers))

    def test_missing_peak_rss_blocks_publication(self) -> None:
        rows = synthetic_rows()
        row = next(value for value in rows if value["method"] == "whitney_compact")
        row["factor_external_peak_rss_bytes_status"] = "unavailable"
        row["factor_external_peak_rss_bytes_value"] = ""
        row["factor_external_peak_rss_bytes_reason"] = "external monitor absent"
        result = c51.analyze(manifest("juq-v1", len(rows)), rows, c51.GateConfig())
        self.assertEqual(result.decision, "FAIL")
        self.assertTrue(
            any("factor_external_peak_rss_bytes" in value for value in result.blockers)
        )

    def test_full_rank_equality_failure_blocks_publication(self) -> None:
        rows = synthetic_rows()
        row = next(
            value
            for value in rows
            if value["method"] == "spectral_r"
            and value["spectral_rank_value"] == value["ambient_dimension"]
        )
        set_measurement(row, "physical_variance_relative_error", 1.0e-3)
        result = c51.analyze(manifest("juq-v1", len(rows)), rows, c51.GateConfig())
        self.assertEqual(result.decision, "FAIL")
        self.assertTrue(any("equality" in value for value in result.blockers))

    def test_measurement_status_semantics_are_strict(self) -> None:
        rows = synthetic_rows()
        rows[0]["sampling_ns_value"] = ""
        with self.assertRaisesRegex(c51.ArchiveError, "measured sampling_ns has no value"):
            c51.analyze(manifest("juq-v1", len(rows)), rows, c51.GateConfig())

    def test_simple_manifest_parser_handles_lists_and_nested_metadata(self) -> None:
        text = """\
schema_version: "c51-accuracy-resource-v2"
study_id: "c51/accuracy-resource"
profile: "smoke"
comparison_frames:
  - "same_discrete_law_representation"
  - "matched_physical_qoi_method"
hardware:
  host: "worker"
  logical_cores: 8
rows_csv: "rows.csv"
row_count: 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text(text, encoding="utf-8")
            parsed = c51.parse_manifest(path)
        self.assertEqual(parsed["comparison_frames"][0], "same_discrete_law_representation")
        self.assertEqual(parsed["hardware"]["host"], "worker")

    def test_negative_raw_variance_is_preserved_as_a_signed_measurement(self) -> None:
        rows = synthetic_rows()
        set_measurement(rows[0], "physical_transformed_variance_min_raw", -1.0e-14)
        result = c51.analyze(manifest("smoke", len(rows)), rows, c51.GateConfig())
        self.assertEqual(result.evidence_tier, "discovery")

    def test_missing_spectral_prepared_storage_blocks_publication(self) -> None:
        rows = synthetic_rows()
        row = next(value for value in rows if value["method"] == "spectral_r")
        row["spectral_prepared_storage_values_status"] = "not_recorded"
        row["spectral_prepared_storage_values_value"] = ""
        row["spectral_prepared_storage_values_reason"] = ""
        result = c51.analyze(manifest("juq-v1", len(rows)), rows, c51.GateConfig())
        self.assertEqual(result.decision, "FAIL")
        self.assertTrue(
            any("spectral_prepared_storage_values" in value for value in result.blockers)
        )

    def test_rank_curve_reports_fixed_and_fractional_smallest_passing_rank(self) -> None:
        rows = synthetic_rows()
        result = c51.analyze(manifest("juq-v1", len(rows)), rows, c51.GateConfig())
        curve = next(
            value
            for value in result.rank_curves
            if value.observation_class == "local_physical_b" and value.refinement == 8
        )
        joint = next(value for value in curve.thresholds if value.metric == "joint")
        self.assertEqual(joint.smallest_eligible_rank, curve.ambient_dimension // 4)
        self.assertAlmostEqual(joint.smallest_eligible_rank_fraction, 0.25)
        self.assertEqual(joint.bracket_status, "bracketed")

    def test_v3_shared_basis_timing_is_not_aggregated(self) -> None:
        rows = synthetic_rows()
        spectral = next(value for value in rows if value["method"] == "spectral_r")
        spectral["method_performance_timing_eligible"] = "false"
        result = c51.analyze(manifest("smoke", len(rows)), rows, c51.GateConfig())
        row_number = rows.index(spectral) + 2
        self.assertFalse(
            any(
                value.group_key == c51.row_group_key(spectral)
                and value.method_label == c51.method_label(spectral, row_number)
                for value in result.aggregates
            )
        )

    def test_v2_archive_remains_readable_without_timing_eligibility_column(self) -> None:
        rows = synthetic_rows()
        for row in rows:
            row["schema_version"] = "c51-accuracy-resource-v2"
            del row["method_performance_timing_eligible"]
        legacy_manifest = manifest("smoke", len(rows))
        legacy_manifest["schema_version"] = "c51-accuracy-resource-v2"
        result = c51.analyze(legacy_manifest, rows, c51.GateConfig())
        self.assertEqual(result.evidence_tier, "discovery")


if __name__ == "__main__":
    unittest.main()
