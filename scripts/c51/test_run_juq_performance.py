#!/usr/bin/env python3

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "run_juq_performance.py"
SPEC = importlib.util.spec_from_file_location("run_juq_performance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


class JuqPerformanceCampaignTests(unittest.TestCase):
    def setUp(self):
        self.profile = campaign.read_json(campaign.DEFAULT_PROFILE)
        campaign.validate_profile(self.profile)

    def decision(self, refinement, count, tolerance=0.05):
        return {
            "decision_id": f"test-n{refinement}-m{count}",
            "refinement": refinement,
            "scalar_observation_count": count,
            "alpha": 2,
            "relative_tolerance": tolerance,
            "observation_design_id": "c51-magnetic-fixed-footprint-maximin-12-v1",
            "physical_qoi_design_id": "rank_study_selection_and_held_out_wendland_c2_18",
            "reference_discrete_law_id": "exact_full_metric_whitney",
            "common_h_g_fixture_sha256": "a" * 64,
            "state_dimension": 1000,
            "rank": 256,
            "spectral_low_rank": {
                "rank": 256,
                "validation_status": "heldout_validated",
                "selection_relative_error": 0.01,
                "heldout_relative_error": 0.01,
                "lower_failing_rank": 128,
                "lower_failing_selection_relative_error": 0.06,
                "rank_status": "low_rank",
                "timing_basis_source": "partial_exact_requested_mixed_shell",
                "shared_maximum_basis_timing_eligible": False,
                "accuracy_artifact_sha256": "a" * 64,
            },
            "whitney_compact": {
                "validation_status": "heldout_validated",
                "selection_relative_error": 0.01,
                "heldout_relative_error": 0.01,
                "accuracy_artifact_sha256": "a" * 64,
                "identical_observation_and_output_maps": True,
                "method_performance_timing_eligible": False,
            },
            "exact_observation_gp": {
                "validation_status": "perf0_same_law_validated",
                "fixture_id": "PERF-0",
                "fixture_artifact_sha256": "d" * 64,
            },
        }

    def complete_decisions(self):
        return {
            campaign.rank_key(refinement, count, self.profile): self.decision(
                refinement, count
            )
            for refinement in (8, 10)
            for count in (512, 2048, 4096, 5184)
        }

    def write_decisions(self, path, decisions):
        path.write_text(
            json.dumps(
                {
                    "schema_version": campaign.RANK_DECISION_SCHEMA_VERSION,
                    "profile": campaign.PROFILE_NAME,
                    "status": "frozen_heldout_validated",
                    "decisions": decisions,
                }
            ),
            encoding="utf-8",
        )

    def test_pending_template_blocks_cell_generation_without_dropping_spectral_cells(self):
        decisions, _ = campaign.load_rank_decisions(
            campaign.DEFAULT_RANK_DECISIONS, self.profile
        )
        self.assertEqual(len(campaign.missing_rank_keys(self.profile, decisions)), 8)
        with self.assertRaisesRegex(ValueError, "missing 5 percent rank"):
            campaign.generate_cells(self.profile, decisions)

    def test_exact_profile_freezes_same_law_methods_and_time_campaign(self):
        profile_path = SCRIPT_DIR / "juq-exact-performance-v1.json"
        profile = campaign.read_json(profile_path)
        campaign.validate_profile(profile)
        self.assertEqual(profile["execution"]["cgroup_mode"], "auto")
        self.assertFalse(profile["execution"]["continue_after_cell_failure"])
        decisions = []
        for refinement in (8, 10):
            decision = self.decision(refinement, 4096)
            decision.pop("whitney_compact")
            decision["spectral_low_rank"]["qualification_rule"] = (
                "pre_registered_2pct_sensitivity"
            )
            decision["spectral_low_rank"]["rank_selection_tolerance"] = 0.02
            decision["exact_constrained"] = {
                "validation_status": "perf0_same_law_validated",
                "fixture_id": "PERF-0",
                "fixture_artifact_sha256": "d" * 64,
            }
            decisions.append(decision)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "exact-decisions.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": campaign.RANK_DECISION_SCHEMA_VERSION,
                        "profile": campaign.EXACT_PROFILE_NAME,
                        "status": "frozen_heldout_validated",
                        "decisions": decisions,
                    }
                ),
                encoding="utf-8",
            )
            indexed, _ = campaign.load_rank_decisions(path, profile)
        cells = campaign.generate_cells(profile, indexed)
        self.assertEqual(len(cells), 96)
        self.assertEqual(
            {cell.method for cell in cells},
            {"fm_aug", "exact_observation_gp", "spectral_low_rank"},
        )
        self.assertEqual(
            {cell.evidence_tier for cell in cells}, {"claim_primary_exact"}
        )
        command = campaign.cell_command(
            cells[0],
            pathlib.Path("/repo"),
            pathlib.Path("/out"),
            campaign.DEFAULT_RUNNER,
            profile,
        )
        self.assertEqual(command[command.index("--cgroup-mode") + 1], "auto")
        command = campaign.cell_command(
            next(cell for cell in cells if cell.method == "fm_aug"),
            pathlib.Path("/repo"),
            pathlib.Path("/out"),
            campaign.DEFAULT_RUNNER,
            profile,
        )
        self.assertEqual(
            command[command.index("--profile") + 1], campaign.EXACT_PROFILE_NAME
        )

    def test_selected_primary_tier_requires_only_its_two_rank_decisions(self):
        decisions = {
            campaign.rank_key(refinement, 2048, self.profile): self.decision(
                refinement, 2048
            )
            for refinement in (8, 10)
        }
        self.assertEqual(
            campaign.missing_rank_keys(
                self.profile, decisions, ["claim_primary"]
            ),
            [],
        )
        self.assertEqual(len(campaign.missing_rank_keys(self.profile, decisions)), 6)
        cells = campaign.generate_cells(
            self.profile, decisions, ["claim_primary"]
        )
        self.assertEqual(len(cells), 96)
        self.assertEqual({cell.evidence_tier for cell in cells}, {"claim_primary"})
        self.assertEqual(
            campaign.projected_cell_counts(self.profile, ["claim_primary"]),
            {"warmup": 24, "measured": 72, "total": 96},
        )

    def test_selected_tiers_are_canonical_and_duplicate_selection_is_rejected(self):
        tiers = campaign.selected_evidence_tiers(
            self.profile, ["capacity_stress", "claim_primary"]
        )
        self.assertEqual(
            [tier["id"] for tier in tiers], ["claim_primary", "capacity_stress"]
        )
        with self.assertRaisesRegex(ValueError, "at most once"):
            campaign.selected_evidence_tiers(
                self.profile, ["claim_primary", "claim_primary"]
            )

    def test_plan_archives_selected_and_omitted_tiers_when_primary_is_ready(self):
        decisions = {
            campaign.rank_key(refinement, 2048, self.profile): self.decision(
                refinement, 2048
            )
            for refinement in (8, 10)
        }
        cells = campaign.generate_cells(
            self.profile, decisions, ["claim_primary"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            plan = campaign.plan_document(
                pathlib.Path("/repo"),
                campaign.DEFAULT_PROFILE,
                campaign.DEFAULT_RANK_DECISIONS,
                campaign.DEFAULT_RUNNER,
                self.profile,
                decisions,
                cells,
                pathlib.Path(temporary),
                ["claim_primary"],
            )
        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["selected_evidence_tiers"], ["claim_primary"])
        self.assertEqual(
            plan["omitted_evidence_tiers"],
            ["scaling_context", "high_data_confirmation", "capacity_stress"],
        )
        self.assertEqual(plan["projected_cell_count"], 96)

    def test_rank_decisions_reject_context_tolerance_as_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "decisions.json"
            self.write_decisions(path, [self.decision(8, 512, tolerance=0.10)])
            with self.assertRaisesRegex(ValueError, "primary 5 percent"):
                campaign.load_rank_decisions(path, self.profile)

    def test_source_identity_requires_clean_complete_revisions(self):
        source = {
            "head": "a" * 40,
            "dirty": False,
            "submodules": {
                "feec": {"head": "b" * 40, "dirty": False},
                "gmrf": {"head": "c" * 40, "dirty": False},
            },
        }
        self.assertEqual(
            campaign.source_identity(source),
            {"root_sha": "a" * 40, "feec_sha": "b" * 40, "gmrf_sha": "c" * 40},
        )
        dirty = copy.deepcopy(source)
        dirty["submodules"]["gmrf"]["dirty"] = True
        with self.assertRaisesRegex(ValueError, "gmrf_dirty=false"):
            campaign.source_identity(dirty)

    def test_tiered_cells_are_complete_nested_and_deterministically_randomized(self):
        decisions = self.complete_decisions()
        first = campaign.generate_cells(self.profile, decisions)
        second = campaign.generate_cells(self.profile, decisions)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 192)
        self.assertEqual(sum(cell.run_kind == "warmup" for cell in first), 24)
        self.assertEqual(sum(cell.run_kind == "measured" for cell in first), 168)
        self.assertEqual(sum(cell.evidence_tier == "claim_primary" for cell in first), 96)
        self.assertEqual(sum(cell.evidence_tier == "scaling_context" for cell in first), 48)
        self.assertEqual(
            sum(cell.evidence_tier == "high_data_confirmation" for cell in first), 36
        )
        self.assertEqual(sum(cell.evidence_tier == "capacity_stress" for cell in first), 12)
        blocks = {}
        for cell in first:
            blocks.setdefault(cell.block_index, set()).add(cell.method)
        self.assertTrue(
            all(methods == set(campaign.EXPECTED_AXES["methods"]) for methods in blocks.values())
        )
        spectral = next(cell for cell in first if cell.method == "spectral_low_rank")
        command = campaign.cell_command(
            spectral,
            pathlib.Path("/repo"),
            pathlib.Path("/out"),
            campaign.DEFAULT_RUNNER,
            self.profile,
        )
        self.assertIn("--spectral-rank", command)
        self.assertEqual(command[command.index("--spectral-rank") + 1], "256")
        self.assertIn("--cgroup-mode", command)
        self.assertEqual(command[command.index("--cgroup-mode") + 1], "required")

    def base_phase_row(self):
        row = {field: f"identity-{field}" for field in campaign.PAIR_KEY_FIELDS}
        row.update(
            {
                "available": "true",
                "common_physical_uq_workload_eligible": "true",
                "matched_physical_qoi_eligible": "true",
                "full_uq_workload_eligible": "true",
                "method_performance_timing_eligible": "true",
                "method_performance_timing_ineligibility_reason": "",
                "observation_class": "fixed_footprint_magnetic_volume",
            }
        )
        for name in campaign.analyze_archive.MEASUREMENT_FIELDS:
            row[f"{name}_status"] = "measured"
            row[f"{name}_value"] = f"base-{name}"
            row[f"{name}_reason"] = ""
        return row

    def test_phase_merge_uses_declared_sources_and_retains_both_manifest_hashes(self):
        setup = self.base_phase_row()
        query = copy.deepcopy(setup)
        for name in campaign.analyze_archive.MEASUREMENT_FIELDS:
            setup[f"{name}_value"] = f"setup-{name}"
            query[f"{name}_value"] = f"query-{name}"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            setup_manifest = root / "setup.json"
            query_manifest = root / "query.json"
            setup_manifest.write_text('{"phase":"setup"}\n', encoding="utf-8")
            query_manifest.write_text('{"phase":"query"}\n', encoding="utf-8")
            merged = campaign.merge_phase_rows(
                setup, query, setup_manifest, query_manifest
            )
        self.assertEqual(merged["feec_assembly_ns_value"], "setup-feec_assembly_ns")
        self.assertEqual(
            merged["factor_external_peak_rss_bytes_value"],
            "setup-factor_external_peak_rss_bytes",
        )
        self.assertEqual(merged["repeated_data_ns_value"], "query-repeated_data_ns")
        self.assertEqual(
            merged["variance_estimation_ns_value"], "query-variance_estimation_ns"
        )
        self.assertEqual(
            merged["query_external_peak_rss_bytes_value"],
            "query-query_external_peak_rss_bytes",
        )
        self.assertEqual(len(merged["cold_setup_runner_manifest_sha256"]), 64)
        self.assertEqual(len(merged["query_physical_uq_runner_manifest_sha256"]), 64)
        self.assertIn("setup/storage", merged["phase_merge_policy"])

    def test_phase_merge_rejects_different_scientific_cells(self):
        setup = self.base_phase_row()
        query = copy.deepcopy(setup)
        query["sensor_count"] = "different"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text("{}\n", encoding="utf-8")
            right.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scientific identity"):
                campaign.merge_phase_rows(setup, query, left, right)

    def test_timeout_and_sigkill_remain_typed_censors(self):
        self.assertEqual(campaign.classify_failure(124, None), "timeout_censored")
        self.assertEqual(
            campaign.classify_failure(
                137, {"process": {"terminating_signal": 9}}
            ),
            "oom_or_sigkill_censored",
        )

    def test_failed_phase_pair_preserves_both_manifest_checksums(self):
        common = {
            "block_index": 0,
            "evidence_tier": "capacity_stress",
            "evidence_role": "stress",
            "run_kind": "measured",
            "replicate": 0,
            "refinement": 8,
            "scalar_observation_count": 5184,
            "replacement_data_count": 100,
            "method": "spectral_low_rank",
            "spectral_rank": 256,
            "rank_decision_id": "test",
        }
        setup = campaign.CampaignCell(
            order_index=0,
            workload_phase="cold_setup",
            cell_id="setup",
            **common,
        )
        query = campaign.CampaignCell(
            order_index=1,
            workload_phase="query_physical_uq",
            cell_id="query",
            **common,
        )
        summaries = {
            "setup": {
                "status": "timeout_censored",
                "runner_manifest": "/archive/setup.json",
                "runner_manifest_sha256": "a" * 64,
            },
            "query": {
                "status": "oom_or_sigkill_censored",
                "runner_manifest": "/archive/query.json",
                "runner_manifest_sha256": "b" * 64,
            },
        }
        pairs, combined = campaign.pair_results(
            [setup, query], summaries, pathlib.Path("/unused")
        )
        self.assertEqual(combined, [])
        self.assertEqual(pairs[0]["status"], "censored_phase_failure_or_ineligibility")
        self.assertEqual(pairs[0]["cold_setup_runner_manifest_sha256"], "a" * 64)
        self.assertEqual(pairs[0]["query_physical_uq_runner_manifest_sha256"], "b" * 64)


if __name__ == "__main__":
    unittest.main()
