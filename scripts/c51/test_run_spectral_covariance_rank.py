#!/usr/bin/env python3

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("run_spectral_covariance_rank.py")
SPEC = importlib.util.spec_from_file_location("run_spectral_covariance_rank", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class SpectralRankRunnerTests(unittest.TestCase):
    @staticmethod
    def fake_build(return_code=0):
        return {
            "command": ["cargo", "build"],
            "command_text": "cargo build",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "ended_at_utc": "2026-01-01T00:00:01Z",
            "duration_seconds": 1.0,
            "raw_return_code": return_code,
            "shell_exit_status": return_code,
            "terminating_signal": None,
            "terminating_signal_name": None,
            "launch_error": None,
            "outside_scientific_measurement": True,
        }

    def fake_command(self, mode, **values):
        command = [
            sys.executable,
            str(MODULE_PATH),
            "--refinement",
            "1",
            "--maximum-rank",
            "1",
            "--output",
            str(values.pop("output", "unused")),
            "--_fake-child",
            mode,
        ]
        for name, value in values.items():
            command.extend([f"--_fake-{name.replace('_', '-')}", str(value)])
        return command

    def monitor(
        self,
        root,
        command,
        wall=2.0,
        rss=1024**3,
        disk=1024**3,
        sampled_rss=16 * 1024**2,
    ):
        # macOS process enumeration is sandbox-restricted in the test host.
        # The process-tree selection itself is tested independently below; the
        # fake children exercise launch, poll, disk, time, and group termination.
        with mock.patch.object(
            runner,
            "sampled_process_tree_rss",
            return_value=(sampled_rss, [12345]),
        ):
            return runner.run_monitored_process(
                command=command,
                cwd=pathlib.Path(root),
                attempt_output=pathlib.Path(root),
                environment=os.environ.copy(),
                wall_time_limit_seconds=wall,
                rss_cap_bytes=rss,
                disk_cap_bytes=disk,
                sample_interval_seconds=0.01,
            )

    def test_scientific_command_is_exact_rank_discovery_cli(self):
        command = runner.scientific_command(
            pathlib.Path("/tmp/bin"),
            "publication-discovery-v1",
            8,
            2048,
            pathlib.Path("/tmp/out/study"),
        )
        self.assertEqual(
            command,
            [
                "/tmp/bin",
                "--profile",
                "publication-discovery-v1",
                "--refinement",
                "8",
                "--maximum-rank",
                "2048",
                "--output",
                "/tmp/out/study",
            ],
        )

    def test_rank_extension_command_has_immutable_profile_without_maximum_override(self):
        command = runner.scientific_command(
            pathlib.Path("/tmp/bin"),
            "rank-extension-v1-n10",
            10,
            None,
            pathlib.Path("/tmp/out/study"),
        )
        self.assertEqual(
            command,
            [
                "/tmp/bin",
                "--profile",
                "rank-extension-v1-n10",
                "--refinement",
                "10",
                "--output",
                "/tmp/out/study",
            ],
        )

    def test_rank_extension_profile_requires_matching_refinement_and_no_maximum(self):
        resolved = runner.resolve_profile("rank-extension-v1-n8", 8, None)
        self.assertEqual(resolved["maximum_rank"], 3032)
        self.assertEqual(
            resolved["rank_grid"], [2048, 2304, 2560, 2816, 3000, 3032]
        )
        self.assertEqual(resolved["expected_state_dimension"], 3032)
        self.assertEqual(resolved["evidence_tier"], "rank_extension")
        self.assertTrue(resolved["post_primary"])
        with self.assertRaisesRegex(ValueError, "requires refinement 8"):
            runner.resolve_profile("rank-extension-v1-n8", 10, None)
        with self.assertRaisesRegex(ValueError, "rejects --maximum-rank"):
            runner.resolve_profile("rank-extension-v1-n8", 8, 3032)

    def test_m4096_n10_rank_extension_is_immutable_and_accuracy_only(self):
        profile = "high-observation-m4096-rank-extension-v1-n10-r4608"
        resolved = runner.resolve_profile(profile, 10, None)
        self.assertEqual(resolved["maximum_rank"], 4608)
        self.assertEqual(resolved["rank_grid"], [4096, 4608])
        self.assertEqual(resolved["expected_state_dimension"], 6130)
        self.assertEqual(
            resolved["evidence_tier"],
            "high_observation_confirmatory_rank_extension",
        )
        self.assertFalse(resolved["rank_decision_required"])
        command = runner.scientific_command(
            pathlib.Path("/tmp/bin"),
            profile,
            10,
            None,
            pathlib.Path("/tmp/out/study"),
        )
        self.assertNotIn("--maximum-rank", command)
        with self.assertRaisesRegex(ValueError, "requires refinement 10"):
            runner.resolve_profile(profile, 8, None)
        with self.assertRaisesRegex(ValueError, "rejects --maximum-rank"):
            runner.resolve_profile(profile, 10, 4608)

    def test_high_observation_profile_reaches_n10_full_rank_and_exports_decision(self):
        profile = "high-observation-primary-m2048-v1-n10"
        resolved = runner.resolve_profile(profile, 10, None)
        self.assertEqual(resolved["maximum_rank"], 6130)
        self.assertEqual(resolved["rank_grid"][-3:], [5632, 6000, 6130])
        self.assertTrue(resolved["rank_decision_required"])
        command = runner.scientific_command(
            pathlib.Path("/tmp/bin"),
            profile,
            10,
            None,
            pathlib.Path("/tmp/out/study"),
            pathlib.Path("/tmp/perf0"),
        )
        self.assertNotIn("--maximum-rank", command)
        self.assertEqual(command[-4:], [
            "--perf0-archive",
            "/tmp/perf0",
            "--rank-decision-output",
            "/tmp/out/study/rank-decision.json",
        ])
        with self.assertRaisesRegex(ValueError, "requires refinement 10"):
            runner.resolve_profile(profile, 8, None)

    def test_replay_profile_stops_at_cached_rank_and_passes_workspace(self):
        for profile in [
            "high-observation-m2048-replay-v1-n10-r4096",
            "high-observation-m4096-replay-v1-n10-r4096",
        ]:
            resolved = runner.resolve_profile(profile, 10, None)
            self.assertEqual(resolved["maximum_rank"], 4096)
            self.assertEqual(resolved["rank_grid"][-3:], [3072, 3584, 4096])
            self.assertFalse(resolved["rank_decision_required"])
            workspace = pathlib.Path("/tmp/cached-basis")
            command = runner.scientific_command(
                pathlib.Path("/tmp/bin"),
                profile,
                10,
                None,
                pathlib.Path("/tmp/out/study"),
                basis_workspace=workspace,
            )
            self.assertEqual(command[-2:], ["--basis-workspace", str(workspace)])
            with self.assertRaisesRegex(ValueError, "requires refinement 10"):
                runner.resolve_profile(profile, 8, None)

    def test_build_is_locked_release_example(self):
        self.assertEqual(
            runner.build_command("cargo"),
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "feg-case-studies",
                "--features",
                "experimental,external-solvers",
                "--example",
                "c51_spectral_covariance_rank",
            ],
        )

    def test_process_tree_includes_descendants_and_reparented_group_members(self):
        sample = runner.ProcessSample
        snapshot = {
            100: sample(100, 1, 100, 1),
            101: sample(101, 100, 100, 2),
            102: sample(102, 101, 100, 3),
            103: sample(103, 1, 100, 4),
            200: sample(200, 1, 200, 5),
        }
        self.assertEqual(runner.process_tree_pids(100, snapshot), {100, 101, 102, 103})

    def test_macos_ps_snapshot_converts_kib_and_ignores_malformed_rows(self):
        snapshot = runner.parse_ps_snapshot(
            " 100 1 100 2048\n 101 100 100 512\nnot a process row\n"
        )
        self.assertEqual(snapshot[100].rss_bytes, 2048 * 1024)
        self.assertEqual(snapshot[101].ppid, 100)
        self.assertEqual(snapshot[101].process_group, 100)

    def test_successful_fake_child_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            marker = root / "marker"
            result = self.monitor(
                root, self.fake_command("success", path=marker, output=root / "unused")
            )
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.raw_return_code, 0)
        self.assertGreaterEqual(result.samples, 1)
        self.assertGreater(result.peak_sampled_process_tree_rss_bytes, 0)

    def test_wall_time_cap_terminates_fresh_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result = self.monitor(
                root,
                self.fake_command("sleep", seconds=2.0, output=root / "unused"),
                wall=0.05,
            )
        self.assertEqual(result.outcome, "wall_time_cap_exceeded")
        self.assertIsNotNone(result.termination)
        self.assertNotEqual(result.raw_return_code, 0)

    def test_sampled_rss_cap_terminates_fake_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result = self.monitor(
                root,
                self.fake_command("sleep", seconds=2.0, output=root / "unused"),
                rss=1,
            )
        self.assertEqual(result.outcome, "rss_cap_exceeded")
        self.assertGreater(result.peak_sampled_process_tree_rss_bytes, 1)

    def test_disk_cap_terminates_fake_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            payload = root / "payload.bin"
            result = self.monitor(
                root,
                self.fake_command(
                    "disk",
                    path=payload,
                    bytes=256 * 1024,
                    seconds=2.0,
                    output=root / "unused",
                ),
                disk=64 * 1024,
            )
        self.assertEqual(result.outcome, "disk_cap_exceeded")
        self.assertGreater(result.maximum_output_directory_bytes, 64 * 1024)

    def test_refuses_to_overwrite_existing_attempt_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            marker = root / "keep"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                runner.reserve_attempt_directory(root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_dry_run_has_caps_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "not-created"
            args = runner.parse_args(
                [
                    "--repository",
                    str(pathlib.Path(__file__).resolve().parents[2]),
                    "--refinement",
                    "8",
                    "--maximum-rank",
                    "2048",
                    "--output",
                    str(output),
                    "--dry-run",
                ]
            )
            plan = runner.dry_run_plan(args, args.repository.resolve())
        self.assertEqual(plan["maximum_rank"], 2048)
        self.assertEqual(plan["refinement"], 8)
        self.assertIn("sampled_process_tree_rss_bytes", plan["caps"])
        self.assertFalse(output.exists())

    def test_rank_extension_dry_run_records_post_primary_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "not-created"
            args = runner.parse_args(
                [
                    "--repository",
                    str(pathlib.Path(__file__).resolve().parents[2]),
                    "--profile",
                    "rank-extension-v1-n10",
                    "--refinement",
                    "10",
                    "--output",
                    str(output),
                    "--dry-run",
                ]
            )
            plan = runner.dry_run_plan(args, args.repository.resolve())
        self.assertEqual(plan["maximum_rank"], 4096)
        self.assertEqual(plan["rank_grid"], [2048, 2560, 3072, 3584, 4096])
        self.assertEqual(plan["expected_state_dimension"], 6130)
        self.assertEqual(plan["evidence_tier"], "rank_extension")
        self.assertTrue(plan["post_primary"])
        self.assertNotIn("--maximum-rank", plan["scientific_command"])
        self.assertFalse(output.exists())

    def test_rank_extension_attempt_manifest_is_explicitly_timing_ineligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = runner.parse_args(
                [
                    "--profile",
                    "rank-extension-v1-n8",
                    "--refinement",
                    "8",
                    "--output",
                    str(root / "attempt"),
                ]
            )
            profile = runner.resolve_profile(args.profile, args.refinement, args.maximum_rank)
            manifest = runner.initial_manifest(
                args,
                pathlib.Path(root),
                root / "attempt",
                root / "binary",
                {},
                {},
                {},
                {},
                profile,
            )
        self.assertEqual(manifest["rank_grid"], [2048, 2304, 2560, 2816, 3000, 3032])
        self.assertEqual(manifest["requested_maximum_rank"], None)
        self.assertEqual(manifest["evidence_tier"], "rank_extension")
        self.assertFalse(manifest["method_performance_timing_eligible"])
        self.assertTrue(manifest["shared_maximum_basis"])
        self.assertIn("post-primary", manifest["post_primary_rationale"])

    def test_atomic_manifest_contains_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / runner.MANIFEST_NAME
            runner.atomic_write_json(path, {"outcome": "running", "rank": 2048})
            contents = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(contents, {"outcome": "running", "rank": 2048})

    def test_build_failure_still_finalizes_attempt_manifest_without_cargo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repo"
            repository.mkdir()
            (repository / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            output = root / "attempt"
            args = runner.parse_args(
                [
                    "--repository",
                    str(repository),
                    "--refinement",
                    "8",
                    "--maximum-rank",
                    "2048",
                    "--output",
                    str(output),
                ]
            )
            with (
                mock.patch.object(runner, "source_metadata", return_value={}),
                mock.patch.object(runner, "tool_metadata", return_value={}),
                mock.patch.object(runner, "build_release", return_value=self.fake_build(17)),
            ):
                status = runner.run_attempt(args)
            manifest = json.loads(
                (output / runner.MANIFEST_NAME).read_text(encoding="utf-8")
            )
        self.assertEqual(status, 17)
        self.assertEqual(manifest["outcome"], "build_failed")
        self.assertIsNotNone(manifest["timestamps"]["attempt_ended_at_utc"])
        self.assertGreater(manifest["maximum_output_directory_bytes"], 0)

    def test_cap_failure_finalizes_manifest_and_never_invokes_cargo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repository = root / "repo"
            repository.mkdir()
            (repository / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            binary = root / "fake-scientific-child"
            binary.write_text(
                "#!/usr/bin/env python3\nimport time\ntime.sleep(2.0)\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            output = root / "attempt"
            args = runner.parse_args(
                [
                    "--repository",
                    str(repository),
                    "--refinement",
                    "8",
                    "--maximum-rank",
                    "2048",
                    "--output",
                    str(output),
                    "--rss-cap-bytes",
                    "1",
                    "--sample-interval-seconds",
                    "0.01",
                ]
            )
            with (
                mock.patch.object(runner, "source_metadata", return_value={}),
                mock.patch.object(runner, "tool_metadata", return_value={}),
                mock.patch.object(runner, "release_binary", return_value=binary),
                mock.patch.object(runner, "build_release", return_value=self.fake_build()),
                mock.patch.object(
                    runner,
                    "sampled_process_tree_rss",
                    return_value=(4096, [12345]),
                ),
            ):
                status = runner.run_attempt(args)
            manifest = json.loads(
                (output / runner.MANIFEST_NAME).read_text(encoding="utf-8")
            )
        self.assertNotEqual(status, 0)
        self.assertEqual(manifest["outcome"], "rss_cap_exceeded")
        self.assertEqual(manifest["peak_sampled_process_tree_rss_bytes"], 4096)
        self.assertIsNotNone(manifest["timestamps"]["attempt_ended_at_utc"])


if __name__ == "__main__":
    unittest.main()
