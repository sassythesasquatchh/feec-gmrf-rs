#!/usr/bin/env python3

import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).parent / "docker" / "run_exact_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_exact_campaign", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


class DockerExactCampaignTests(unittest.TestCase):
    def test_dry_run_prepares_native_dependencies_and_delegates_exact_profile(self):
        commands = []

        def record(command, *, environment=None):
            commands.append((list(command), environment))

        with (
            mock.patch.object(pathlib.Path, "is_file", return_value=True),
            mock.patch.object(os, "access", return_value=True),
            mock.patch.object(campaign, "slepc_root", return_value=pathlib.Path("/slepc")),
            mock.patch.object(
                campaign, "petsc_root", return_value=pathlib.Path("/petsc")
            ),
            mock.patch.object(campaign, "run", side_effect=record),
        ):
            status = campaign.main(
                [
                    "--rank-decisions",
                    "out/c51/decisions.json",
                    "--output",
                    "out/c51/linux",
                    "--dry-run",
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn(
            ["make", "-C", "feec/petsc-solver", "clean", "all"],
            [command for command, _ in commands],
        )
        delegated = commands[-1]
        self.assertEqual(delegated[0][0:2], ["python3", "scripts/c51/run_juq_performance.py"])
        self.assertIn("scripts/c51/juq-exact-performance-v1.json", delegated[0])
        self.assertEqual(delegated[0][-1], "--dry-run")
        self.assertEqual(delegated[1]["SLEPC_DIR"], "/slepc")
        self.assertEqual(delegated[1]["PETSC_DIR"], "/petsc")
        self.assertTrue(delegated[1]["PETSC_SOLVER_PATH"].endswith("/feec/petsc-solver/ghiep.out"))

    def test_missing_writable_cgroup_is_a_hard_error(self):
        with (
            mock.patch.object(pathlib.Path, "is_file", return_value=True),
            mock.patch.object(os, "access", return_value=False),
            mock.patch.object(campaign, "run") as run,
        ):
            status = campaign.main(
                ["--rank-decisions", "decisions.json", "--output", "out/c51/linux"]
            )
        self.assertEqual(status, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
