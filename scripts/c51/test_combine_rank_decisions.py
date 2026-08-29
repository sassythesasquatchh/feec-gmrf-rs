#!/usr/bin/env python3

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "combine_rank_decisions.py"
SPEC = importlib.util.spec_from_file_location("combine_rank_decisions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
combiner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = combiner
SPEC.loader.exec_module(combiner)


class CombineRankDecisionsTests(unittest.TestCase):
    def decision(self, refinement: int, perf0_hash: str = "a" * 64):
        return {
            "refinement": refinement,
            "scalar_observation_count": 2048,
            "alpha": 2,
            "relative_tolerance": 0.05,
            "observation_design_id": "frozen",
            "exact_observation_gp": {
                "fixture_artifact_sha256": perf0_hash,
            },
        }

    def write(self, path: pathlib.Path, decision: dict):
        path.write_text(
            json.dumps(
                {
                    "schema_version": combiner.SCHEMA_VERSION,
                    "profile": combiner.PROFILE,
                    "decisions": [decision],
                }
            ),
            encoding="utf-8",
        )

    def test_combines_in_key_order_and_rejects_duplicate_or_mixed_perf0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            n10 = root / "n10.json"
            n8 = root / "n8.json"
            self.write(n10, self.decision(10))
            self.write(n8, self.decision(8))
            combined = combiner.combine([n10, n8])
            self.assertEqual(
                [entry["refinement"] for entry in combined["decisions"]], [8, 10]
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                combiner.combine([n8, n8])
            mixed = copy.deepcopy(self.decision(10))
            mixed["exact_observation_gp"]["fixture_artifact_sha256"] = "b" * 64
            self.write(n10, mixed)
            with self.assertRaisesRegex(ValueError, "one immutable PERF-0"):
                combiner.combine([n8, n10])


if __name__ == "__main__":
    unittest.main()
