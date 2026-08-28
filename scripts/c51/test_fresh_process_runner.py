#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("fresh_process_runner.py")
SPEC = importlib.util.spec_from_file_location("fresh_process_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FreshProcessRunnerTests(unittest.TestCase):
    def test_gnu_time_parser_converts_kbytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "time.txt"
            path.write_text(
                "\n".join(
                    [
                        "User time (seconds): 1.25",
                        "System time (seconds): 0.50",
                        "Maximum resident set size (kbytes): 1234",
                    ]
                ),
                encoding="utf-8",
            )
            result = runner.parse_time_measurements("gnu", path)
        self.assertEqual(result["peak_rss_bytes"]["value"], 1234 * 1024)
        self.assertEqual(result["user_cpu_seconds"]["value"], 1.25)

    def test_macos_time_parser_keeps_byte_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "time.txt"
            path.write_text(
                "0.50 real 0.25 user 0.10 sys\n  987654 maximum resident set size\n",
                encoding="utf-8",
            )
            result = runner.parse_time_measurements("macos_bsd", path)
        self.assertEqual(result["peak_rss_bytes"]["value"], 987654)
        self.assertEqual(result["system_cpu_seconds"]["value"], 0.10)

    def test_signal_exit_is_shell_normalized(self):
        status, signal_number = runner.normalized_exit_status(-9)
        self.assertEqual(status, 137)
        self.assertEqual(signal_number, 9)

    def test_peak_selection_names_fallback_source(self):
        cgroup = runner.unavailable("not delegated", "cgroup")
        external_time = runner.measured(1000, "bytes", "gnu_time")
        selected = runner.select_peak_rss(cgroup, external_time)
        self.assertEqual(selected["source"], "gnu_time")
        self.assertIn("cgroup-v2", selected["selection_rule"])
        self.assertEqual(selected["cgroup_unavailable_reason"], "not delegated")

    def test_row_attribution_is_not_implied_by_peak_selection(self):
        selected = runner.select_peak_rss(
            runner.unavailable("not delegated"),
            runner.measured(2048, "bytes", "macos_bsd_time_maximum_rss"),
        )
        self.assertEqual(selected["status"], "measured")
        self.assertNotIn("method", selected)

    def test_thread_environment_is_complete(self):
        environment = runner.thread_environment(3)
        self.assertEqual(set(environment), set(runner.THREAD_ENV_NAMES))
        self.assertEqual(set(environment.values()), {"3"})
        with self.assertRaises(ValueError):
            runner.thread_environment(0)


if __name__ == "__main__":
    unittest.main()
