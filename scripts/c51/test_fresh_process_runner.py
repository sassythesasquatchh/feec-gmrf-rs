#!/usr/bin/env python3

import importlib.util
import csv
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
    def write_resource_rows(self, path, rows):
        fieldnames = [
            "resource_scope",
            "workload_phase",
            "method",
            "refinement",
            "observation_class",
            "sensor_count",
            "repeated_data_count",
            "factor_external_peak_rss_bytes_status",
            "factor_external_peak_rss_bytes_value",
            "factor_external_peak_rss_bytes_reason",
            "query_external_peak_rss_bytes_status",
            "query_external_peak_rss_bytes_value",
            "query_external_peak_rss_bytes_reason",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def resource_selection(self, phase="cold_setup", method="whitney_compact"):
        return {
            "method": method,
            "refinement": 2,
            "observation_class": "local_physical_b",
            "sensor_target_kind": "count",
            "sensor_target_value": 4,
            "repeated_data_count": 1,
            "spectral_rank": None,
            "workload_phase": phase,
        }

    def resource_row(self, phase="cold_setup", method="whitney_compact"):
        return {
            "resource_scope": "method_phase",
            "workload_phase": phase,
            "method": method,
            "refinement": "2",
            "observation_class": "local_physical_b",
            "sensor_count": "4",
            "repeated_data_count": "1",
            "factor_external_peak_rss_bytes_status": "not_recorded",
            "factor_external_peak_rss_bytes_value": "",
            "factor_external_peak_rss_bytes_reason": "",
            "query_external_peak_rss_bytes_status": "not_recorded",
            "query_external_peak_rss_bytes_value": "",
            "query_external_peak_rss_bytes_reason": "",
        }

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

    def test_method_phase_rss_attaches_to_exactly_one_matching_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "rows.csv"
            self.write_resource_rows(path, [self.resource_row()])
            result = runner.attach_peak_rss_to_resource_row(
                path,
                self.resource_selection(),
                runner.measured(4096, "bytes", "test_peak"),
            )
            with path.open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["resource_scope"], "method_phase")
        self.assertEqual(row["factor_external_peak_rss_bytes_status"], "measured")
        self.assertEqual(row["factor_external_peak_rss_bytes_value"], "4096")
        self.assertEqual(row["query_external_peak_rss_bytes_status"], "not_recorded")

    def test_query_full_uq_rss_attaches_only_to_query_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "rows.csv"
            self.write_resource_rows(path, [self.resource_row(phase="query_full_uq")])
            runner.attach_peak_rss_to_resource_row(
                path,
                self.resource_selection(phase="query_full_uq"),
                runner.measured(8192, "bytes", "test_peak"),
            )
            with path.open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
        self.assertEqual(row["factor_external_peak_rss_bytes_status"], "not_recorded")
        self.assertEqual(row["query_external_peak_rss_bytes_status"], "measured")
        self.assertEqual(row["query_external_peak_rss_bytes_value"], "8192")

    def test_row_attribution_refuses_multiple_rows_and_method_phase_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "rows.csv"
            row = self.resource_row()
            self.write_resource_rows(path, [row, row])
            with self.assertRaisesRegex(ValueError, "exactly one CSV row"):
                runner.attach_peak_rss_to_resource_row(
                    path,
                    self.resource_selection(),
                    runner.measured(4096, "bytes", "test_peak"),
                )
            self.write_resource_rows(path, [self.resource_row(method="fm_direct")])
            with self.assertRaisesRegex(ValueError, "does not match selection"):
                runner.attach_peak_rss_to_resource_row(
                    path,
                    self.resource_selection(),
                    runner.measured(4096, "bytes", "test_peak"),
                )
            self.write_resource_rows(path, [self.resource_row(phase="query_full_uq")])
            with self.assertRaisesRegex(ValueError, "does not match selection"):
                runner.attach_peak_rss_to_resource_row(
                    path,
                    self.resource_selection(),
                    runner.measured(4096, "bytes", "test_peak"),
                )


if __name__ == "__main__":
    unittest.main()
