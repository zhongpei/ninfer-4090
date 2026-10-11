"""CPU-only validation of GDN cold-L2 Op A/B selection and admission checks."""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

from tools.bench.run_gdn_op_ab import (
    compare, main, parse_row, positive_list,
)


HEADER = ("state_form,normalization,implementation,dtype,state_dim,qk_heads,"
          "value_heads,tokens,batch,workspace_bytes,logical_bytes,traffic_bytes,"
          "intermediate_traffic_bytes,graph_nodes,cache,execution,warmup,repeat,"
          "median_us,min_us,p95_us,logical_gbps,traffic_gbps\n")


def make_csv(heads=48, tokens=64, median=100.0, p95=110.0):
    fields = ["running", "fused", "public.persistent_fp32", "BF16", "128", "16",
              str(heads), str(tokens), "1", "0", "100000", "100000",
              "0", "1", "cold_l2", "cuda_graph", "2", "50",
              str(median), str(median), str(p95), "1", "1"]
    return HEADER + ",".join(fields) + "\n"


class GdnOpAbTests(unittest.TestCase):
    def test_strict_shape_lists(self):
        self.assertEqual(positive_list("32,48"), [32, 48])
        for bad in ("", "0", "32,32", "abc", "32,"):
            with self.assertRaises(ValueError):
                positive_list(bad)

    def test_exact_op_csv_contract(self):
        v = parse_row(make_csv(), 48, 64)
        self.assertEqual((v["median_us"], v["p95_us"], v["graph_nodes"]),
                         (100.0, 110.0, 1))
        for wrong in (
            make_csv(tokens=128),
            make_csv(median=100, p95=80),
            "SKIP: no CUDA devices\n",
            make_csv() + make_csv(),
        ):
            with self.assertRaises(ValueError):
                parse_row(wrong, 48, 64)

    def test_paired_perf_and_amdahl(self):
        rows = []
        for i in range(3):
            for arm, latency in (("baseline", 100.0), ("candidate", 90.0)):
                rows.append({"pair": i, "arm": arm,
                             "data": {"median_us": latency, "p95_us": latency * 1.1}})
        result = compare(rows, 3, attributed_fraction=.2)
        self.assertGreater(result["paired_speedup_pct"]["median"], 10)
        self.assertTrue(result["performance_screen"])
        self.assertLess(result["estimated_model_gain_pct"], 3)
        self.assertFalse(result["production_qualified"])
        rows[-1]["data"]["median_us"] = 120.0
        self.assertFalse(compare(rows, 3)["performance_screen"])
        with self.assertRaises(ValueError):
            compare(rows[:-1], 3)

    def test_one_op_campaign_is_interleaved_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "fake-gdn-bench"
            executable.write_text("mock")
            output = root / "out"
            arm_env = []

            def fake_run(command, **kwargs):
                self.assertIn("--running", command)
                self.assertIn("--csv", command)
                self.assertEqual(kwargs["env"]["NINFER_DEVICE_ROUTE_MODE"], "off")
                self.assertEqual(kwargs["env"]["NINFER_GDN_TWO_STAGE"], "0")
                candidate = kwargs["env"]["NINFER_GDN_EXACT_PREFETCH"] == "1"
                arm_env.append(candidate)
                return CompletedProcess(command, 0, make_csv(median=90 if candidate else 100,
                                                             p95=100 if candidate else 110), "")

            with patch("tools.bench.run_gdn_op_ab.subprocess.run", side_effect=fake_run):
                self.assertEqual(main(["--exe", str(executable), "--out", str(output),
                                       "--heads", "48", "--tokens", "64",
                                       "--pairs", "3", "--warmup", "1",
                                       "--repeat", "2", "--flush-mib", "1"]), 0)
            self.assertEqual(arm_env, [False, True, True, False, False, True])
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["schema"], "ninfer.gdn-op-ab.v1")
            self.assertEqual(len(result["comparisons"]), 1)
            self.assertTrue(result["comparisons"][0]["performance_screen"])
            self.assertEqual(len((output / "records.jsonl").read_text().splitlines()), 6)
            self.assertIn("µs", (output / "summary.md").read_text())
            self.assertFalse(result["production_qualified"])


if __name__ == "__main__":
    unittest.main()
