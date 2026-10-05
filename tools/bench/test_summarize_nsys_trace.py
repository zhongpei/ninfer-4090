import unittest

from summarize_nsys_trace import summarize


def event(start, duration, name="kernel", grid="1"):
    return {"Start (ns)": str(start), "Duration (ns)": str(duration), "Name": name,
            "GrdX": grid, "GrdY": "1", "GrdZ": "1", "BlkX": "256", "BlkY": "1",
            "BlkZ": "1", "Reg/Trd": "32", "StcSMem (MB)": "0", "DymSMem (MB)": "0.04",
            "Device": "GPU (0)", "Ctx": "1"}


class TraceSummaryTest(unittest.TestCase):
    def test_decode_window_excludes_prefill_and_does_not_double_count_overlap(self):
        rows = [event(0, 50, "prefill"), event(100, 20), event(110, 30),
                event(150, 10, "memcpy", grid="")]
        result = summarize(rows, (100, 160))
        self.assertEqual(result["gpu_busy_union_ns"], 50)
        self.assertEqual(result["no_traced_gpu_work_ns"], 10)
        self.assertEqual(result["kernel_duration_sum_ns"], 50)
        self.assertEqual([(k["name"], k["calls"]) for k in result["kernels"]], [("kernel", 2)])

    def test_incomplete_window_and_missing_context_isolation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cuts a GPU event"):
            summarize([event(100, 20)], (110, 130))
        rows = [event(100, 20), {**event(100, 20), "Ctx": "2"}]
        with self.assertRaisesRegex(ValueError, "multiple devices/contexts"):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()
