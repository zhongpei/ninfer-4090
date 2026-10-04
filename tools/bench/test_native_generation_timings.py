"""CPU option checks for the native public generation timing sidecar."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("NINFER_CALIBRATION_NATIVE_EXE"),
                     "set NINFER_CALIBRATION_NATIVE_EXE to the built native benchmark")
class NativeGenerationTimingsTest(unittest.TestCase):
    def run_native(self, *args):
        return subprocess.run([os.environ["NINFER_CALIBRATION_NATIVE_EXE"], *args],
                              capture_output=True, text=True, timeout=30)

    def assert_rejected(self, args, message):
        result = self.run_native(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertNotIn("model must be", result.stderr)

    def test_help_states_distinct_timing_scopes(self):
        result = self.run_native("--help")
        self.assertEqual(result.returncode, 0)
        for message in ("--timing-output PATH", "latency exclude Engine startup",
                        "first-token time", "includes prompt preparation", "--session --model PATH",
                        "uploads weights once", "KV, cache and graphs are not shared"):
            self.assertIn(message, result.stdout)

    def test_identity_only_rejects_timing_without_loading_model(self):
        self.assert_rejected(["--identity-only", "--timing-output", "unused.json"],
                             "identity-only excludes timing-output")

    def test_same_output_path_rejects_normalized_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            Path(directory, "nested").mkdir()
            alias = Path(directory) / "nested" / ".." / "report.json"
            self.assert_rejected(["--model", "absent", "--prompt-file", "absent",
                                  "--output", str(report), "--timing-output", str(alias)],
                                 "output and timing-output must be separate paths")
            self.assertFalse(report.exists())

    def test_invalid_sidecar_destination_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.json"
            existing.write_text("retain me", encoding="utf-8")
            for path, message in (
                    (existing, "timing-output already exists"),
                    (root / "missing" / "timings.json", "existing parent directory"),
                    ("", "existing parent directory")):
                with self.subTest(path=path):
                    self.assert_rejected(["--model", "absent", "--prompt-file", "absent",
                                          "--output", str(root / "report.json"),
                                          "--timing-output", str(path)], message)
            self.assertEqual(existing.read_text(encoding="utf-8"), "retain me")
            self.assertFalse((root / "report.json").exists())

    def test_session_startup_rejects_job_options_before_model_loading(self):
        for args, message in (
                (["--session", "--identity-only"], "session startup accepts only"),
                (["--session", "--draft-tokens", "7"], "session startup accepts only"),
                (["--session", "--session"], "duplicate session flag")):
            with self.subTest(args=args):
                self.assert_rejected(args, message)


if __name__ == "__main__":
    unittest.main()
