"""Real subprocess regressions for bounded capture; stdlib only."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_log import BoundedLog


class BoundedLogTests(unittest.TestCase):
    def test_two_megabyte_burst_has_exact_prefix_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "burst.log"
            with path.open("xb") as stream:
                proc = subprocess.Popen([sys.executable, "-c", "import os; os.write(1, b'prefix:'+b'x'*(2*1024*1024))"], stdout=subprocess.PIPE)
                capture = BoundedLog(proc.stdout, stream, 1024)
                try:
                    deadline = time.monotonic() + 10
                    while proc.poll() is None:
                        self.assertLess(time.monotonic(), deadline)
                        self.assertLessEqual(capture.pump(.05), capture.PUMP_BYTES)
                        self.assertLessEqual(path.stat().st_size, 1024)
                    proc.wait(timeout=1)
                    capture.finish()
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=1)
                    capture.finish()
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(path.read_bytes(), b"prefix:" + b"x" * (1024 - 7))
            self.assertEqual(capture.written, 1024)
            self.assertEqual(capture.observed_bytes, 7 + 2 * 1024 * 1024)
            self.assertTrue(capture.truncated)
            self.assertTrue(capture.eof)
            self.assertFalse(capture.drain_timed_out)

    def test_fast_exit_burst_caught_during_final_drain(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fast.log"
            with path.open("xb") as stream:
                proc = subprocess.Popen([sys.executable, "-c", "import os; os.write(1, b'x'*4096)"], stdout=subprocess.PIPE)
                capture = BoundedLog(proc.stdout, stream, 1024)
                proc.wait(timeout=5)
                capture.finish()
            self.assertEqual(path.stat().st_size, 1024)
            self.assertTrue(capture.truncated)
            self.assertEqual(capture.observed_bytes, 4096)
            self.assertTrue(capture.eof)

    def test_normal_log_and_exit_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "normal.log"
            with path.open("xb") as stream:
                proc = subprocess.Popen([sys.executable, "-c", "import os; os.write(1, b'hello\\n'); os.write(2, b'error\\n'); raise SystemExit(7)"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                capture = BoundedLog(proc.stdout, stream, 1024)
                proc.wait(timeout=5)
                capture.finish()
                self.assertFalse(stream.closed)
                capture.finish()  # Finalization is safe to repeat.
            self.assertEqual(proc.returncode, 7)
            self.assertEqual(path.read_bytes(), b"hello\nerror\n")
            self.assertFalse(capture.truncated)
            self.assertTrue(capture.eof)
            self.assertTrue(capture.closed)
            self.assertTrue(proc.stdout.closed)

    def test_exact_limit_is_not_truncation(self):
        with tempfile.TemporaryDirectory() as temp, (Path(temp) / "exact.log").open("xb") as stream:
            proc = subprocess.Popen([sys.executable, "-c", "import os; os.write(1, b'x'*1024)"], stdout=subprocess.PIPE)
            capture = BoundedLog(proc.stdout, stream, 1024)
            proc.wait(timeout=5)
            capture.finish()
            self.assertEqual(capture.written, 1024)
            self.assertFalse(capture.truncated)

    def test_unclosed_pipe_has_bounded_finalization(self):
        with tempfile.TemporaryDirectory() as temp, (Path(temp) / "held.log").open("xb") as stream:
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.PIPE)
            capture = BoundedLog(proc.stdout, stream, 1024)
            try:
                started = time.monotonic()
                capture.finish(timeout=.02)
                self.assertLess(time.monotonic() - started, 1)
                self.assertTrue(capture.drain_timed_out)
                self.assertFalse(capture.eof)
                self.assertTrue(capture.closed)
            finally:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
