"""Exercise an installed POSIX live adapter using invented weights and PCM."""
import json
import math
from pathlib import Path
import select
import struct
import subprocess
import sys
import tempfile
import unittest


def tone_fixture(pack_path):
    """Route three distinct Fourier bins through the three nonblank CTC tokens."""
    n_params, n_bins, n_mels, n_hidden = 43876, 257, 40, 96
    n_floats = n_params + n_bins * n_mels + 2 * n_mels
    pack = bytearray(16 + 4 * n_floats)
    struct.pack_into("<8sII", pack, 0, b"MERO43V1", 1, n_floats)

    def set_weight(index):
        struct.pack_into("<f", pack, 16 + 4 * index, 1.0)

    bins = (10, 30, 60)
    for mel in range(n_mels):
        fft_bin = bins[mel % len(bins)]
        set_weight(n_params + fft_bin * n_mels + mel)
        set_weight(n_params + n_bins * n_mels + n_mels + mel)
    head_start = n_params - 4 * n_hidden - 4
    for mel in range(len(bins)):
        set_weight(mel * n_mels + mel)
        set_weight(head_start + (mel + 1) * n_hidden + mel)
    pack_path.write_bytes(pack)
    return b"".join(struct.pack("<h", round(16000 * math.sin(2 * math.pi * fft_bin * i / 512)))
                    for fft_bin in bins for i in range(640))


class LiveInput(unittest.TestCase):
    def test_event_before_eof_and_failure_on_partial_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            pack = Path(temp) / "invented.pack"
            pcm = tone_fixture(pack)
            cmd = [self.binary, str(pack), "5000", "0"]
            with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0) as child:
                try:
                    child.stdin.write(pcm[:1])
                    child.stdin.write(pcm[1:])
                    ready, _, _ = select.select([child.stdout], [], [], 5)
                    self.assertTrue(ready, "event missing with PCM input still open")
                    event = json.loads(child.stdout.readline())
                    self.assertEqual(event["event"], "hey_mero")
                    self.assertEqual(event["keyword_id"], 1)
                    self.assertEqual(event["available_after_sample"], 512 + 160 * event["frame_index"])
                    self.assertGreater(event["callback_monotonic_ns"], 0)
                    self.assertIsNone(child.poll())
                    remaining, diagnostic = child.communicate(timeout=5)
                finally:
                    child.kill()
                self.assertEqual(child.returncode, 0, diagnostic)
                self.assertEqual(remaining, b"")
                self.assertIn(b"samples=1920 frames=9 events=1", diagnostic)

            odd = subprocess.run(cmd, input=b"\x00", capture_output=True, timeout=5)
            self.assertNotEqual(odd.returncode, 0)
            self.assertEqual(odd.stdout, b"")
            self.assertIn(b"incomplete PCM16 sample", odd.stderr)
            missing_policy = subprocess.run(cmd[:2], input=b"", capture_output=True, timeout=5)
            self.assertNotEqual(missing_policy.returncode, 0)


if __name__ == "__main__":
    LiveInput.binary, = sys.argv[1:]
    unittest.main(argv=[sys.argv[0]])
