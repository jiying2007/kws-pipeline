#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def require(condition: bool, message: str = "board benchmark contract mismatch") -> None:
    if not condition:
        raise AssertionError(message)


def verify_statistics(root: pathlib.Path) -> None:
    # Pure arrays only: never links or loads a model/runtime/benchmark runner.
    source = root / "statistics.c"
    executable = root / "statistics"
    source.write_text(r'''#include "kws_bench_statistics.h"
#include <stdio.h>
#define CHECK(expression) do { if (!(expression)) { \
  fprintf(stderr, "failed at line %d: %s\n", __LINE__, #expression); \
  return 1; } } while (0)
int main(void) {
  const double singleton[] = {7.0};
  const double three[] = {1.0, 2.0, 9000.0};
  const double ties[] = {0.0, 0.0, 0.0, 5.0, 5.0};
  double hundred_one[101];
  double tail[101];
  for (size_t i = 0u; i < 101u; ++i) {
    hundred_one[i] = (double)(i + 1u);
    tail[i] = i < 99u ? 1.0 : (double)(i - 98u) * 5000.0;
  }
  CHECK(kws_bench_percentile_nearest_rank(NULL, 0u, 0.99) == 0.0);
  CHECK(kws_bench_percentile_nearest_rank(singleton, 1u, 0.99) == 7.0);
  CHECK(kws_bench_percentile_nearest_rank(three, 3u, 0.0) == 1.0);
  CHECK(kws_bench_percentile_nearest_rank(three, 3u, 1.0) == 9000.0);
  CHECK(kws_bench_percentile_nearest_rank(three, 3u, 0.50) == 2.0);
  CHECK(kws_bench_percentile_nearest_rank(three, 3u, 0.95) == 9000.0);
  CHECK(kws_bench_percentile_nearest_rank(three, 3u, 0.99) == 9000.0);
  CHECK(kws_bench_percentile_nearest_rank(hundred_one, 100u, 0.99) == 99.0);
  CHECK(kws_bench_percentile_nearest_rank(hundred_one, 101u, 0.50) == 51.0);
  CHECK(kws_bench_percentile_nearest_rank(hundred_one, 101u, 0.95) == 96.0);
  CHECK(kws_bench_percentile_nearest_rank(hundred_one, 101u, 0.99) == 100.0);
  CHECK(kws_bench_percentile_nearest_rank(tail, 101u, 0.99) == 5000.0);
  CHECK(kws_bench_percentile_nearest_rank(ties, 5u, 0.50) == 0.0);
  CHECK(kws_bench_percentile_nearest_rank(ties, 5u, 0.99) == 5.0);
  return 0;
}
''', encoding="utf-8")
    command = shlex.split(os.environ.get("CC", "cc"))
    subprocess.run(
        command + ["-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                   "-Wconversion", "-Wshadow", "-Wcast-qual", "-I", str(ROOT / "tools"),
                   str(source), "-lm", "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=pathlib.Path)
    parser.add_argument("--statistics-only", action="store_true")
    args = parser.parse_args()

    if not args.statistics_only and args.runner is None:
        parser.error("--runner is required unless --statistics-only is selected")
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        verify_statistics(root)
        if args.statistics_only:
            print("test_board_bench: nearest-rank statistics ok (no model execution)")
            return 0
        from qualification_fixture import (
            sha256_file, write_model, write_pack, write_tokens, write_wav,
        )
        tokens = root / "tokens.txt"
        model = root / "model.kwm"
        pack = root / "keywords.kwk"
        wav = root / "audio.wav"
        _, fingerprint = write_tokens(tokens)
        model_bytes = write_model(model, fingerprint)
        pack_bytes = write_pack(pack, fingerprint)
        write_wav(wav, seconds=1)

        completed = subprocess.run(
            [str(args.runner), str(model), str(pack), str(wav), "2"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        result = json.loads(completed.stdout)
        require(result["schema_version"] == 1)
        require(result["runner_sha256"] == sha256_file(args.runner))
        require(result["model_sha256"] == sha256_file(model))
        require(result["keyword_pack_sha256"] == sha256_file(pack))
        require(result["audio_sha256"] == sha256_file(wav))
        require(result["sample_rate_hz"] == 16000)
        require(result["frame_length_samples"] == 400)
        require(result["frame_hop_samples"] == 320)
        require(result["final_block_samples"] == 320)
        require(result["timing_unit"] == "input-call")
        require(result["deadline_basis"] == "nominal-block")
        require(result["tail_policy"] == "short-final-call-included-no-padding")
        require(result["percentile_estimator"] == "nearest-rank-ceil-v1")
        require(result["block_samples"] == 320)
        require(result["block_deadline_us"] == 20000.0)
        require(result["audio_seconds"] == 1.0)
        require(result["repeats"] == 2)
        require(result["blocks"] == 100)
        require(result["model_bytes"] == model_bytes)
        require(result["keyword_pack_bytes"] == pack_bytes)
        require(result["arena_bytes"] > 0)
        require(result["total_process_us"] >= 0.0)
        require(result["mean_process_us"] >= 0.0)
        require(0.0 <= result["p50_process_us"] <= result["p95_process_us"])
        require(result["p95_process_us"] <= result["p99_process_us"])
        require(result["p99_process_us"] <= result["max_process_us"])
        require(result["rtf"] >= 0.0)
        require(result["p99_headroom"] >= 0.0)

    print("test_board_bench: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
