#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import tempfile
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from qualification_metrics import (BOARD_SEMANTICS, board_wav_stats,
                                   validate_board, require_board_raw_binding)


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



def write_pcm_fixture(path: pathlib.Path, samples: int) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * samples)


def board_summary_fixture(audio: pathlib.Path, repeats: int = 2) -> dict:
    """Invented timing metadata, never measured model or product evidence."""
    geometry = board_wav_stats(audio)
    blocks = geometry["blocks_per_repeat"] * repeats
    total_us = 1000.0 * blocks
    return {
        "schema_version": 2, **BOARD_SEMANTICS,
        **{key: geometry[key] for key in ("audio_samples", "audio_seconds", "blocks_per_repeat",
                                         "processed_frames_per_repeat", "final_block_samples")},
        "sample_rate_hz": 16000, "frame_length_samples": 400, "frame_hop_samples": 320,
        "runner_sha256": "1" * 64, "model_sha256": "2" * 64,
        "keyword_pack_sha256": "3" * 64, "audio_sha256": geometry["file_sha256"],
        "runtime_version": "fixture", "runtime_source_revision": "a" * 40,
        "runtime_config_digest": "b" * 64, "runtime_target": "fixture-target",
        "block_samples": 320, "block_deadline_us": 20000.0,
        "repeats": repeats, "blocks": blocks,
        "processed_frames": geometry["processed_frames_per_repeat"] * repeats,
        "processed_samples": geometry["audio_samples"] * repeats,
        "model_bytes": 1, "keyword_pack_bytes": 1, "arena_bytes": 8192,
        "total_process_us": total_us, "mean_process_us": 1000.0,
        "p50_process_us": 800.0, "p95_process_us": 900.0,
        "p99_process_us": 1000.0, "max_process_us": 1100.0,
        "rtf": total_us / (geometry["audio_seconds"] * repeats * 1_000_000.0),
        "p99_headroom": 20.0,
    }


def verify_metadata(root: pathlib.Path) -> None:
    audio = root / "geometry.wav"
    for samples in (0, 1, 320, 399):
        write_pcm_fixture(audio, samples)
        try:
            board_wav_stats(audio)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ineffective {samples}-sample reset workload admitted")
    for samples, frames, blocks, tail in ((400, 1, 2, 80), (719, 1, 3, 79),
                                          (720, 2, 3, 80), (960, 2, 3, 320),
                                          (16000, 49, 50, 320)):
        write_pcm_fixture(audio, samples)
        identity = board_wav_stats(audio)
        require(identity["processed_frames_per_repeat"] == frames)
        require(identity["blocks_per_repeat"] == blocks)
        require(identity["final_block_samples"] == tail)
        summary = board_summary_fixture(audio, repeats=1000)
        hashes = {key: summary[key] for key in ("runner_sha256", "model_sha256", "keyword_pack_sha256", "audio_sha256")}
        def check(value: dict) -> dict:
            return validate_board(value, 1, 1, "a" * 40, hashes, identity)
        require(check(summary)["processed_frames"] == frames * 1000)
        mutations = {
            "schema_version": 1, "sample_rate_hz": 8000, "frame_length_samples": 320,
            "frame_hop_samples": 160, "block_samples": 160, "final_block_samples": 1,
            "audio_samples": samples + 1, "audio_seconds": 999.0,
            "blocks_per_repeat": blocks + 1, "blocks": 1, "processed_samples": 1,
            "processed_frames_per_repeat": 0, "processed_frames": 0, "repeats": 1,
            **{key: "unsupported" for key in BOARD_SEMANTICS},
        }
        for key, value in mutations.items():
            changed = dict(summary, **{key: value})
            try:
                check(changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"tampered {key} admitted")
        changed = dict(summary)
        for key in ("total_process_us", "mean_process_us", "p50_process_us", "p95_process_us",
                    "p99_process_us", "max_process_us", "rtf", "p99_headroom"):
            changed[key] = 0.0
        try:
            check(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("zero-work timing admitted")
    audio.write_bytes(audio.read_bytes()[:-2])
    try:
        board_wav_stats(audio)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated PCM admitted")
    for invalid_bytes in (b"", b"non-wave", b"RIFF\x00\x00\x00\x00WAVE"):
        audio.write_bytes(invalid_bytes)
        try:
            board_wav_stats(audio)
        except ValueError:
            pass
        else:
            raise AssertionError("non-WAV audio admitted")
    for channels, rate, width in ((2, 16000, 2), (1, 8000, 2), (1, 16000, 1)):
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(width)
            stream.setframerate(rate)
            stream.writeframes(b"\x00" * (16000 * channels * width))
        try:
            board_wav_stats(audio)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported PCM geometry admitted")
    artifact = {"name": "board-summary.json", "sha256": "c" * 64, "bytes": 100}
    require_board_raw_binding(artifact, [artifact])
    for rows in ([], [dict(artifact, sha256="d" * 64)], [artifact, artifact]):
        try:
            require_board_raw_binding(artifact, rows)
        except ValueError:
            pass
        else:
            raise AssertionError("unbound board summary admitted")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=pathlib.Path)
    parser.add_argument("--statistics-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    if not args.statistics_only and not args.metadata_only and args.runner is None:
        parser.error("--runner is required unless --statistics-only is selected")
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        verify_metadata(root)
        if args.metadata_only:
            print("test_board_bench: metadata geometry, timing and attestation contracts ok (no model execution)")
            return 0
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
        require(result["schema_version"] == 2)
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
        require(result["blocks_per_repeat"] == 50)
        require(result["audio_samples"] == 16000)
        require(result["processed_samples"] == 32000)
        require(result["processed_frames_per_repeat"] == 49)
        require(result["processed_frames"] == 98)
        require(result["repeat_policy"] == "reset-before-each-repeat")
        require(result["workload_contract"] == "pcm16-400-window-320-hop-v1")
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
        for samples in (0, 320, 399):
            write_pcm_fixture(wav, samples)
            rejected = subprocess.run([str(args.runner), str(model), str(pack), str(wav), "1000"],
                                      text=True, capture_output=True, check=False)
            require(rejected.returncode != 0, "sub-window reset workload admitted by runner")
            require(not rejected.stdout, "ineffective workload emitted benchmark metrics")

    print("test_board_bench: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
