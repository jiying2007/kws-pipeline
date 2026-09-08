#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from final_afe_identity import canonical_sha, inspect_adapter, sha256_file  # noqa: E402

SAMPLE_RATE_HZ = 16000


def inspect_output(path: pathlib.Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
            raise ValueError(f"{path}: AFE output must be uncompressed PCM16 WAV")
        if reader.getframerate() != SAMPLE_RATE_HZ or reader.getnchannels() != 1:
            raise ValueError(f"{path}: AFE output must be mono 16-kHz")
        frames = reader.getnframes()
    if frames <= 0:
        raise ValueError(f"{path}: AFE output is empty")
    return {"frames": frames, "duration_s": frames / float(SAMPLE_RATE_HZ)}


def render_argv(template: list[str], values: dict[str, str]) -> list[str]:
    result = []
    for token in template:
        rendered = token
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        if "{" in rendered or "}" in rendered:
            raise ValueError(f"unresolved AFE command placeholder: {rendered}")
        result.append(rendered)
    return result


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", required=True, type=pathlib.Path)
    parser.add_argument("--adapter", required=True, type=pathlib.Path)
    parser.add_argument("--expected-identity", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    adapter, executable, identity = inspect_adapter(args.adapter)
    expected_identity = json.loads(args.expected_identity.read_text(encoding="utf-8"))
    if identity != expected_identity:
        raise ValueError("final AFE identity drifted after held-out corpus exposure")
    template = adapter["command_argv"]

    output_dir = args.output_dir.resolve()
    post_root = output_dir / "post-afe"
    result_root = output_dir / "result-sidecars"
    post_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    references = []
    evidence_rows = []
    output_identity = []
    latency_samples_all = []

    for index, row in enumerate(manifest.get("recordings", [])):
        recording = str(row.get("recording", "")).strip()
        if not recording:
            raise ValueError(f"recordings[{index}] has no recording ID")
        rel = pathlib.Path(str(row.get("input_path", "")))
        input_path = rel if rel.is_absolute() else args.audio_root / rel
        input_path = input_path.resolve(strict=True)
        if sha256_file(input_path) != str(row.get("input_sha256", "")):
            raise ValueError(f"{recording}: raw input SHA drifted after corpus intake")

        output_path = post_root / f"{recording}.wav"
        result_path = result_root / f"{recording}.json"
        argv = render_argv(
            template,
            {
                "executable": str(executable),
                "input": str(input_path),
                "output": str(output_path),
                "result": str(result_path),
            },
        )
        completed = subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"{recording}: AFE command failed ({completed.returncode}): "
                f"{completed.stderr[-1000:]}"
            )
        if not output_path.is_file() or not result_path.is_file():
            raise ValueError(f"{recording}: AFE command did not create output/result")

        output_geometry = inspect_output(output_path)
        sidecar = json.loads(result_path.read_text(encoding="utf-8"))
        latency_samples = sidecar.get("latency_samples")
        if (
            isinstance(latency_samples, bool)
            or not isinstance(latency_samples, int)
            or latency_samples < 0
        ):
            raise ValueError(
                f"{recording}: result sidecar requires non-negative integer latency_samples"
            )
        latency_samples_all.append(latency_samples)
        latency_s = latency_samples / float(SAMPLE_RATE_HZ)

        expected = []
        for event in row.get("expected", []):
            start_s = float(event["start_s"]) + latency_s
            end_s = float(event["end_s"]) + latency_s
            if end_s > output_geometry["duration_s"] + 1e-9:
                raise ValueError(
                    f"{recording}: latency-adjusted expected event exceeds AFE output"
                )
            expected.append(
                {
                    "keyword_id": int(event["keyword_id"]),
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )

        output_sha = sha256_file(output_path)
        result_sha = sha256_file(result_path)
        references.append(
            {
                "recording": recording,
                "audio_path": output_path.name,
                "duration_s": output_geometry["duration_s"],
                "speaker_id": row["speaker_id"],
                "session_id": row["session_id"],
                "source_id": row["source_id"],
                "room_id": row["room_id"],
                "device_id": row["device_id"],
                "distance_m": row["distance_m"],
                "azimuth_deg": row["azimuth_deg"],
                "snr_db": row.get("snr_db"),
                "tags": row["tags"],
                "afe_latency_samples": latency_samples,
                "expected": expected,
            }
        )
        evidence_rows.append(
            {
                "schema_version": 1,
                "recording": recording,
                "backend": "command",
                "shipping_authority": True,
                "executable_sha256": identity["executable_sha256"],
                "config_bundle_sha256": identity["config_bundle_sha256"],
                "input_pcm_sha256": row["input_sha256"],
                "output_pcm_sha256": output_sha,
                "result_sidecar_sha256": result_sha,
                "pipeline_source_sha": identity.get("pipeline_source_sha"),
                "toolchain": identity["toolchain"],
                "latency_samples": latency_samples,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "channels": 1,
                "sample_format": "pcm_s16le",
                "sku": identity["sku"],
                "microphone_revision": identity["microphone_revision"],
                "enclosure_revision": identity["enclosure_revision"],
                "audio_route": identity["audio_route"],
            }
        )
        output_identity.append(
            {
                "recording": recording,
                "sha256": output_sha,
                "bytes": output_path.stat().st_size,
                "duration_s": output_geometry["duration_s"],
            }
        )

    if not references:
        raise ValueError("manifest contains no recordings")
    write_jsonl(output_dir / "references.post-afe.jsonl", references)
    write_jsonl(output_dir / "afe-evidence.jsonl", evidence_rows)
    summary = {
        "schema_version": 1,
        "qualification_id": manifest["qualification_id"],
        "deployment_tag": manifest["deployment_tag"],
        "recordings": len(references),
        "afe": identity,
        "latency_samples": {
            "min": min(latency_samples_all),
            "max": max(latency_samples_all),
        },
        "post_afe_corpus_sha256": canonical_sha(
            sorted(output_identity, key=lambda item: item["recording"])
        ),
        "references_sha256": sha256_file(
            output_dir / "references.post-afe.jsonl"
        ),
        "afe_evidence_sha256": sha256_file(output_dir / "afe-evidence.jsonl"),
    }
    (output_dir / "afe-corpus-summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
