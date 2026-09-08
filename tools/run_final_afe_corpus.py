#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import wave

SAMPLE_RATE_HZ = 16000


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def bundle_identity(paths: list[pathlib.Path]) -> tuple[str, list[dict]]:
    rows = []
    seen_names = set()
    for path in paths:
        path = path.resolve(strict=True)
        name = path.name
        if name in seen_names:
            raise ValueError(f"config bundle contains duplicate basename: {name}")
        seen_names.add(name)
        rows.append({"name": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    rows.sort(key=lambda row: row["name"])
    return canonical_sha(rows), rows


def render_argv(template: list[str], values: dict[str, str]) -> list[str]:
    required = {"{executable}", "{input}", "{output}", "{result}"}
    joined = "\n".join(template)
    missing = sorted(token for token in required if token not in joined)
    if missing:
        raise ValueError(f"AFE command template is missing placeholders: {missing}")
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
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", required=True, type=pathlib.Path)
    parser.add_argument("--adapter", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or adapter.get("schema_version") != 1:
        raise ValueError("manifest/adapter schema_version must be 1")

    executable = pathlib.Path(str(adapter.get("executable_path", ""))).resolve(strict=True)
    if not executable.is_file():
        raise ValueError("AFE executable_path must resolve to a file")
    executable_sha = sha256_file(executable)
    config_paths = [pathlib.Path(str(item)) for item in adapter.get("config_files", [])]
    if not config_paths:
        raise ValueError("AFE adapter config_files must not be empty")
    config_sha, config_files = bundle_identity(config_paths)
    template = adapter.get("command_argv")
    if not isinstance(template, list) or not template or any(not isinstance(item, str) or not item for item in template):
        raise ValueError("AFE command_argv must be a non-empty string array")

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
        completed = subprocess.run(argv, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            raise ValueError(f"{recording}: AFE command failed ({completed.returncode}): {completed.stderr[-1000:]}")
        if not output_path.is_file() or not result_path.is_file():
            raise ValueError(f"{recording}: AFE command did not create output/result")

        output_geometry = inspect_output(output_path)
        sidecar = json.loads(result_path.read_text(encoding="utf-8"))
        latency_samples = sidecar.get("latency_samples")
        if isinstance(latency_samples, bool) or not isinstance(latency_samples, int) or latency_samples < 0:
            raise ValueError(f"{recording}: result sidecar requires non-negative integer latency_samples")
        latency_samples_all.append(latency_samples)
        latency_s = latency_samples / float(SAMPLE_RATE_HZ)

        expected = []
        for event in row.get("expected", []):
            start_s = float(event["start_s"]) + latency_s
            end_s = float(event["end_s"]) + latency_s
            if end_s > output_geometry["duration_s"] + 1e-9:
                raise ValueError(f"{recording}: latency-adjusted expected event exceeds AFE output")
            expected.append({"keyword_id": int(event["keyword_id"]), "start_s": start_s, "end_s": end_s})

        output_sha = sha256_file(output_path)
        result_sha = sha256_file(result_path)
        refs = {
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
        references.append(refs)
        evidence_rows.append(
            {
                "schema_version": 1,
                "recording": recording,
                "backend": "command",
                "shipping_authority": True,
                "executable_sha256": executable_sha,
                "config_bundle_sha256": config_sha,
                "input_pcm_sha256": row["input_sha256"],
                "output_pcm_sha256": output_sha,
                "result_sidecar_sha256": result_sha,
                "pipeline_source_sha": adapter.get("pipeline_source_sha"),
                "toolchain": adapter["toolchain"],
                "latency_samples": latency_samples,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "channels": 1,
                "sample_format": "pcm_s16le",
                "sku": adapter["sku"],
                "microphone_revision": adapter["microphone_revision"],
                "enclosure_revision": adapter["enclosure_revision"],
                "audio_route": adapter["audio_route"],
            }
        )
        output_identity.append({"recording": recording, "sha256": output_sha, "bytes": output_path.stat().st_size})

    if not references:
        raise ValueError("manifest contains no recordings")
    write_jsonl(output_dir / "references.post-afe.jsonl", references)
    write_jsonl(output_dir / "afe-evidence.jsonl", evidence_rows)
    summary = {
        "schema_version": 1,
        "qualification_id": manifest["qualification_id"],
        "deployment_tag": manifest["deployment_tag"],
        "recordings": len(references),
        "afe": {
            "backend": "command",
            "shipping_authority": True,
            "executable_sha256": executable_sha,
            "config_bundle_sha256": config_sha,
            "config_files": config_files,
            "sku": adapter["sku"],
            "microphone_revision": adapter["microphone_revision"],
            "enclosure_revision": adapter["enclosure_revision"],
            "audio_route": adapter["audio_route"],
            "toolchain": adapter["toolchain"],
            "pipeline_source_sha": adapter.get("pipeline_source_sha"),
        },
        "latency_samples": {
            "min": min(latency_samples_all),
            "max": max(latency_samples_all),
        },
        "post_afe_corpus_sha256": canonical_sha(sorted(output_identity, key=lambda item: item["recording"])),
        "references_sha256": sha256_file(output_dir / "references.post-afe.jsonl"),
        "afe_evidence_sha256": sha256_file(output_dir / "afe-evidence.jsonl"),
    }
    (output_dir / "afe-corpus-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
