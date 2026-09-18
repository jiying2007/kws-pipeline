#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

POLICY_ID = "speech-like-command-provider-v1"
PROVIDER_SCHEMA = 1
REQUEST_SCHEMA = 1
MANIFEST_SCHEMA = 1
MANIFEST_CLASS = "speech-like-synthetic-recording-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: request list is empty")
    return rows


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def require_sha256(value: object, label: str) -> str:
    result = require_text(value, label)
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be lowercase sha256")
    return result


def load_policy(path: pathlib.Path) -> dict:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != POLICY_ID:
        raise ValueError("command-provider policy identity mismatch")
    execution = value.get("execution")
    if not isinstance(execution, dict) or execution.get("shell") is not False:
        raise ValueError("command-provider policy must forbid shell execution")
    if execution.get("require_executable_sha256") is not True or execution.get("require_asset_sha256") is not True:
        raise ValueError("command-provider policy must require executable/asset hashes")
    required = value.get("required_template_placeholders")
    if required != ["executable", "text", "output"]:
        raise ValueError("command-provider v1 placeholder contract drifted")
    groups = value.get("allowed_request_groups")
    if not isinstance(groups, list) or set(groups) != {"train", "generalization-search", "generalization-freeze"}:
        raise ValueError("command-provider v1 groups drifted")
    output = value.get("output_format")
    if output != {"sample_rate_hz": 16000, "channels": 1, "sample_width_bits": 16, "container": "wav-pcm"}:
        raise ValueError("command-provider v1 output format drifted")
    return value


def inspect_wav(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise ValueError(f"provider did not create WAV: {path}")
    with wave.open(str(path), "rb") as reader:
        channels = int(reader.getnchannels())
        width = int(reader.getsampwidth())
        rate = int(reader.getframerate())
        frames = int(reader.getnframes())
        compression = reader.getcomptype()
        pcm = reader.readframes(frames)
    if channels != 1 or width != 2 or rate != 16000 or compression != "NONE" or frames <= 0:
        raise ValueError(f"provider output is not mono PCM16 16-kHz WAV: {path}")
    return {
        "frames": frames,
        "duration_s": frames / 16000.0,
        "file_sha256": sha256_file(path),
        "pcm_sha256": sha256_bytes(pcm),
    }


def normalize_provider(path: pathlib.Path, value: dict, policy: dict) -> dict:
    if int(value.get("schema_version", 0)) != PROVIDER_SCHEMA:
        raise ValueError("provider spec schema_version must be 1")
    if value.get("provider_kind") != policy["provider_kind"]:
        raise ValueError("provider_kind must be offline-tts")
    provider_name = require_text(value.get("provider_name"), "provider_name")
    provider_version = require_text(value.get("provider_version"), "provider_version")
    license_id = require_text(value.get("license_id"), "license_id")
    locale = require_text(value.get("locale"), "locale")
    if locale.lower() not in {"zh-cn", "zh_cn", "cmn-hans-cn"}:
        raise ValueError("command-provider v1 is scoped to Mandarin zh-CN")

    executable = value.get("executable")
    if not isinstance(executable, dict):
        raise ValueError("executable must be an object")
    exe_path_raw = require_text(executable.get("path"), "executable.path")
    exe_path = pathlib.Path(exe_path_raw)
    exe_path = exe_path.resolve() if exe_path.is_absolute() else (path.parent / exe_path).resolve()
    if not exe_path.is_file():
        raise ValueError(f"provider executable is missing: {exe_path}")
    exe_sha = require_sha256(executable.get("sha256"), "executable.sha256")
    if sha256_file(exe_path) != exe_sha:
        raise ValueError("provider executable sha256 mismatch")

    assets: dict[str, dict] = {}
    for index, raw in enumerate(value.get("assets", [])):
        if not isinstance(raw, dict):
            raise ValueError(f"asset {index} must be an object")
        role = require_text(raw.get("role"), f"asset {index}.role")
        if role in assets:
            raise ValueError(f"duplicate asset role: {role}")
        asset_path_raw = require_text(raw.get("path"), f"asset {index}.path")
        asset_path = pathlib.Path(asset_path_raw)
        asset_path = asset_path.resolve() if asset_path.is_absolute() else (path.parent / asset_path).resolve()
        if not asset_path.is_file():
            raise ValueError(f"provider asset is missing: {asset_path}")
        expected = require_sha256(raw.get("sha256"), f"asset {index}.sha256")
        if sha256_file(asset_path) != expected:
            raise ValueError(f"provider asset sha256 mismatch: {role}")
        assets[role] = {"path": asset_path, "sha256": expected}

    argv_template = value.get("argv_template")
    if not isinstance(argv_template, list) or not argv_template or any(not isinstance(item, str) for item in argv_template):
        raise ValueError("argv_template must be a non-empty string list")
    placeholders = {match for token in argv_template for match in PLACEHOLDER_RE.findall(token)}
    for required in policy["required_template_placeholders"]:
        if required not in placeholders:
            raise ValueError(f"argv_template is missing required placeholder: {required}")
    if argv_template[0] != "{executable}":
        raise ValueError("argv_template[0] must be {executable}")
    for placeholder in placeholders:
        if placeholder.startswith("asset:") and placeholder.split(":", 1)[1] not in assets:
            raise ValueError(f"argv_template references unknown asset: {placeholder}")

    timeout = int(value.get("timeout_seconds", 120))
    max_timeout = int(policy["execution"]["timeout_seconds_max"])
    if timeout <= 0 or timeout > max_timeout:
        raise ValueError("provider timeout_seconds is outside policy")

    identity = {
        "schema_version": PROVIDER_SCHEMA,
        "provider_kind": value["provider_kind"],
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "locale": locale,
        "executable_sha256": exe_sha,
        "assets": [{"role": role, "sha256": assets[role]["sha256"]} for role in sorted(assets)],
        "argv_template": list(argv_template),
    }
    return {
        "provider_name": provider_name,
        "provider_version": provider_version,
        "license_id": license_id,
        "locale": locale,
        "executable": exe_path,
        "assets": assets,
        "argv_template": list(argv_template),
        "timeout_seconds": timeout,
        "identity": identity,
    }


def normalize_requests(path: pathlib.Path, rows: list[dict], policy: dict) -> list[dict]:
    allowed_groups = set(policy["allowed_request_groups"])
    normalized: list[dict] = []
    owners: dict[str, dict[str, set[str]]] = {"voice_id": {}, "source_id": {}}
    for index, row in enumerate(rows):
        if int(row.get("schema_version", 0)) != REQUEST_SCHEMA:
            raise ValueError(f"{path}:{index + 1}: schema_version must be 1")
        group = require_text(row.get("group"), f"request {index}.group")
        if group not in allowed_groups:
            raise ValueError(f"request {index}: unsupported group {group}")
        text = require_text(row.get("text"), f"request {index}.text")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(not isinstance(item, str) or not item for item in tokens):
            raise ValueError(f"request {index}.tokens must be a non-empty string list")
        voice_id = require_text(row.get("voice_id"), f"request {index}.voice_id")
        source_id = require_text(row.get("source_id"), f"request {index}.source_id")
        parameters = row.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"request {index}.parameters must be an object")
        normalized_parameters: dict[str, str] = {}
        for key, value in parameters.items():
            key_text = require_text(key, f"request {index}.parameter key")
            if key_text in {"executable", "text", "output"} or key_text.startswith("asset:"):
                raise ValueError(f"request {index}: reserved parameter name {key_text}")
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"request {index}.parameters.{key_text} must be string/number")
            normalized_parameters[key_text] = str(value)
        for field, identity in (("voice_id", voice_id), ("source_id", source_id)):
            owners[field].setdefault(identity, set()).add(group)
        normalized.append({
            "group": group,
            "text": text,
            "tokens": list(tokens),
            "voice_id": voice_id,
            "source_id": source_id,
            "parameters": normalized_parameters,
        })
    for field, values in owners.items():
        bad = sorted(identity for identity, groups in values.items() if len(groups) > 1)
        if bad:
            raise ValueError(f"request cross-group {field} overlap: {', '.join(bad)}")
    return normalized


def render_token(token: str, mapping: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in mapping:
            raise ValueError(f"argv_template placeholder has no value: {key}")
        return mapping[key]
    return PLACEHOLDER_RE.sub(replace, token)


def generate(
    *,
    policy: dict,
    provider: dict,
    requests: list[dict],
    output_root: pathlib.Path,
    workers: int = 1,
) -> dict:
    if workers <= 0 or workers > 4:
        raise ValueError("generation workers must be in [1,4]")
    output_root.mkdir(parents=True, exist_ok=True)
    group_indexes: dict[str, int] = {
        group: 0 for group in policy["allowed_request_groups"]
    }
    tasks: list[tuple[int, dict, str, int, pathlib.Path]] = []
    for ordinal, row in enumerate(requests):
        group = row["group"]
        index = group_indexes[group]
        group_indexes[group] += 1
        group_dir = output_root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        wav_path = group_dir / f"recording-{index:06d}.wav"
        tasks.append((ordinal, row, group, index, wav_path))

    def render_one(
        task: tuple[int, dict, str, int, pathlib.Path]
    ) -> tuple[int, str, int, dict]:
        ordinal, row, group, index, wav_path = task
        mapping = {
            "executable": provider["executable"].as_posix(),
            "text": row["text"],
            "output": wav_path.as_posix(),
            **row["parameters"],
        }
        for role, asset in provider["assets"].items():
            mapping[f"asset:{role}"] = asset["path"].as_posix()
        argv = [
            render_token(token, mapping) for token in provider["argv_template"]
        ]
        if pathlib.Path(argv[0]).resolve() != provider["executable"]:
            raise ValueError("rendered executable drifted from verified executable")
        subprocess.run(
            argv,
            check=True,
            shell=False,
            timeout=provider["timeout_seconds"],
        )
        inspected = inspect_wav(wav_path)
        generation_identity = {
            "provider": provider["identity"],
            "parameters": dict(sorted(row["parameters"].items())),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "evidence_class": MANIFEST_CLASS,
            "synthetic": True,
            "provider_kind": "offline-tts",
            "provider_name": provider["provider_name"],
            "provider_version": provider["provider_version"],
            "license_id": provider["license_id"],
            "voice_id": row["voice_id"],
            "source_id": row["source_id"],
            "locale": provider["locale"],
            "text": row["text"],
            "tokens": row["tokens"],
            "generation_config_sha256": canonical_sha256(generation_identity),
            "audio": wav_path.name,
            "file_sha256": inspected["file_sha256"],
            "pcm_sha256": inspected["pcm_sha256"],
        }
        return ordinal, group, index, manifest

    rendered: list[tuple[int, str, int, dict]] = []
    if workers == 1:
        rendered = [render_one(task) for task in tasks]
    else:
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(render_one, task): task[0] for task in tasks
            }
            for future in as_completed(future_map):
                ordinal = future_map[future]
                try:
                    rendered.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"request {ordinal}: {exc}")
        if failures:
            raise RuntimeError(
                "speech-like provider generation failed: "
                + "; ".join(sorted(failures))
            )

    manifests: dict[str, list[tuple[int, dict]]] = {
        group: [] for group in policy["allowed_request_groups"]
    }
    for _ordinal, group, index, manifest in rendered:
        manifests[group].append((index, manifest))

    outputs: dict[str, str] = {}
    recording_count = 0
    for group, indexed_rows in manifests.items():
        if not indexed_rows:
            continue
        indexed_rows.sort(key=lambda item: item[0])
        rows = [row for _index, row in indexed_rows]
        recording_count += len(rows)
        manifest_path = output_root / group / "manifest.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        outputs[group] = manifest_path.as_posix()
    return {
        "schema_version": 1,
        "policy": POLICY_ID,
        "evidence_class": "speech-like-command-generation-summary-v1",
        "provider_identity_sha256": canonical_sha256(provider["identity"]),
        "recordings": recording_count,
        "workers": workers,
        "groups": outputs,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate offline speech-like corpus via a hash-bound command provider.")
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--provider", required=True, type=pathlib.Path)
    parser.add_argument("--requests", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    policy = load_policy(args.policy.resolve())
    provider_path = args.provider.resolve()
    provider = normalize_provider(provider_path, load_object(provider_path), policy)
    requests = normalize_requests(args.requests.resolve(), load_jsonl(args.requests.resolve()), policy)
    summary = generate(
        policy=policy,
        provider=provider,
        requests=requests,
        output_root=args.output_root.resolve(),
        workers=args.workers,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"speech-like command provider: recordings={summary['recordings']} "
        f"workers={summary['workers']} provider={provider['provider_name']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
