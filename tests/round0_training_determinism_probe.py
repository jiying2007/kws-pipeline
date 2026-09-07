#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from hard_negative_replay import render_hard_negative_replay  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def pcm_sha256(path: pathlib.Path) -> str:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
        raw = reader.readframes(reader.getnframes())
    return hashlib.sha256(raw).hexdigest()


def cpu_model() -> str:
    info = pathlib.Path("/proc/cpuinfo")
    if info.is_file():
        for raw in info.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.lower().startswith("model name") and ":" in raw:
                return raw.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def repository_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def domain_train_digests(index: pathlib.Path) -> dict:
    rows = [row for row in load_jsonl(index) if str(row["split"]) == "train"]
    semantic: list[dict] = []
    sources: list[dict] = []
    rendered: list[dict] = []
    for row in rows:
        source = pathlib.Path(str(row["source_path"]))
        wav = pathlib.Path(str(row["path"]))
        key = {
            "family_id": str(row["family_id"]),
            "variant": int(row["variant"]),
            "scene_seed": int(row["scene_seed"]),
        }
        semantic.append({**key, "scene": row["scene"], "target_ids": row["target_ids"]})
        sources.append({**key, "pcm_sha256": pcm_sha256(source)})
        rendered.append({**key, "pcm_sha256": pcm_sha256(wav)})
    return {
        "examples": len(rows),
        "semantic_sha256": canonical_digest(semantic),
        "source_pcm_tree_sha256": canonical_digest(sources),
        "rendered_pcm_tree_sha256": canonical_digest(rendered),
    }


def replay_digests(directory: pathlib.Path) -> dict:
    rows: list[dict] = []
    for path in sorted((directory / "wav").glob("*.wav")):
        rows.append({"name": path.name, "pcm_sha256": pcm_sha256(path)})
    return {"examples": len(rows), "pcm_tree_sha256": canonical_digest(rows)}


def tensor_digest(state_dict: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "xiaowo.torch-domain.json",
    )
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "build" / "round0-training-determinism-probe",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "build" / "round0-training-determinism-probe.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    work = args.work_dir.resolve()
    output = args.output.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    dataset_dir = work / "datasets" / "round-00"
    render_domain_dataset(config_path, dataset_dir, curriculum_weights=None)
    domain = domain_train_digests(dataset_dir / "domain-index.jsonl")

    replay_dir = work / "hard-negative-replay" / "round-00"
    replay_evidence = render_hard_negative_replay(
        config_path,
        replay_dir,
        round_index=0,
        curriculum_weights=None,
    )
    replay = replay_digests(replay_dir)
    if replay["examples"] != int(replay_evidence["examples"]):
        raise ValueError("hard-negative replay evidence/example count mismatch")

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    frontends = model_cfg.get("frontends", ["logmel"])
    if frontends != ["logmel"]:
        raise ValueError("probe expects the formal single logmel frontend")

    checkpoint = work / "model.pt"
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(dataset_dir / "train.tsv"),
        "--manifest",
        str(replay_dir / "hard-negatives.tsv"),
        "--tokens",
        str((ROOT / str(cfg["tokens"])).resolve()),
        "--keywords",
        str((ROOT / str(cfg["keywords"])).resolve()),
        "--frontend",
        "logmel",
        "--feature-dim",
        str(int(model_cfg.get("feature_dim", 32))),
        "--hidden-dim",
        str(int(model_cfg.get("hidden_dim", 48))),
        "--epochs",
        str(int(train_cfg.get("epochs", 10))),
        "--batch-size",
        str(int(train_cfg.get("batch_size", 16))),
        "--lr",
        str(float(train_cfg.get("lr", 0.001))),
        "--seed",
        str(int(cfg.get("seed", 1337))),
        "--output",
        str(checkpoint),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"train_ctc failed with exit code {completed.returncode}")

    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report = {
        "schema_version": 1,
        "repository_sha": repository_sha(),
        "config_sha256": sha256_file(config_path),
        "environment": {
            "cpu_model": cpu_model(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
        "domain_train": domain,
        "round0_replay": {
            **replay,
            "manifest_sha256": sha256_file(replay_dir / "hard-negatives.tsv"),
        },
        "training": {
            "examples": int(saved["training_examples"]),
            "corpus_sha256": str(saved["training_corpus_identity"]["corpus_sha256"]),
            "tensor_sha256": tensor_digest(saved["state_dict"]),
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "seed": int(saved["seed"]),
            "epochs": int(saved["epochs"]),
            "batch_size": int(saved["batch_size"]),
            "learning_rate": float(saved["learning_rate"]),
            "training_environment": saved["training_environment"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
