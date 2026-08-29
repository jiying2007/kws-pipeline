#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_RATE_HZ = 16000


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_line(label: str, metrics: dict) -> str:
    return (
        f"{label}: frr={float(metrics['frr']):.6f} "
        f"far_per_hour={float(metrics['far_per_hour']):.6f} "
        f"p95_ms={float(metrics['p95_post_end_latency_ms']):.3f} "
        f"expected={int(metrics['expected'])} matched={int(metrics['matched'])} "
        f"fa={int(metrics['false_accepts'])} fr={int(metrics['false_rejects'])}"
    )


def jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def false_accept_details(work: pathlib.Path, metrics: dict, split: str) -> list[str]:
    false_path = pathlib.Path(str(metrics.get("false_positives_path", "")))
    rows = jsonl(false_path)
    if not rows:
        return []
    dataset_dir = work / "dataset"
    summary_path = dataset_dir / "dataset-summary.json"
    index_path = dataset_dir / "dataset-index.jsonl"
    if not summary_path.is_file() or not index_path.is_file():
        return [f"{split}.fp raw={row}" for row in rows[:12]]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gap_samples = int(
        round(float(summary.get("continuous_gap_ms", 1600.0)) * SAMPLE_RATE_HZ / 1000.0)
    )
    clips = [row for row in jsonl(index_path) if row.get("split") == split]
    intervals: list[tuple[int, int, dict]] = []
    cursor = gap_samples
    for clip in clips:
        start = cursor
        end = start + int(clip["frames"])
        intervals.append((start, end, clip))
        cursor = end + gap_samples

    details: list[str] = []
    for row in rows[:20]:
        sample = int(round(float(row["time_s"]) * SAMPLE_RATE_HZ))
        source = None
        for start, end, clip in intervals:
            if start <= sample <= end:
                source = clip
                break
        if source is None:
            details.append(
                f"{split}.fp kw={row.get('keyword_id')} t={float(row['time_s']):.3f} "
                f"conf={float(row.get('confidence', 0.0)):.4f} source=gap"
            )
        else:
            details.append(
                f"{split}.fp kw={row.get('keyword_id')} t={float(row['time_s']):.3f} "
                f"conf={float(row.get('confidence', 0.0)):.4f} "
                f"source={source.get('kind')} family={source.get('family_id')} "
                f"tokens={' '.join(source.get('tokens', []))}"
            )
    return details


def failure_diagnostic(work: pathlib.Path, completed: subprocess.CompletedProcess[str]) -> str:
    lines = [f"iterate rc={completed.returncode}"]
    manifest_path = work / "synthetic-loop-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("rounds", []):
                round_index = int(item.get("round", -1))
                lines.append(metric_line(f"round{round_index}.calibration", item["calibration"]))
                lines.append(metric_line(f"round{round_index}.test", item["test"]))
                if round_index == 0:
                    lines.extend(false_accept_details(work, item["calibration"], "calibration"))
                    lines.extend(false_accept_details(work, item["test"], "test"))
                lines.append(
                    f"round{round_index}.replay: hard_negatives="
                    f"{int(item.get('cumulative_hard_negatives', 0))} "
                    f"missed_positives={int(item.get('cumulative_missed_positives', 0))}"
                )
            if isinstance(manifest.get("synthetic_qualification"), dict):
                qualification = manifest["synthetic_qualification"]
                lines.append(metric_line("qualification", qualification))
                lines.extend(false_accept_details(work, qualification, "qualification"))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            lines.append(f"manifest diagnostic failed: {exc}")
    else:
        generated = sorted(
            str(path.relative_to(work)) for path in work.rglob("*") if path.is_file()
        )
        lines.append("manifest missing; generated files: " + ", ".join(generated[-20:]))
    stderr_lines = [line for line in completed.stderr.splitlines() if line.strip()]
    if stderr_lines:
        lines.append("stderr tail: " + " | ".join(stderr_lines[-12:]))
    return "\n".join(lines)


def validate_prototype_evidence(work: pathlib.Path, manifest: dict) -> dict:
    first_round = manifest["rounds"][0]
    model_path = pathlib.Path(first_round["model"])
    provenance_path = pathlib.Path(str(model_path) + ".synthetic-provenance.json")
    assert provenance_path.is_file(), provenance_path
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 2
    assert provenance["evidence_class"] == "synthetic-trained-softmax-prototype"
    assert provenance["model_sha256"] == first_round["model_sha256"]
    assert first_round["model_provenance_sha256"] == sha256_file(provenance_path)
    validation = provenance["validation_confusion"]
    assert int(validation["examples"]) > 0
    assert float(validation["accuracy"]) >= 0.995
    assert float(validation["min_top1_margin"]) > 0.0

    diagnostics_path = model_path.parent / "prototype-fit" / "softmax-diagnostics.json"
    samples_path = model_path.parent / "prototype-fit" / "token-fit-samples.jsonl"
    assert diagnostics_path.is_file(), diagnostics_path
    assert samples_path.is_file() and samples_path.stat().st_size > 0
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["validation_confusion"] == validation
    assert float(diagnostics["optimizer"]["final_loss"]) < float(
        diagnostics["optimizer"]["initial_loss"]
    )
    assert len(str(provenance["fit_samples_sha256"])) == 64
    assert len(str(provenance["diagnostics_sha256"])) == 64
    return provenance


def validate_best_freeze(work: pathlib.Path, manifest: dict) -> None:
    best_dir = work / "best"
    model = best_dir / "model.kwm"
    provenance_path = best_dir / "model-provenance.json"
    assert model.is_file() and model.stat().st_size > 72
    assert provenance_path.is_file() and provenance_path.stat().st_size > 0
    assert manifest["best_model_sha256"] == sha256_file(model)
    assert manifest["best_model_provenance_sha256"] == sha256_file(provenance_path)
    assert manifest["best_checkpoint_sha256"] is None
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["model_sha256"] == manifest["best_model_sha256"]
    assert int(manifest["best_round"]) < len(manifest["rounds"])

    for round_record in manifest["rounds"]:
        expected = float(round_record["calibration_score"]) + float(
            round_record["test_score"]
        )
        assert abs(float(round_record["score"]) - expected) < 1.0e-9


def validate_dataset_contract(work: pathlib.Path) -> dict:
    dataset = work / "dataset"
    summary = json.loads((dataset / "dataset-summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert summary["evidence_class"] == "synthetic-only"
    assert summary["event_boundary"] == "pre-augmentation-active-signal"
    assert set(summary["background_profiles"]) == {"white", "fan", "motor", "media"}
    for split in ("train", "calibration", "test", "qualification"):
        stats = summary["splits"][split]
        assert int(stats["positive_examples"]) > 0
        assert int(stats["token_negative_examples"]) > 0
        assert int(stats["background_examples"]) > 0

    positives = [
        row for row in jsonl(dataset / "dataset-index.jsonl") if row["kind"] == "positive"
    ]
    assert positives
    for row in positives:
        assert isinstance(row["event_start_frame"], int)
        assert isinstance(row["event_end_frame"], int)
        assert 0 <= row["event_start_frame"] < row["event_end_frame"] < row["frames"]

    qualification_ref = jsonl(dataset / "qualification.references.jsonl")
    assert len(qualification_ref) == 1
    # The profile intentionally contributes at least two minutes of pure
    # background before token/confusable clips and isolation gaps are counted.
    assert float(qualification_ref[0]["duration_s"]) > 120.0
    return summary


def assert_unsafe_workdir_rejected(runner: pathlib.Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "training" / "iterate.py"),
            "--config",
            str(ROOT / "configs" / "training" / "xiaowo.synthetic.json"),
            "--runner",
            str(runner),
            "--work-dir",
            pathlib.Path("/").anchor,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "refusing unsafe synthetic work directory" in completed.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    runner = args.runner.resolve()
    assert runner.is_file(), runner

    assert_unsafe_workdir_rejected(runner)

    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td) / "loop"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "training" / "iterate.py"),
                "--config",
                str(ROOT / "configs" / "training" / "xiaowo.synthetic.json"),
                "--runner",
                str(runner),
                "--work-dir",
                str(work),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.returncode == 0, failure_diagnostic(work, completed)
        manifest_path = work / "synthetic-loop-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 2
        assert manifest["evidence_class"] == "synthetic-only"
        assert manifest["qualified"] is True
        assert len(manifest["rounds"]) >= 2
        qualification = manifest["synthetic_qualification"]
        assert qualification["frr"] <= 0.05
        assert qualification["far_per_hour"] == 0.0
        assert qualification["p95_post_end_latency_ms"] <= 800.0
        assert "replay" in manifest
        assert len(manifest["replay"]["hard_negatives_sha256"]) == 64
        assert len(manifest["replay"]["missed_positives_sha256"]) == 64
        assert (work / "best" / "keywords.kwk").stat().st_size > 24
        assert (work / "dataset" / "train.tsv").stat().st_size > 0
        assert (work / "dataset" / "calibration.tsv").stat().st_size > 0
        assert (work / "dataset" / "test.tsv").stat().st_size > 0
        assert (work / "dataset" / "qualification.tsv").stat().st_size > 0
        audit = json.loads((work / "dataset-audit.json").read_text(encoding="utf-8"))
        assert audit["clean"] is True
        dataset_summary = validate_dataset_contract(work)
        provenance = validate_prototype_evidence(work, manifest)
        validate_best_freeze(work, manifest)

        print(metric_line("synthetic_qualification", qualification))
        print(
            "prototype_quantized_validation: "
            f"accuracy={float(provenance['validation_confusion']['accuracy']):.6f} "
            f"min_margin={float(provenance['validation_confusion']['min_top1_margin']):.6f}"
        )
        print(
            "synthetic_background: "
            f"qualification_examples={int(dataset_summary['splits']['qualification']['background_examples'])} "
            f"audio_hours={float(qualification['audio_hours']):.6f}"
        )
        print(
            "best_candidate: "
            f"round={int(manifest['best_round'])} score={float(manifest['best_score']):.6f} "
            "lineage=frozen"
        )

    print("test_synthetic_loop: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
