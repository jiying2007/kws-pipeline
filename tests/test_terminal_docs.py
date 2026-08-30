from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def require_all(text: str, path: pathlib.Path, values: tuple[str, ...]) -> None:
    for value in values:
        assert value in text, f"{path}: missing {value}"


def main() -> int:
    docs = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "docs" / "TERMINAL_HARDENING.md",
        ROOT / "docs" / "CORPUS_IDENTITY.md",
        ROOT / "docs" / "TARGET_EVIDENCE.md",
        ROOT / "docs" / "RELEASE_QUALIFICATION.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "GOVERNANCE_TARGET.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), path

    boundary = (ROOT / "docs" / "TERMINAL_HARDENING.md").read_text(encoding="utf-8")
    assert "does not claim real Mandarin/device qualification" in boundary

    required_evidence_cli = (
        "--evidence-raw",
        "--attestation-verification",
        "--board-runner",
        "--model",
        "--keyword-pack",
        "--board-audio",
        "--sku",
        "--source-sha",
        "--builder-id",
        "--dut-id",
        "--collector-id",
    )
    for relative in ("README.md", "README.zh-CN.md", "docs/TARGET_EVIDENCE.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        require_all(path.read_text(encoding="utf-8"), path, required_evidence_cli)

    required_manifest_cli = ("--evidence-raw", "--attestation-verification", "--sku", "--corpus-id")
    for relative in ("README.md", "README.zh-CN.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        require_all(path.read_text(encoding="utf-8"), path, required_manifest_cli)

    workflow = (ROOT / ".github" / "workflows" / "training-integration.yml").read_text(encoding="utf-8")
    assert "vars.KWS_TRAINING_IMAGE" in workflow
    for relative in ("README.md", "README.zh-CN.md", "docs/RELEASE_QUALIFICATION.md"):
        path = ROOT / relative
        assert "KWS_TRAINING_IMAGE" in path.read_text(encoding="utf-8"), f"{path}: training image variable drift"

    evidence_example = json.loads(
        (ROOT / "configs" / "qualification.evidence.example.json").read_text(encoding="utf-8")
    )
    assert evidence_example["schema_version"] == 2
    assert evidence_example["evidence_class"] == "product-board"
    for key in (
        "sku",
        "source_sha",
        "builder_id",
        "dut_id",
        "collector_id",
        "raw_evidence_sha256",
        "attestation_verification_sha256",
        "board_runner_sha256",
        "model_sha256",
        "keyword_pack_sha256",
        "board_audio_sha256",
        "runtime_soak_raw",
    ):
        assert key in evidence_example, f"qualification.evidence.example.json: missing {key}"

    print("test_terminal_docs: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
