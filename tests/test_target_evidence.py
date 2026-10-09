from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
from qualification_fixture import runtime_soak_fixture
from collect_target_evidence import load_runtime_soak
from qualification_metrics import _runtime_soak_metrics


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_runtime_contract(root: pathlib.Path) -> None:
    path = root / "synthetic-soak-contract.json"

    def validate(value: dict) -> tuple[dict, dict]:
        text = json.dumps(value, sort_keys=True)
        path.write_text(text, encoding="utf-8")
        return (load_runtime_soak(path), _runtime_soak_metrics(
            {"runtime_soak_raw": text}, hashlib.sha256(text.encode()).hexdigest()))

    # One-core CPU units never depend on machine capacity. Multithread process
    # time may exceed 100%; zero usage and the exact 100% boundary are legal.
    for percent, capacity, threads in ((0, 1, 1), (10, 1, 1), (20, 2, 1),
                                       (20, 8, 1), (100, 1, 1), (150, 2, 2)):
        left, right = validate(runtime_soak_fixture(1.0, cpu_percent=percent,
                                                   capacity=capacity, threads=threads))
        assert left == right
        assert abs(left["cpu_percent"] - percent) < 1e-9
        assert left["max_thread_count"] == threads
        assert left["audio_seconds"] is None
        assert left["cpu_seconds_per_audio_second"] is None

    mutations = [
        ("schema_version", 2),
        ("measurement_contract_id", "legacy-online-capacity"),
        ("cpu_percent_semantics", "process_cpu_time / elapsed / online_cpu_capacity * 100"),
        ("average_cpu_percent", 10.0),  # 20% on two cores is NOT 10%.
        ("process_cpu_seconds", 360.0),
        ("wall_seconds", 7200.0),
        ("max_thread_count", 2),
        ("cpu_capacity_count", True),
        ("audio_seconds", 3600.0),
        ("cpu_seconds_per_audio_second", 0.2),
    ]
    invalid = []
    for key, replacement in mutations:
        value = runtime_soak_fixture(1.0, cpu_percent=20.0)
        value[key] = replacement
        invalid.append(value)
    for key in ("measurement_contract_id", "max_thread_count", "audio_seconds"):
        value = runtime_soak_fixture(1.0)
        del value[key]
        invalid.append(value)
    for key, replacement in (("cpu_seconds", None), ("cpu_seconds", 9.0),
                             ("elapsed_s", 3500.0), ("thread_count", 0),
                             ("thread_count", True)):
        value = runtime_soak_fixture(1.0)
        value["samples"][-1][key] = replacement
        invalid.append(value)
    for value in invalid:
        text = json.dumps(value, sort_keys=True)
        path.write_text(text, encoding="utf-8")
        for validator in (lambda: load_runtime_soak(path), lambda: _runtime_soak_metrics(
                {"runtime_soak_raw": text}, hashlib.sha256(text.encode()).hexdigest())):
            try:
                validator()
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid runtime CPU evidence accepted: {value}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        check_runtime_contract(root)
        collector = ROOT / "tools" / "collect_target_evidence.py"
        power = root / "power.csv"
        power.write_text("t,power_mw\n0,123\n", encoding="utf-8")
        soak = root / "runtime-soak.json"
        soak.write_text(
            json.dumps(
                runtime_soak_fixture(1.01, requested_hours=1.0),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        board_runner = root / "kws_board_bench"
        model = root / "model.kwm"
        keyword_pack = root / "keywords.kwk"
        board_audio = root / "board.wav"
        board_runner.write_bytes(b"fixture-board-runner")
        model.write_bytes(b"fixture-model")
        keyword_pack.write_bytes(b"fixture-keyword-pack")
        board_audio.write_bytes(b"fixture-board-audio")
        afe_sha = "c" * 64
        afe_identity = "d" * 64

        evidence_raw = root / "evidence-raw.jsonl"
        raw_paths = [soak, power]
        evidence_raw.write_text(
            "".join(
                json.dumps(
                    {
                        "name": path.name,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    },
                    sort_keys=True,
                )
                + "\n"
                for path in raw_paths
            ),
            encoding="utf-8",
        )

        attestation = root / "attestation-verification.json"
        attestation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verified": True,
                    "subject_kind": "kws-target-evidence",
                    "issuer": "fixture-trusted-attestor",
                    "trust_policy": "fixture-product-policy",
                    "verified_at_utc": "2026-08-30T00:00:00Z",
                    "subject_sha256": sha256_file(evidence_raw),
                    "collector_sha256": sha256_file(collector),
                    "board_runner_sha256": sha256_file(board_runner),
                    "model_sha256": sha256_file(model),
                    "keyword_pack_sha256": sha256_file(keyword_pack),
                    "audio_frontend_identity_sha256": afe_identity,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        output = root / "evidence.json"
        subprocess.check_call(
            [
                sys.executable,
                str(collector),
                "--output", str(output),
                "--target", "fixture",
                "--board-revision", "A",
                "--soc", "fixture-soc",
                "--toolchain", "fixture-gcc",
                "--compiler-flags=-O3",
                "--audio-frontend", "fixture-afe",
                "--audio-frontend-sha256", afe_sha,
                "--audio-frontend-identity-sha256", afe_identity,
                "--runtime-soak", str(soak),
                "--stack-high-water-bytes", "4096",
                "--average-power-mw", "123",
                "--power-raw", str(power),
                "--evidence-raw", str(evidence_raw),
                "--attestation-verification", str(attestation),
                "--board-runner", str(board_runner),
                "--model", str(model),
                "--keyword-pack", str(keyword_pack),
                "--board-audio", str(board_audio),
                "--sku", "fixture-sku",
                "--source-sha", "b" * 40,
                "--builder-id", "fixture-builder",
                "--dut-id", "fixture-dut",
                "--collector-id", "fixture-collector",
                "--instrument-id", "meter-1",
                "--calibration-id", "cal-1",
            ]
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        assert value["schema_version"] == 3
        assert value["evidence_class"] == "product-board"
        assert value["sku"] == "fixture-sku"
        assert value["source_sha"] == "b" * 40
        assert value["target"] == "fixture"
        assert value["builder_id"] == "fixture-builder"
        assert value["dut_id"] == "fixture-dut"
        assert value["collector_id"] == "fixture-collector"
        assert value["audio_frontend_sha256"] == afe_sha
        assert value["audio_frontend_identity_sha256"] == afe_identity
        assert value["soak_hours"] == 1.01
        assert abs(value["cpu_percent"] - 5.0) < 1e-9
        assert value["rss_kib"] == 512.0
        assert value["max_temp_c"] == 55.0
        assert value["collector_sha256"] == sha256_file(collector)
        assert value["raw_evidence_sha256"] == sha256_file(evidence_raw)
        assert value["attestation_verification_sha256"] == sha256_file(attestation)
        assert len(value["runtime_soak_sha256"]) == 64
        assert value["runtime_soak_raw"] == soak.read_text(encoding="utf-8")
        assert len(value["power_raw_sha256"]) == 64
        names = {item["name"] for item in value["raw_evidence"]}
        assert names == {"runtime-soak.json", "power.csv"}
        assert value["instrument_id"] == "meter-1"
    print("test_target_evidence: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
