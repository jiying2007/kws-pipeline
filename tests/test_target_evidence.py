from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CPU_PERCENT_SEMANTICS = "process_cpu_time / elapsed / online_cpu_capacity * 100"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        collector = ROOT / "tools" / "collect_target_evidence.py"
        power = root / "power.csv"
        power.write_text("t,power_mw\n0,123\n", encoding="utf-8")
        soak = root / "runtime-soak.json"
        soak.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "command": ["fixture"],
                    "pid": 123,
                    "cpu_capacity_count": 2,
                    "cpu_percent_semantics": CPU_PERCENT_SEMANTICS,
                    "requested_hours": 1.0,
                    "elapsed_seconds": 3636.0,
                    "elapsed_hours": 1.01,
                    "completed_requested_duration": True,
                    "termination_returncode": -15,
                    "sample_seconds": 60.0,
                    "initial_cpu_seconds": 10.0,
                    "samples": [
                        {
                            "elapsed_s": 0.0,
                            "rss_kib": 500.0,
                            "cpu_seconds": 10.0,
                            "temp_c": 50.0,
                        },
                        {
                            "elapsed_s": 3636.0,
                            "rss_kib": 512.0,
                            "cpu_seconds": 373.6,
                            "temp_c": 55.0,
                        },
                    ],
                    "max_rss_kib": 512.0,
                    "average_cpu_percent": 5.0,
                    "max_temp_c": 55.0,
                },
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
        assert value["schema_version"] == 2
        assert value["evidence_class"] == "product-board"
        assert value["sku"] == "fixture-sku"
        assert value["source_sha"] == "b" * 40
        assert value["target"] == "fixture"
        assert value["builder_id"] == "fixture-builder"
        assert value["dut_id"] == "fixture-dut"
        assert value["collector_id"] == "fixture-collector"
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
