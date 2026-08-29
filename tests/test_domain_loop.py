#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        config = json.loads(
            (ROOT / "configs" / "training" / "xiaowo.domain.json").read_text(encoding="utf-8")
        )
        for split in ("train", "calibration", "test", "qualification"):
            config["dataset"][split] = {
                "positive_families_per_keyword": 1,
                "confusable_families_per_keyword": 1,
                "random_negative_families": 1,
                "background_seconds_per_profile": 0.1,
                "variants_per_family": 1,
            }
            config["domains"]["scenes_per_example"][split] = 1
        config["domains"]["distance_bands"] = {
            "near": {"distance_m": [0.4, 0.8], "weight": 1.0},
            "mid": {"distance_m": [1.2, 1.8], "weight": 1.0},
            "far": {"distance_m": [3.0, 3.6], "weight": 2.0},
        }
        config["domains"]["azimuth_deg"] = [-60, 0, 60]
        config["domains"]["rt60_s"] = [0.12, 0.28]
        config["domains"]["snr_db"] = [22.0, 34.0]
        config["domains"]["playback"]["probability"] = 0.1
        config["model"]["frontends"] = ["logmel", "pcen-lite"]
        # Keep the production minimum here. With only 8 variants the 75/25
        # train/validation split leaves two validation scenes per token; PCEN's
        # stateful compression makes that unnecessarily high-variance while not
        # exercising a different contract. Sixteen gives 12 train + 4 held-out
        # domain scenes per token and retains the hard 98.5% quantized-fit gate.
        config["model"]["domain_variants_per_token"] = 16
        config["model"]["prototype_candidates"] = [
            {"input_scale": 0.010, "output_scale": 0.050, "blank_bias": 1.8, "token_bias": -1.2}
        ]
        config["calibration"] = {"thresholds": [0.25, 0.55], "coordinate_rounds": 1}
        config["domain_iteration"] = {
            "backend": "prototype",
            "max_rounds": 1,
            "min_rounds": 1,
            "patience": 0,
            "curriculum_strength": 2.0,
            "max_domain_weight": 4.0,
            "stop_on_gate": True,
        }
        # Smoke test validates orchestration and domain accounting, not the formal
        # product-facing strict policy in xiaowo.domain.json.
        config["domain_gates"] = {
            "max_frr": 1.0,
            "max_far_per_hour": 1000000.0,
            "max_p95_latency_ms": 1000000.0,
            "max_far_frr": 1.0,
        }
        path = root / "domain-smoke.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        work = root / "work"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "training" / "iterate_domain.py"),
                "--config",
                str(path),
                "--runner",
                str(args.runner.resolve()),
                "--work-dir",
                str(work),
            ],
            check=False,
        )
        assert completed.returncode == 0, completed.returncode
        manifest = json.loads((work / "domain-loop-manifest.json").read_text(encoding="utf-8"))
        assert manifest["qualified"] is True
        assert manifest["evidence_class"] == "synthetic-domain-qualified"
        assert manifest["best_frontend"] in {"logmel", "pcen-lite"}
        assert {row["frontend"] for row in manifest["records"]} == {"logmel", "pcen-lite"}
        far = manifest["qualification_domains"]["domains"]["distance:far"]
        assert int(far["expected"]) >= 1
        assert pathlib.Path(work / "best" / "model.kwm").is_file()
        assert pathlib.Path(work / "best" / "keywords.kwk").is_file()

    print("test_domain_loop: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
