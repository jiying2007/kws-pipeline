#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import iterate_domain  # noqa: E402
from wake_pressure_balance import (  # noqa: E402
    WAKE_BALANCE_POLICY,
    derive_wake_pressure_balance,
    static_replay_focus_rows,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="base-wake-balance-") as tmp:
        root = pathlib.Path(tmp)
        tokens = root / "tokens.txt"
        keywords = root / "keywords.tsv"
        train_manifest = root / "train.tsv"
        replay_manifest = root / "replay.tsv"
        failure_manifest = root / "failure-replay.tsv"
        tokens.write_text("<blk> 0\na 1\nb 2\nc 3\nd 4\n", encoding="utf-8")
        keywords.write_text(
            "0\twake-zero\t0.55\ta b\n"
            "2\twake-two\t0.55\tc d\n",
            encoding="utf-8",
        )
        train_manifest.write_text(
            "w0.wav\t1 2\n"
            "w2.wav\t3 4\n"
            "n0.wav\t1 3\n"
            "n1.wav\t1 3\n",
            encoding="utf-8",
        )
        replay_manifest.write_text(
            "r0.wav\t1 3\n"
            "r1.wav\t1 3\n"
            "rp0.wav\t1 2\n"
            "rp2.wav\t3 4\n",
            encoding="utf-8",
        )
        failure_manifest.write_text(
            "f0.wav\t1 3\n",
            encoding="utf-8",
        )
        static = {
            "manifest": str(replay_manifest),
            "examples": 4,
            "sequences": [
                {"examples": 2, "focus_keyword_id": 0},
            ],
            "positive_stress": [
                {"examples": 1, "keyword_id": 0},
                {"examples": 1, "keyword_id": 2},
            ],
        }
        focus = static_replay_focus_rows(static)
        assert focus[replay_manifest.resolve()] == [(0,), (0,), (0,), (2,)]

        balance = derive_wake_pressure_balance(
            manifests=[train_manifest, replay_manifest],
            tokens=tokens,
            keywords=keywords,
            positive_example_weight=2.0,
            focus_rows_by_manifest=focus,
        )
        assert balance["policy"] == WAKE_BALANCE_POLICY
        assert balance["wake_rows_by_keyword"] == {"0": 2, "2": 2}
        assert balance["explicit_focus_nonwake_rows"] == 2
        assert balance["fallback_edit_distance_nonwake_rows"] == 2
        assert set(balance["wake_keyword_weights"]) == {"0", "2"}
        assert balance["wake_keyword_weights"]["0"] >= 1.0
        assert balance["wake_keyword_weights"]["2"] >= 1.0
        assert abs(
            balance["effective_wake_mass"]
            - sum(
                item["effective_wake_mass"]
                for item in balance["keyword_balance"].values()
            )
        ) < 1.0e-12

        commands: list[list[str]] = []
        original_run = iterate_domain.run
        try:
            iterate_domain.run = lambda argv: commands.append(list(argv))
            cfg = {
                "seed": 1337,
                "model": {"feature_dim": 32, "hidden_dim": 64},
                "train": {
                    "epochs": 12,
                    "warm_start_epochs": 6,
                    "batch_size": 16,
                    "lr": 0.001,
                    "feature_cache_max_items": 8192,
                },
                "domain_iteration": {
                    "lr_decay_per_round": 0.85,
                    "training_seed_stride": 1009,
                },
            }
            model, checkpoint = iterate_domain.build_torch(
                cfg=cfg,
                frontend="logmel",
                tokens=tokens,
                keywords=keywords,
                manifest=train_manifest,
                output=root / "candidate",
                previous=None,
                hard_negative_manifest=replay_manifest,
                failure_replay_manifest=failure_manifest,
                wake_balance=balance,
                warm_start_strategy="full",
                round_index=0,
            )
        finally:
            iterate_domain.run = original_run

        assert model.name == "model.kwm"
        assert checkpoint.name == "model.pt"
        assert len(commands) == 2
        train_command = commands[0]
        assert "--positive-example-weight" in train_command
        assert "--wake-example-weight" in train_command
        assert "--wake-keyword-weights" in train_command
        weights = json.loads(
            train_command[train_command.index("--wake-keyword-weights") + 1]
        )
        assert weights == balance["wake_keyword_weights"]
        assert train_command.count("--manifest") == 3
        manifests = [
            train_command[index + 1]
            for index, value in enumerate(train_command)
            if value == "--manifest"
        ]
        assert str(failure_manifest) in manifests

    trainer_source = (ROOT / "training" / "train_ctc.py").read_text(
        encoding="utf-8"
    )
    assert "from synthetic_audio import UINT32_MAX" in trainer_source
    assert "keyword_id < 0 or keyword_id > UINT32_MAX" in trainer_source
    assert "keyword id must be unique and fit uint32" in trainer_source
    assert "keyword_id <= 0" not in trainer_source

    product = json.loads(
        (ROOT / "configs/training/xiaowo.torch-domain.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        product["train"]["wake_pressure_balance_policy"]
        == WAKE_BALANCE_POLICY
    )
    assert product["train"]["positive_example_weight"] == 2.0

    print("base wake pressure balance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
