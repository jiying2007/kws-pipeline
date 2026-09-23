#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import adversarial_lexicon as adversarial  # noqa: E402
from synthetic_audio import command_tts_text, write_wav  # noqa: E402


def main() -> int:
    active_tokens = ["ni3", "hao3", "xiao3", "wo1"]
    forbidden = [
        ["ni3", "hao3", "xiao3", "wo1"],
        ["xiao3", "wo1", "xiao3", "wo1"],
    ]
    candidates = adversarial.enumerate_safe_sequences(
        active_tokens,
        forbidden,
        max_length=5,
    )
    assert len(candidates) == 1330
    assert len(candidates) * 2 == 2660
    assert len(
        adversarial.enumerate_safe_sequences(
            active_tokens,
            forbidden,
            max_length=5,
            max_sequences=1330,
        )
    ) == 1330
    try:
        adversarial.enumerate_safe_sequences(
            active_tokens,
            forbidden,
            max_length=5,
            max_sequences=100,
        )
    except ValueError as exc:
        assert "exceeds configured budget" in str(exc)
    else:
        raise AssertionError("adversarial search-space budget was not enforced")

    ranked = [
        {"tokens": ["ni3"], "focus_keyword_id": 1, "max_confidence": 0.9},
        {"tokens": ["hao3"], "focus_keyword_id": 2, "max_confidence": 0.8},
    ]
    keywords = [{"id": 1, "tokens": ["ni3", "hao3"]}, {"id": 2, "tokens": ["xiao3", "wo1"]}]
    selected = adversarial.select_adversarial_candidates(
        ranked,
        keywords,
        top_k=2,
        min_per_keyword=1,
        include_strict_prefix_anchors=False,
    )
    assert len(selected) == 2

    assert adversarial._command_tts_worker_count(0, cpu_count=8) == 0
    assert adversarial._command_tts_worker_count(1, cpu_count=8) == 1
    assert adversarial._command_tts_worker_count(8, cpu_count=1) == 1
    assert adversarial._command_tts_worker_count(8, cpu_count=2) == 2
    assert adversarial._command_tts_worker_count(8, cpu_count=8) == 4
    surface_forms = {
        "ni3": "你",
        "hao3": "好",
        "xiao3": "小",
        "wo1": "窝",
    }
    assert command_tts_text(["ni3", "hao3", "xiao3", "wo1"], surface_forms) == "你好小窝"

    original_render = adversarial.render_command_tts
    original_augment = adversarial.augment
    original_sample_scene = adversarial.sample_scene
    original_focus = adversarial.hard_negative_stress_focus
    original_apply_focus = adversarial.apply_focus
    original_render_scene = adversarial.render_scene

    lock = threading.Lock()
    active = 0
    maximum_active = 0
    calls: list[tuple[str, tuple[str, ...], str, str]] = []

    def fake_render_command_tts(
        text: str,
        token_names: list[str],
        kind: str,
        output: pathlib.Path,
        cfg: dict,
    ) -> list[int]:
        del cfg
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            payload = [
                len(text) * 10,
                len(token_names) * 20,
                len(output.name) * 30,
                -len(output.name) * 30,
            ]
            write_wav(output, payload)
            with lock:
                calls.append((text, tuple(token_names), kind, output.name))
            return payload
        finally:
            with lock:
                active -= 1

    try:
        adversarial.render_command_tts = fake_render_command_tts
        adversarial.augment = lambda clean, rng, cfg: list(clean)
        adversarial.sample_scene = (
            lambda domains, rng, curriculum_weights=None, forced_band=None: {"distance": "near"}
        )
        adversarial.hard_negative_stress_focus = (
            lambda domains, round_index, item_index, example_index: {"stress": False}
        )
        adversarial.apply_focus = lambda scene, focus, domains: dict(scene)
        adversarial.render_scene = (
            lambda samples, scene, seed, afe: (list(samples), {"seed": seed})
        )

        with tempfile.TemporaryDirectory(prefix="adversarial-command-tts-") as tmp:
            root = pathlib.Path(tmp)
            tasks = [
                (["ni3", "hao3"], root / f"q{index:02d}.wav")
                for index in range(6)
            ]
            workers = adversarial._pre_render_command_tts(
                tasks,
                {"backend": "command"},
                surface_forms,
                workers=2,
            )
            assert workers == 2
            assert maximum_active == 2
            assert sorted(row[3] for row in calls) == sorted(path.name for _, path in tasks)
            for token_names, path in tasks:
                assert path.is_file()
                samples = adversarial._read_command_tts_output(path)
                assert samples[0] == len(command_tts_text(token_names, surface_forms)) * 10

            serial_dir = root / "serial"
            parallel_dir = root / "parallel"
            serial_path = serial_dir / "same-name.wav"
            parallel_path = parallel_dir / "same-name.wav"
            token_names = ["xiao3", "wo1", "xiao3"]
            render_args = {
                "token_names": token_names,
                "sequence_index": 7,
                "example_index": 1,
                "round_index": 2,
                "seed": 1337,
                "carriers": {},
                "tts": {"backend": "command"},
                "augment_config": {},
                "domains": {"afe": {}},
                "command_surface_forms": surface_forms,
            }

            serial_samples, serial_meta = adversarial._render_sequence(
                **render_args,
                output_path=serial_path,
                command_tts_pre_rendered=False,
            )
            before = len(calls)
            adversarial._pre_render_command_tts(
                [(token_names, parallel_path)],
                {"backend": "command"},
                surface_forms,
                workers=1,
            )
            after_prerender = len(calls)
            assert after_prerender == before + 1
            parallel_samples, parallel_meta = adversarial._render_sequence(
                **render_args,
                output_path=parallel_path,
                command_tts_pre_rendered=True,
            )
            assert len(calls) == after_prerender
            assert parallel_samples == serial_samples
            assert parallel_meta == serial_meta

            tone_workers = adversarial._pre_render_command_tts(
                [(["ni3"], root / "tone.wav")],
                {"backend": "tone"},
                {},
                workers=2,
            )
            assert tone_workers == 0
            assert not (root / "tone.wav").exists()
        source = (ROOT / "training/adversarial_lexicon.py").read_text(
            encoding="utf-8"
        )
        for key in (
            '"probe_tts_prerender"',
            '"probe_render_and_score"',
            '"replay_tts_prerender"',
            '"replay_render_and_materialize"',
            '"total_to_manifest"',
        ):
            assert key in source
        assert '"adversarial-mining-timing-v1"' in source
        semantic_block = source.split("evidence = {", 1)[1].split(
            'timing_path = output / "adversarial-lexicon-timing.json"', 1
        )[0]
        assert '"timing_seconds"' not in semantic_block

    finally:
        adversarial.render_command_tts = original_render
        adversarial.augment = original_augment
        adversarial.sample_scene = original_sample_scene
        adversarial.hard_negative_stress_focus = original_focus
        adversarial.apply_focus = original_apply_focus
        adversarial.render_scene = original_render_scene

    print("adversarial command TTS parallel prerender: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
