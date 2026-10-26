#!/usr/bin/env python3
from __future__ import annotations

import json
import io
import itertools
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import pathlib
import random
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "eval")]
import domain_metrics
import gate_robustness
from domain_curriculum import update_curriculum

from render_domains import (  # noqa: E402
    _deterministic_eval_scene,
    _evaluation_axes,
    validate_domains,
)


def row(name: str, band: str, expected: list[dict], *, azimuth: float | None = None) -> dict:
    return {
        "recording": name,
        "path": f"{name}.wav",
        "duration_s": 10.0,
        "expected": expected,
        "domain": {
            "distance_band": band,
            "distance_m": 0.5 if band == "near" else 4.5,
            "azimuth_deg": (0.0 if band == "near" else 90.0) if azimuth is None else azimuth,
            "rt60_s": 0.2 if band == "near" else 0.7,
            "snr_db": 20.0 if band == "near" else 2.0,
            "noise_profile": "fan" if band == "near" else "motor",
            "playback_sir_db": None if band == "near" else 5.0,
        },
    }


def gate_metric(
    *,
    expected: int = 4,
    positive_recordings: int = 4,
    negative_recordings: int = 4,
    negative_hours: float = 0.01,
    frr: float = 0.0,
    far: float = 0.0,
) -> dict:
    return {
        "expected": expected,
        "positive_recordings": positive_recordings,
        "negative_recordings": negative_recordings,
        "negative_audio_hours": negative_hours,
        "frr": frr,
        "far_per_hour": far,
        "wake_rate": 1.0 - frr,
    }


def run_gate_case(
    root: pathlib.Path,
    summary: dict,
    config: dict,
    name: str,
) -> tuple[int, dict]:
    summary_path = root / f"{name}-summary.json"
    config_path = root / f"{name}-config.json"
    output_path = root / f"{name}-out.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval" / "gate_robustness.py"),
            "--summary",
            str(summary_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        check=False,
    )
    result = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.is_file()
        else {}
    )
    return completed.returncode, result


def assert_formal_triple_stress_support() -> None:
    config = json.loads(
        (ROOT / "configs" / "training" / "xiaowo.torch-domain.json").read_text(
            encoding="utf-8"
        )
    )
    gates = config["robustness_gates"]
    triple = "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical"
    assert triple in gates["required_stress_slices"]
    domains = validate_domains(config)
    axes = _evaluation_axes(config, domains)
    assert axes is not None

    keyword_path = ROOT / str(config["keywords"])
    keyword_count = sum(
        1
        for raw in keyword_path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    )
    qualification = config["dataset"]["qualification"]
    variants = int(qualification["variants_per_family"])
    scenes_per_example = int(config["domains"]["scenes_per_example"]["qualification"])
    positive_scenes = (
        int(qualification["positive_families_per_keyword"])
        * keyword_count
        * variants
        * scenes_per_example
    )
    negative_scenes = (
        (
            int(qualification["confusable_families_per_keyword"]) * keyword_count
            + int(qualification["random_negative_families"])
        )
        * variants
        * scenes_per_example
    )

    def triple_count(count: int) -> tuple[int, set[float]]:
        matches = []
        for index in range(count):
            scene = _deterministic_eval_scene(
                domains,
                axes,
                index,
                random.Random(100_000 + index),
            )
            if (
                float(scene["distance_m"]) > 4.0
                and abs(float(scene["azimuth_deg"])) > 90.0
                and float(scene["snr_db"]) <= 6.0
            ):
                matches.append(scene)
        return len(matches), {float(scene["azimuth_deg"]) for scene in matches}

    positive_count, positive_azimuths = triple_count(positive_scenes)
    negative_count, _ = triple_count(negative_scenes)
    assert positive_count >= int(gates["min_expected_wakes"])
    assert negative_count >= int(gates["min_negative_recordings"])
    assert {-150.0, -120.0, 150.0, 180.0}.issubset(positive_azimuths)



class MissingAxisTests(unittest.TestCase):
    def test_sparse_domain_does_not_invent_axes(self):
        self.assertEqual(domain_metrics.domain_keys({'domain': {'distance_band': 'far'}}),
                         ['all', 'distance:far'])
        self.assertEqual(domain_metrics.domain_keys({'domain': {}}), ['all'])
        self.assertEqual(domain_metrics.domain_keys({}), ['all'])

    def test_explicit_zero_is_not_missing(self):
        keys = domain_metrics.domain_keys({'domain': {'distance_m': 0, 'azimuth_deg': 0,
            'snr_db': 0, 'rt60_s': 0, 'playback_sir_db': None}})
        for key in ('distance_bin:0.5m', 'azimuth:front', 'azimuth_deg:0',
                    'snr:critical', 'rt60:dry', 'playback:no-playback',
                    'distance_azimuth_snr:distance_bin=0.5m|azimuth=front|snr=critical'):
            self.assertIn(key, keys)
        self.assertFalse(any(key.startswith('composite:') for key in keys))

    def test_joint_slices_need_all_constituent_axes(self):
        keys = domain_metrics.domain_keys({'domain': {'distance_m': 5, 'snr_db': 2}})
        self.assertIn('distance_snr:distance_bin=5m|snr=critical', keys)
        for prefix in ('distance_azimuth:', 'azimuth_snr:', 'distance_azimuth_snr:', 'composite:'):
            self.assertFalse(any(key.startswith(prefix) for key in keys), prefix)

    def test_present_invalid_numeric_values_still_rejected(self):
        for axis in ('distance_m', 'azimuth_deg', 'rt60_s', 'snr_db'):
            for value in (None, True, '0', float('nan'), float('inf')):
                with self.subTest(axis=axis, value=value), self.assertRaises(ValueError):
                    domain_metrics.domain_keys({'domain': {axis: value}})

    def test_full_metadata_key_contract_is_unchanged(self):
        self.assertEqual(domain_metrics.domain_keys(row('full', 'near', [])), [
            'all', 'distance:near', 'distance_bin:0.5m', 'azimuth:front', 'azimuth_deg:0',
            'snr:mid', 'distance_azimuth:distance_bin=0.5m|azimuth=front',
            'distance_snr:distance_bin=0.5m|snr=mid', 'azimuth_snr:azimuth=front|snr=mid',
            'distance_azimuth_snr:distance_bin=0.5m|azimuth=front|snr=mid',
            'rt60:dry', 'noise:fan', 'playback:no-playback',
            'composite:distance=near|az=front|rt60=dry|noise=fan|no-playback'])

    def test_every_partial_field_set_preserves_only_supported_slices(self):
        full = row('full', 'near', [])['domain']
        requirements = [
            ('all', set()), ('distance:near', {'distance_band'}),
            ('distance_bin:0.5m', {'distance_m'}), ('azimuth:front', {'azimuth_deg'}),
            ('azimuth_deg:0', {'azimuth_deg'}), ('snr:mid', {'snr_db'}),
            ('distance_azimuth:distance_bin=0.5m|azimuth=front', {'distance_m', 'azimuth_deg'}),
            ('distance_snr:distance_bin=0.5m|snr=mid', {'distance_m', 'snr_db'}),
            ('azimuth_snr:azimuth=front|snr=mid', {'azimuth_deg', 'snr_db'}),
            ('distance_azimuth_snr:distance_bin=0.5m|azimuth=front|snr=mid',
             {'distance_m', 'azimuth_deg', 'snr_db'}),
            ('rt60:dry', {'rt60_s'}), ('noise:fan', {'noise_profile'}),
            ('playback:no-playback', {'playback_sir_db'}),
            ('composite:distance=near|az=front|rt60=dry|noise=fan|no-playback',
             {'distance_band', 'azimuth_deg', 'rt60_s', 'noise_profile', 'playback_sir_db'})]
        for count in range(len(full) + 1):
            for fields in itertools.combinations(full, count):
                with self.subTest(fields=fields):
                    domain = {name: full[name] for name in fields}
                    expected = [key for key, needed in requirements if needed <= set(fields)]
                    self.assertEqual(domain_metrics.domain_keys({'domain': domain}), expected)

    def test_sparse_report_cannot_satisfy_unknown_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            refs, dets, out = root/'refs.jsonl', root/'dets.jsonl', root/'out.json'
            rows = [row('p', 'far', [{'keyword_id': 1, 'start_s': 1., 'end_s': 2.}]),
                    row('n', 'far', [])]
            for item in rows:
                item['domain'] = {'distance_band': 'far'}
            refs.write_text(''.join(json.dumps(item)+'\n' for item in rows))
            dets.write_text(json.dumps({'recording': 'p', 'keyword_id': 1,
                                       'time_s': 2., 'confidence': .9})+'\n')
            with patch.object(sys, 'argv', ['domain_metrics', '--references', str(refs),
                    '--detections', str(dets), '--output', str(out)]), redirect_stdout(io.StringIO()):
                self.assertEqual(domain_metrics.main(), 0)
            result = json.loads(out.read_text())
            self.assertEqual(set(result['domains']), {'all', 'distance:far'})
            self.assertEqual(set(result['keyword_domains']['1']['domains']), {'all', 'distance:far'})
            self.assertEqual(result['overall']['expected'], 1)
            self.assertEqual(result['overall']['negative_recordings'], 1)
            self.assertEqual(result['overall']['frr'], 0.)
            self.assertEqual(result['overall']['far_per_hour'], 0.)
            config = {'robustness_gates': {'required_distance_bins': ['0.5m'],
                'required_azimuth_deg': [0], 'required_snr_bands': ['critical'],
                'required_rt60_bands': ['dry']}}
            gate = gate_robustness.evaluate({'qualification_domains': result}, config)
            self.assertFalse(gate['qualified'])
            self.assertEqual({item['reason'] for item in gate['failures']}, {'missing-slice'})
            curriculum = update_curriculum(result)
            self.assertNotIn('azimuth', curriculum['dimension_weights'])
            self.assertNotIn('snr', curriculum['dimension_weights'])


def main() -> int:
    assert_formal_triple_stress_support()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        refs = root / "refs.jsonl"
        dets = root / "dets.jsonl"
        out = root / "domains.json"
        references = [
            row("near", "near", [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]),
            row("far", "far", [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]),
            row("far-bg", "far", [], azimuth=180.0),
            row(
                "overlap",
                "near",
                [
                    {"keyword_id": 1, "start_s": 4.0, "end_s": 5.0},
                    {"keyword_id": 2, "start_s": 4.1, "end_s": 5.1},
                ],
            ),
        ]
        refs.write_text("\n".join(json.dumps(item) for item in references) + "\n", encoding="utf-8")
        detections = [
            {"recording": "near", "keyword_id": 1, "time_s": 2.1, "confidence": 0.9},
            {"recording": "far", "keyword_id": 2, "time_s": 2.1, "confidence": 0.8},
            {"recording": "far-bg", "keyword_id": 1, "time_s": 5.0, "confidence": 0.7},
            {"recording": "overlap", "keyword_id": 2, "time_s": 5.1, "confidence": 0.95},
        ]
        dets.write_text("\n".join(json.dumps(item) for item in detections) + "\n", encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "domain_metrics.py"),
                "--references",
                str(refs),
                "--detections",
                str(dets),
                "--output",
                str(out),
            ]
        )
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["schema_version"] == 6
        assert result["domains"]["distance:near"]["frr"] > 0.0
        assert result["domains"]["distance:far"]["frr"] == 1.0
        assert result["domains"]["distance_bin:0.5m"]["expected"] == 3
        assert result["domains"]["distance_bin:5m"]["false_accepts"] == 2
        assert result["domains"]["distance_bin:5m"]["negative_recordings"] == 1
        assert result["domains"]["azimuth_deg:0"]["expected"] == 3
        assert result["domains"]["azimuth_deg:180"]["negative_audio_hours"] > 0.0
        assert result["domains"]["azimuth_deg:180"]["negative_recordings"] == 1
        assert result["domains"]["snr:critical"]["false_accepts"] == 2
        assert result["domains"]["snr:mid"]["expected"] == 3
        assert 0.0 <= result["domains"]["distance_bin:0.5m"]["wake_rate"] <= 1.0
        assert result["domains"]["azimuth:rear"]["expected"] == 0
        assert result["domains"]["azimuth:rear"]["negative_audio_hours"] > 0.0
        assert result["domains"]["azimuth:rear"]["false_accepts"] == 1
        assert result["domains"]["azimuth:rear"]["negative_recording_false_accepts"] == 1
        assert result["domains"]["azimuth:rear"]["negative_recording_audio_hours"] > 0.0
        assert result["domains"]["azimuth:rear"]["negative_recording_far_per_hour"] > 0.0
        assert (
            result["domains"]["azimuth:rear"]["negative_recording_far_upper_95_per_hour"]
            > result["domains"]["azimuth:rear"]["negative_recording_far_per_hour"]
        )
        assert result["domains"]["distance_azimuth:distance_bin=5m|azimuth=rear"]["false_accepts"] == 1
        assert result["domains"]["distance_snr:distance_bin=5m|snr=critical"]["false_accepts"] == 2
        assert result["domains"]["azimuth_snr:azimuth=rear|snr=critical"]["negative_recordings"] == 1
        triple = "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical"
        assert result["domains"][triple]["false_accepts"] == 1
        assert result["domains"][triple]["negative_recordings"] == 1
        assert result["worst_domain"] is not None
        assert result["slice_contract"]["azimuth_quantization_deg"] == 30
        assert "distance_snr" in result["slice_contract"]["pairwise"]
        assert "distance_azimuth_snr" in result["slice_contract"]["triple"]
        assert "keyword" in result["slice_contract"]["keyword_conditioning"]

        confusion = result["keyword_confusion"]
        assert confusion["assignment"] == "global-monotonic-one-to-one-v1"
        assert confusion["expected_events"] == 4
        assert confusion["correct_keyword"] == 2
        assert confusion["wrong_keyword"] == 1
        assert confusion["missed"] == 1
        assert confusion["matrix"]["1"]["2"] >= 1

        keyword_domains = result["keyword_domains"]
        assert set(keyword_domains) == {"1", "2"}
        assert keyword_domains["1"]["overall"]["expected"] == 3
        assert keyword_domains["2"]["overall"]["expected"] == 1
        assert keyword_domains["1"]["domains"][triple]["false_accepts"] == 1
        assert keyword_domains["1"]["domains"][triple]["negative_recordings"] == 1
        assert keyword_domains["2"]["domains"][triple]["false_accepts"] == 0
        assert keyword_domains["1"]["worst_domain"] is not None

        stress = [
            "distance_azimuth:distance_bin=5m|azimuth=rear",
            "distance_snr:distance_bin=5m|snr=critical",
            "azimuth_snr:azimuth=rear|snr=critical",
            triple,
        ]
        gate_config = {
            "robustness_gates": {
                "max_frr": 0.0,
                "max_far_per_hour": 0.0,
                "min_expected_wakes": 4,
                "min_negative_recordings": 4,
                "min_negative_audio_hours": 0.0,
                "required_distance_bins": ["0.5m", "5m"],
                "required_azimuth_deg": [0, 180],
                "required_snr_bands": ["critical", "high"],
                "required_stress_slices": stress,
            }
        }
        gate_domains = {
            "distance_bin:0.5m": gate_metric(),
            "distance_bin:5m": gate_metric(),
            "azimuth_deg:0": gate_metric(),
            "azimuth_deg:180": gate_metric(),
            "snr:critical": gate_metric(),
            "snr:high": gate_metric(),
            **{key: gate_metric() for key in stress},
        }
        gate_summary = {"qualification_domains": {"domains": gate_domains}}

        code, gate_result = run_gate_case(root, gate_summary, gate_config, "pass")
        assert code == 0
        assert gate_result["schema_version"] == 4
        assert gate_result["qualified"] is True
        assert not gate_result["failures"]
        assert gate_result["slices"]["distance_bin:5m"]["wake_rate"] == 1.0
        assert gate_result["gates"]["required_stress_slices"] == stress

        failing = json.loads(json.dumps(gate_summary))
        failing["qualification_domains"]["domains"][triple]["frr"] = 0.25
        failing["qualification_domains"]["domains"][triple]["wake_rate"] = 0.75
        code, gate_result = run_gate_case(root, failing, gate_config, "triple-frr-fail")
        assert code == 1
        assert gate_result["qualified"] is False
        assert any(
            item["slice"] == triple and item["reason"] == "frr"
            for item in gate_result["failures"]
        )

        unsupported = json.loads(json.dumps(gate_summary))
        unsupported["qualification_domains"]["domains"][triple]["expected"] = 2
        unsupported["qualification_domains"]["domains"][triple]["negative_recordings"] = 2
        code, gate_result = run_gate_case(root, unsupported, gate_config, "support-fail")
        assert code == 1
        reasons = {
            item["reason"]
            for item in gate_result["failures"]
            if item["slice"] == triple
        }
        assert reasons == {"insufficient-positive-support", "insufficient-negative-recordings"}

        invalid_config = json.loads(json.dumps(gate_config))
        invalid_config["robustness_gates"]["required_stress_slices"] = ["composite:anything"]
        code, gate_result = run_gate_case(root, gate_summary, invalid_config, "invalid-stress")
        assert code == 2
        assert gate_result == {}

    print("test_domain_metrics: ok")
    return 0


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(MissingAxisTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(main())
