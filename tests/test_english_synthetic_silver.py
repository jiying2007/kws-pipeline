"""Byte/label boundary fixtures only; no speech or model execution evidence."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("english_silver", ROOT / "training/english_synthetic_silver.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SilverContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.profile = json.loads((ROOT / "configs/research/english-pronunciation-profile-20261115.json").read_text())
        identities = [dict(recognizer_id=name, model_family=module.MODEL_PINS[name][0],
                           model_sha256=module.MODEL_PINS[name][1],
                           decode_config_sha256=module.MODEL_PINS[name][2])
                      for name in ("qwen", "whisper", "parakeet", "phoneme")]
        groups = []
        for index in range(24):
            groups.append(dict(source_group_id="libritts-speaker-" + str(index + 100),
                               source_speaker_id="p" + str(index + 100), source_speaker_index=index + 10,
                               voice_family="piper-libritts-" + str(index + 100),
                               role=module.ROLES[0 if index < 12 else 1 + (index - 12) // 4]))
        cells = [dict(cell_id="mero-natural", requested_text="Hey Mero.", requested_class=2,
                      pronunciation_hypothesis="mero-us-eh-v1"),
                 dict(cell_id="mero-alt", requested_text="Hey Mero.", requested_class=2,
                      pronunciation_hypothesis="mero-us-eh-v1"),
                 dict(cell_id="vosen-natural", requested_text="Hey Vosen.", requested_class=1,
                      pronunciation_hypothesis="vosen-us-s-v1"),
                 dict(cell_id="vosen-alt", requested_text="Hey Vosen.", requested_class=1,
                      pronunciation_hypothesis="vosen-us-s-v1")]
        cells.extend(dict(cell_id="ordinary-" + str(i), requested_text=text, requested_class=0,
                          pronunciation_hypothesis="ordinary-negative-v1")
                     for i, text in enumerate(module.ORDINARY))
        cls.plan = {"silver_admission": dict(rule_id=module.RULE_ID, rules_sha256=module.RULES_SHA256,
                    recognizers=identities[:3], phone_recognizer=identities[3], source_groups=groups,
                    cells=cells, excluded_source_speaker_ids=["p0", "p1"],
                    supervised_tiers=["silver", "weak"], training_tier_weights={"silver": 1.0, "weak": 0.5},
                    minimum_balanced_groups=dict(train=6, development=2, calibration=2, **{"locked-test": 2}))}
        phone_text = {"mero-natural": "h eɪ m ɛ ɹ oʊ", "mero-alt": "h eɪ m ɛ ɹ oʊ",
                      "vosen-natural": "h eɪ v oʊ s ə n", "vosen-alt": "h eɪ v oʊ s ə n"}
        rows, evidence = [], []
        for group in groups:
            for cell in cells:
                number = len(rows)
                row = dict(id="fixture-" + str(number), **group, **cell, generation_status="success",
                           sample_rate=16000, channels=1, pcm_samples=6400, pcm_path=None)
                pcm = struct.pack("<6400h", *([1000 + number] * 6400))
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as output:
                    output.setparams((1, 2, 16000, 6400, "NONE", "not compressed"))
                    output.writeframes(pcm)
                wav = buffer.getvalue()
                row.update(wav_path="audio/" + row["id"] + ".wav", wav_sha256=module.sha(wav),
                           pcm_sha256=module.sha(pcm))
                (cls.root / "audio").mkdir(exist_ok=True)
                (cls.root / row["wav_path"]).write_bytes(wav)
                packet = dict(id=row["id"], observations=[])
                for identity in identities:
                    capture = dict(**identity, source_pcm_sha256=row["pcm_sha256"],
                                   source_wav_sha256=row["wav_sha256"], source_samples=6400,
                                   status="success", target_conditioned=False,
                                   input_coverage_complete=True, decoder_output_complete=True)
                    if identity["recognizer_id"] == "phoneme":
                        capture["phone_tokens"] = phone_text.get(cell["cell_id"], "w ʌ t t aɪ m ɪ z ɪ t").split()
                    else:
                        capture["raw_text"] = ("Hey Vosen." if row["requested_class"] == 1 else row["requested_text"])
                    path = "captures/" + row["id"] + "-" + identity["recognizer_id"] + ".json"
                    raw = module.canonical(capture)
                    (cls.root / "captures").mkdir(exist_ok=True)
                    (cls.root / path).write_bytes(raw)
                    observation = dict(**capture, capture_path=path, capture_sha256=module.sha(raw))
                    if identity["recognizer_id"] == "phoneme":
                        packet["phone_observation"] = observation
                    else:
                        packet["observations"].append(observation)
                rows.append(row)
                evidence.append(packet)
        cls.corpus = dict(schema="english-synthetic192-corpus-v1", rows=rows)
        cls.evidence = dict(schema="english-synthetic-machine-evidence-v1", rows=evidence)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.c = copy.deepcopy(self.corpus)
        self.e = copy.deepcopy(self.evidence)
        self.p = copy.deepcopy(self.plan)

    def run_contract(self, **kwargs):
        return module.build_admission(kwargs.get("corpus", self.c), kwargs.get("evidence", self.e), self.root,
                                      profile=kwargs.get("profile", self.profile), plan=kwargs.get("plan", self.p))

    def change_capture(self, observation, **changes):
        observation.update(changes)
        capture = {k: v for k, v in observation.items() if k not in ("capture_path", "capture_sha256")}
        raw = module.canonical(capture)
        path = "captures/changed-" + module.sha(raw) + ".json"
        (self.root / path).write_bytes(raw)
        observation.update(capture_path=path, capture_sha256=module.sha(raw))

    def test_synthetic_only_training_and_heldout_roles(self):
        result = self.run_contract()
        self.assertTrue(result["training_snapshot_allowed"])
        self.assertEqual(result["tier_counts"], {"silver": 192})
        self.assertEqual(result["balanced_groups_by_role"], dict(train=12, development=4, calibration=4, **{"locked-test": 4}))
        for row in result["rows"]:
            self.assertFalse(row["human_gold"])
            self.assertFalse(row["independent_real_evaluation_allowed"])
            self.assertEqual(row["training_allowed"], row["role"] == "train")
            if row["role"] != "train":
                self.assertIsNone(row["training_class"])
                self.assertEqual(row["training_weight"], 0)

    def test_three_aliases_are_silver_without_raw_spelling_equality(self):
        for observation, text in zip(self.e["rows"][0]["observations"], ["Hey Mero.", "Hey Merrow!", "hey marrow"]):
            self.change_capture(observation, raw_text=text)
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "silver")

    def test_superseded_meero_miro_and_voiced_vosen_not_accepted(self):
        for spelling in ("Hey Meero.", "Hey Miro.", "Hay Meero.", "Hey Vozen."):
            with self.subTest(spelling=spelling):
                self.e = copy.deepcopy(self.evidence)
                self.change_capture(self.e["rows"][0]["observations"][2], raw_text=spelling)
                self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_superseded_requested_ipa_rejected_before_training(self):
        for idx, text, hypothesis in ((1, "Hey Meero.", "mero-us-i-v1"),
                                      (3, "Hey Vosen.", "vosen-us-z-v1")):
            with self.subTest(hypothesis=hypothesis):
                self.p = copy.deepcopy(self.plan)
                self.p["silver_admission"]["cells"][idx].update(
                    requested_text=text, pronunciation_hypothesis=hypothesis)
                with self.assertRaisesRegex(ValueError, "HYPOTHESIS_CLASS"):
                    self.run_contract()

    def test_voiced_vosen_and_meer_phones_never_admitted(self):
        for idx, phones in ((0, ["h", "eɪ", "m", "iː", "r", "oʊ"]),
                            (2, ["h", "eɪ", "v", "oʊ", "z", "ə", "n"])):
            with self.subTest(idx=idx):
                self.e = copy.deepcopy(self.evidence)
                self.change_capture(self.e["rows"][idx]["phone_observation"], phone_tokens=phones)
                self.assertEqual(self.run_contract()["rows"][idx]["tier"], "unknown")

    def test_two_votes_remain_weak_and_are_weighted_only_in_train(self):
        self.change_capture(self.e["rows"][0]["observations"][2], raw_text="Hey Merro.")
        row = self.run_contract()["rows"][0]
        self.assertEqual(row["tier"], "weak")
        self.assertEqual(row["training_weight"], 0.5)
        self.assertEqual(row["label_kind"], "synthetic_weak_machine_research")

    def test_other_wake_class_vetoes_two_votes(self):
        self.change_capture(self.e["rows"][0]["observations"][2], raw_text="Hey Vosen.")
        row = self.run_contract()["rows"][0]
        self.assertEqual(row["tier"], "unknown")
        self.assertIsNone(row["training_class"])

    def test_missing_asr_or_phone_never_becomes_background(self):
        self.e["rows"][0]["observations"].pop()
        self.e["rows"][4]["phone_observation"] = {"status": "missing"}
        result = self.run_contract()
        for index in (0, 4):
            self.assertEqual(result["rows"][index]["tier"], "unknown")
            self.assertIsNone(result["rows"][index]["research_class"])

    def test_predeclared_core_contradiction_vetoes_weak_label(self):
        self.change_capture(self.e["rows"][0]["observations"][2], raw_text="Hey Nero.")
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_unclear_hey_phrase_cannot_be_ordinary_negative(self):
        self.change_capture(self.e["rows"][4]["observations"][2], raw_text="Hey Rosen.")
        self.assertEqual(self.run_contract()["rows"][4]["tier"], "unknown")

    def test_request_and_three_text_votes_cannot_override_incompatible_phones(self):
        self.change_capture(self.e["rows"][0]["phone_observation"], phone_tokens=["h", "eɪ", "n", "ɛ", "r", "oʊ"])
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_discriminative_vowels_and_extra_core_phones_are_not_deleted(self):
        for phones in (["h", "eɪ", "m", "e", "r", "oʊ"],
                       ["h", "eɪ", "m", "ɪ", "r", "oʊ"],
                       ["h", "eɪ", "m", "ɛ", "n", "r", "oʊ"]):
            with self.subTest(phones=phones):
                self.change_capture(self.e["rows"][0]["phone_observation"], phone_tokens=phones)
                self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_self_reported_tier_cannot_replace_evidence(self):
        self.c["rows"][0].update(tier="silver", training_admitted=True, human_gold=True)
        self.e["rows"][0].update(tier="silver", training_admitted=True)
        self.e["rows"][0]["phone_observation"] = {"status": "missing"}
        row = self.run_contract()["rows"][0]
        self.assertEqual(row["tier"], "unknown")
        self.assertFalse(row["training_allowed"])
        self.assertFalse(row["human_gold"])

    def test_claimed_success_without_real_source_bytes_fails(self):
        self.c["rows"][0]["wav_path"] = "audio/missing.wav"
        with self.assertRaisesRegex(ValueError, "MEMBER_MISSING_OR_SIZE"):
            self.run_contract()

    def test_hey_nucleus_must_be_present(self):
        self.change_capture(self.e["rows"][0]["phone_observation"], phone_tokens=["m", "ɛ", "r", "oʊ"])
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_weak_h_onset_needs_three_text_votes(self):
        self.change_capture(self.e["rows"][0]["phone_observation"], phone_tokens=["eɪ", "m", "ɛ", "r", "oʊ"])
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "silver")
        self.change_capture(self.e["rows"][0]["observations"][2], raw_text="Hey Merro.")
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_ordinary_negative_needs_positive_sentence_support(self):
        for observation in self.e["rows"][4]["observations"]:
            self.change_capture(observation, raw_text="")
        self.assertEqual(self.run_contract()["rows"][4]["tier"], "unknown")

    def test_any_wake_in_ordinary_negative_is_unknown(self):
        self.change_capture(self.e["rows"][4]["observations"][2], raw_text="What time is it? Hey Mero.")
        self.assertEqual(self.run_contract()["rows"][4]["tier"], "unknown")

    def test_target_conditioning_is_not_independent_observation(self):
        self.change_capture(self.e["rows"][0]["phone_observation"], target_conditioned=True)
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_hash_tampering_fails_closed(self):
        self.e["rows"][0]["observations"][0]["capture_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "MEMBER_HASH_MISMATCH"):
            self.run_contract()

    def test_outer_transcript_cannot_replace_capture(self):
        self.e["rows"][0]["observations"][0]["raw_text"] = "Hey Vosen."
        with self.assertRaisesRegex(ValueError, "CAPTURE_SEMANTIC_MISMATCH"):
            self.run_contract()

    def test_foreign_pcm_observation_fails(self):
        self.change_capture(self.e["rows"][0]["observations"][0], source_pcm_sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "OBSERVATION_SOURCE_DRIFT"):
            self.run_contract()

    def test_split_move_fails_before_training(self):
        self.c["rows"][0]["role"] = "development"
        with self.assertRaisesRegex(ValueError, "ROW_PLAN_OR_SPLIT_DRIFT"):
            self.run_contract()

    def test_duplicate_speaker_alias_cannot_cross_roles(self):
        self.p["silver_admission"]["source_groups"][12]["source_speaker_id"] = self.p["silver_admission"]["source_groups"][0]["source_speaker_id"]
        with self.assertRaisesRegex(ValueError, "SOURCE_ANCESTRY_ALIAS"):
            self.run_contract()

    def test_exposed_source_is_rejected(self):
        self.p["silver_admission"]["excluded_source_speaker_ids"].append(self.c["rows"][0]["source_speaker_id"])
        with self.assertRaisesRegex(ValueError, "EXPOSED_SPEAKER_REUSED"):
            self.run_contract()

    def test_repeated_pcm_cannot_manufacture_new_source(self):
        for key in ("wav_path", "wav_sha256", "pcm_sha256"):
            self.c["rows"][1][key] = self.c["rows"][0][key]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PCM_IDENTITY"):
            self.run_contract()

    def test_unbalanced_group_loses_all_supervision_without_resplit(self):
        self.e["rows"][0]["phone_observation"] = {"status": "missing"}
        self.e["rows"][1]["phone_observation"] = {"status": "missing"}
        result = self.run_contract()
        self.assertEqual(result["balanced_groups_by_role"]["train"], 11)
        self.assertTrue(all(not r["supervision_allowed"] for r in result["rows"][:8]))
        self.assertTrue(all(r["role"] == "train" for r in result["rows"][:8]))

    def test_predeclared_floor_fails_without_weak_upgrade(self):
        self.p["silver_admission"]["minimum_balanced_groups"]["train"] = 12
        self.e["rows"][0]["phone_observation"] = {"status": "missing"}
        self.e["rows"][1]["phone_observation"] = {"status": "missing"}
        self.assertFalse(self.run_contract()["training_snapshot_allowed"])

    def test_generation_failure_is_preserved_as_unknown(self):
        self.c["rows"][0]["generation_status"] = "failed"
        result = self.run_contract()
        self.assertEqual(len(result["rows"]), 192)
        self.assertEqual(result["rows"][0]["tier"], "unknown")

    def test_old_policy_cannot_silently_use_new_intake(self):
        old = copy.deepcopy(self.profile)
        old["schema"] = "english-kws-pronunciation-profile-v2"
        with self.assertRaisesRegex(ValueError, "SYNTHETIC_RESEARCH_PROFILE_REQUIRED"):
            self.run_contract(profile=old)

    def test_rules_or_recognizer_identity_cannot_drift(self):
        self.p["silver_admission"]["rules_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "MACHINE_RULE_NOT_PINNED"):
            self.run_contract()
        self.p = copy.deepcopy(self.plan)
        self.p["silver_admission"]["recognizers"][1]["model_family"] = self.p["silver_admission"]["recognizers"][0]["model_family"]
        with self.assertRaisesRegex(ValueError, "FROZEN_NATIVE_OBSERVER_PIN"):
            self.run_contract()

    def test_arbitrary_models_cannot_mint_votes_even_with_a_matching_plan(self):
        self.p["silver_admission"]["recognizers"][0]["model_sha256"] = "2" * 64
        self.change_capture(self.e["rows"][0]["observations"][0], model_sha256="2" * 64)
        with self.assertRaisesRegex(ValueError, "FROZEN_NATIVE_OBSERVER_PIN"):
            self.run_contract()

    def test_traversal_and_symlink_rejected(self):
        self.c["rows"][0]["wav_path"] = "../external.wav"
        with self.assertRaisesRegex(ValueError, "INVALID_MEMBER_PATH"):
            self.run_contract()
        self.c = copy.deepcopy(self.corpus)
        link = self.root / "linked.wav"
        link.symlink_to(self.root / self.c["rows"][0]["wav_path"])
        self.c["rows"][0]["wav_path"] = "linked.wav"
        with self.assertRaisesRegex(ValueError, "SYMLINK_MEMBER_FORBIDDEN"):
            self.run_contract()


# Reuse the unchanged label/PCM boundary suite for the independent source campaign.
phone_spec = importlib.util.spec_from_file_location("english_phonetic_silver", ROOT / "training/english_phonetic_silver.py")
phonetic = importlib.util.module_from_spec(phone_spec)
phone_spec.loader.exec_module(phonetic)


class PhoneticSourceContractTests(SilverContractTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        groups = []
        for offset, (index, speaker) in enumerate(phonetic.SOURCE_VOICES):
            groups.append(dict(source_group_id=f"libritts-speaker-{speaker}",
                source_speaker_id=f"p{speaker}", source_speaker_index=index,
                voice_family=f"piper-libritts-{speaker}",
                role=module.ROLES[0 if offset < 12 else 1 + (offset - 12) // 4]))
        cls.plan["silver_admission"].update(rule_id=phonetic.RULE_ID, rules_sha256=phonetic.RULES_SHA256,
            source_design_id=phonetic.SOURCE_DESIGN_ID,
            source_groups=groups, excluded_source_speaker_ids=[f"p{x}" for x in phonetic.EXPOSED_SOURCE_IDS])
        for number, row in enumerate(cls.corpus["rows"]):
            row.update(groups[number // 8])

    def run_contract(self, **kwargs):
        return phonetic.build_admission(kwargs.get("corpus", self.c), kwargs.get("evidence", self.e), self.root,
            profile=kwargs.get("profile", self.profile), plan=kwargs.get("plan", self.p))

    def test_v2_changes_only_declared_transcript_compatibility(self):
        expected = copy.deepcopy(module.RULES)
        expected["rule_id"] = phonetic.RULE_ID
        expected["aliases"]["1"] += ["hey vosin"]
        for label in ("1", "2"):
            expected["aliases"][label] += ["hay " + x[4:] for x in list(expected["aliases"][label])]
            expected["contradictory_core_phrases"][label] += ["hay " + x[4:]
                for x in list(expected["contradictory_core_phrases"][label])]
        self.assertEqual(phonetic.canonical(phonetic.RULES), module.canonical(expected))
        self.assertNotEqual(phonetic.RULES_SHA256, module.RULES_SHA256)
        self.assertEqual(phonetic.MODEL_PINS, module.MODEL_PINS)

    def test_new_spellings_need_same_complete_acoustic_support(self):
        row = self.e["rows"][2]
        for observation, text in zip(row["observations"], ["Hey Vosin.", "Hay Vosen.", "Hay Vohsen."]):
            self.change_capture(observation, raw_text=text)
        self.assertEqual(self.run_contract()["rows"][2]["tier"], "silver")
        for phones in (["h", "eɪ", "v", "oʊ", "z", "ɪ", "n"],
                       ["h", "eɪ", "b", "oʊ", "z", "ə", "n"]):
            self.change_capture(row["phone_observation"], phone_tokens=phones)
            self.assertEqual(self.run_contract()["rows"][2]["tier"], "unknown")

    def test_hay_mero_is_spelling_only_not_a_missing_hey_allowance(self):
        for observation in self.e["rows"][0]["observations"]:
            self.change_capture(observation, raw_text="Hay Mero.")
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "silver")
        self.change_capture(self.e["rows"][0]["phone_observation"], phone_tokens=["h", "iː", "m", "ɛ", "r", "oʊ"])
        self.assertEqual(self.run_contract()["rows"][0]["tier"], "unknown")

    def test_two_new_spelling_votes_do_not_override_a_contradictory_core(self):
        for observation, text in zip(self.e["rows"][2]["observations"], ["Hey Vosin.", "Hay Vosen.", "Hey Rosen."]):
            self.change_capture(observation, raw_text=text)
        self.assertEqual(self.run_contract()["rows"][2]["tier"], "unknown")
        for text in ("Hay Rosen.", "Hay Mero.", "Hay Chosen."):
            self.change_capture(self.e["rows"][2]["observations"][2], raw_text=text)
            self.assertEqual(self.run_contract()["rows"][2]["tier"], "unknown")

    def test_new_source_design_cannot_be_omitted(self):
        self.p["silver_admission"].pop("source_design_id")
        with self.assertRaisesRegex(ValueError, "EXACT_NEW_SOURCE_DESIGN"):
            self.run_contract()

    def test_consumed_source_indices_cannot_be_reused(self):
        for g in self.p["silver_admission"]["source_groups"]:
            g["source_speaker_index"] -= 24
        with self.assertRaisesRegex(ValueError, "NEW_SOURCE_INDICES"):
            self.run_contract()

    def test_previous_batch_speaker_cannot_be_disguised(self):
        self.p["silver_admission"]["source_groups"][0]["source_speaker_id"] = "p3615"
        with self.assertRaisesRegex(ValueError, "EXACT_NEW_SPEAKER_AND_ROLE_MAP"):
            self.run_contract()

    def test_speakers_cannot_move_between_frozen_roles(self):
        groups = self.p["silver_admission"]["source_groups"]
        groups[0]["role"], groups[-1]["role"] = groups[-1]["role"], groups[0]["role"]
        with self.assertRaisesRegex(ValueError, "EXACT_NEW_SPEAKER_AND_ROLE_MAP"):
            self.run_contract()

    def test_every_previous_source_must_remain_excluded(self):
        self.p["silver_admission"]["excluded_source_speaker_ids"].remove("p3615")
        with self.assertRaisesRegex(ValueError, "ALL_HISTORICAL_SOURCE_EXCLUSIONS"):
            self.run_contract()

    def test_old_module_rejects_the_independent_new_campaign(self):
        with self.assertRaisesRegex(ValueError, "MACHINE_RULE_NOT_PINNED"):
            module.build_admission(self.c, self.e, self.root, profile=self.profile, plan=self.p)
        old_rule_plan = copy.deepcopy(self.p)
        old_rule_plan["silver_admission"].update(rule_id=module.RULE_ID, rules_sha256=module.RULES_SHA256)
        with self.assertRaisesRegex(ValueError, "NEW_SOURCE_INDICES"):
            module.build_admission(self.c, self.e, self.root, profile=self.profile, plan=old_rule_plan)


if __name__ == "__main__":
    unittest.main()
