"""No audio, TTS, ASR, IPA or optimizer: English research label-bridge tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "training"))
import english_research_label_bridge as en


def fake_sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def invented_report() -> dict:
    rows, pairs = [], []
    names = en.TRAIN_VOICES + en.DEVELOPMENT_VOICES
    for v in names:
        mapping = {}
        for p, (intent, _) in en.REQUESTS.items():
            ident = "en-" + fake_sha(v + ":" + p)[:24]
            mapping[p] = ident
            rows.append({
                "cell_id":ident,
                "voice_id":v,
                "source_family_id":"qwen3-preset-" + v,
                "role":en.role(v),
                "requested_text":p,
                "requested_intent":intent,
                "wav_sha256":fake_sha("wav:"+v+":"+p),
                "pcm_sha256":fake_sha("pcm:"+v+":"+p),
                "asr_observations":{
                    key:{"text":"invented-"+p,"normalized":"invented",
                         "native_sha256":fake_sha("native:"+key+":"+v+":"+p)}
                    for key in ("qwen","whisper","parakeet")},
                "ipa_tokens":["h","eɪ","x"],
                "human_gold":False,
                "training_admitted":False,
                "trusted_negative":False,
                "phonetic_label_authenticated":False,
                "independent_evaluation":False,
            })
        for word,plain,marked in [
            ("vosen","Hey Vosen","Hey, Vosen!"),
            ("mero","Hey Mero","Hey, Mero!")]:
            pairs.append({
                "voice_id":v,
                "source_family_id":"qwen3-preset-"+v,
                "role":en.role(v),
                "wake_candidate":word,
                "requested_texts":[plain,marked],
                "cell_ids":[mapping[plain],mapping[marked]],
                "machine_only_priority":True,
                "training_admitted":False,
                "human_gold":False,
                "trusted_negative":False,
                "phonetic_label_authenticated":False,
            })
    return {
        "schema":en.SOURCE_SCHEMA,
        "status":"MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD",
        "source_archive_sha256":dict(en.SOURCE_ZIP_SHA256),
        "source_payload_sha256":dict(en.SOURCE_PAYLOAD_SHA256),
        "source_tts_run_id":43242278962,
        "original_three_asr_run_id":43246104418,
        "source_ipa_run_id":43340235981,
        "audio_rows_reused":63,
        "asr_observations_reused":189,
        "new_model_calls":0,
        "voice_count_synthetic":7,
        "review_wake_pairs":14,
        "nearphone_challenges":21,
        "training_admitted":False,
        "human_gold":False,
        "independent_human_validation":"NOT_RUN",
        "rows":rows,
        "wake_pairs":pairs,
    }


class EnglishResearchLabelTests(unittest.TestCase):
    def test_source_roster_and_weak_diagnostic_counts(self):
        src=invented_report()
        self.assertEqual(len(en.verify_three_source_report(src)),63)
        p=en.diagnostic_plan(src)
        self.assertEqual(p["schema"],en.POLICY)
        self.assertEqual(p["locale"],"en-US")
        self.assertEqual(p["class_roster"],{
            "background":0, "Hey Vosen":1, "Hey Mero":2})
        self.assertEqual(p["head_kind"],"whole-recording-3-class-not-ctc")
        self.assertEqual(p["counts"],{
            "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY":30,
            "EXPOSED_DEVELOPMENT_NO_TUNE":12,
            "QUARANTINE_AMBIGUOUS_NEARPHONE":21})
        self.assertFalse(p["training_admitted"])
        self.assertFalse(p["human_gold"])
        self.assertEqual(p["admitted_training_rows"],0)
        self.assertEqual(len(p["rows"]),63)
        self.assertTrue(all(r["ctc_target_ids"] is None and
                            not r["training_admitted"] for r in p["rows"]))
        near=[r for r in p["rows"]
              if r["action"]=="QUARANTINE_AMBIGUOUS_NEARPHONE"]
        self.assertEqual(len(near),21)
        self.assertTrue(all(r["weak_wholeclip_class"] is None for r in near))
        self.assertEqual({r["weak_wholeclip_class"] for r in p["rows"] if
            r["action"]=="TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY"}, {0,1,2})

    def test_all_development_preset_rows_are_untrainable(self):
        out=en.diagnostic_plan(invented_report())["rows"]
        for r in out:
            if r["source_family_id"] in {
                    "qwen3-preset-serena","qwen3-preset-uncle_fu"}:
                self.assertNotEqual(r["action"],
                                    "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY")
        self.assertEqual(sum(r["role"]=="development-only" for r in out),18)

    def test_nearphone_cannot_become_trusted_negative_or_positive(self):
        for field, value in [
            ("trusted_negative",True),
            ("training_admitted",True),
            ("human_gold",True),
            ("phonetic_label_authenticated",True)]:
            sample=invented_report()
            wake=next(r for r in sample["rows"]
                      if r["requested_text"]=="Hey Wosen")
            wake[field]=value
            with self.assertRaisesRegex(
                    ValueError,"ENGLISH_UNAUTHORIZED_MACHINE_LABEL_PROMOTION"):
                en.diagnostic_plan(sample)
        sample=invented_report()
        wake=next(r for r in sample["rows"]
                  if r["requested_text"]=="Hey Rosen")
        wake["requested_intent"]="ordinary_negative"
        with self.assertRaisesRegex(ValueError,"ENGLISH_SOURCE_ROLE_OR_INTENT_DRIFT"):
            en.diagnostic_plan(sample)

    def test_duplicate_pcm_and_wav_and_cell_fail(self):
        for key in ("cell_id","wav_sha256","pcm_sha256"):
            sample=invented_report()
            sample["rows"][1][key]=sample["rows"][0][key]
            with self.assertRaisesRegex(ValueError,"ENGLISH_DUPLICATE_OR_UNBOUND_AUDIO"):
                en.verify_three_source_report(sample)

    def test_source_family_split_and_single_model_lexicon_frozen(self):
        for key,value in [
                ("role","train-candidate"),
                ("source_family_id","qwen3-preset-vivian"),
                ("voice_id","unreviewed-new-voice")]:
            sample=invented_report()
            target=next(r for r in sample["rows"]
                        if r["voice_id"]=="uncle_fu")
            target[key]=value
            with self.assertRaisesRegex(ValueError,"ENGLISH_SOURCE_ROLE_OR_INTENT_DRIFT"):
                en.verify_three_source_report(sample)
        self.assertEqual(en.CLASSES["Hey Vosen"],1)
        self.assertEqual(en.CLASSES["Hey Mero"],2)
        self.assertNotIn("ni3",en.CLASSES)

    def test_pair_roster_wake_text_and_punctuation_cannot_drift(self):
        sample=invented_report()
        sample["wake_pairs"][0]["cell_ids"].reverse()
        with self.assertRaisesRegex(ValueError,"ENGLISH_PAIR_CELL_ID_DRIFT"):
            en.verify_three_source_report(sample)
        sample=invented_report()
        sample["wake_pairs"][0]["training_admitted"]=True
        with self.assertRaisesRegex(
                ValueError,"ENGLISH_UNAUTHORIZED_MACHINE_LABEL_PROMOTION"):
            en.verify_three_source_report(sample)
        sample=invented_report()
        sample["wake_pairs"].pop()
        with self.assertRaisesRegex(ValueError,"ENGLISH_WAKE_REVIEW_PAIR_COUNT"):
            en.verify_three_source_report(sample)

    def test_source_digest_and_asr_authority_fail_closed(self):
        sample=invented_report()
        sample["source_payload_sha256"]["ipa"]="0"*64
        with self.assertRaisesRegex(ValueError,"ENGLISH_REVIEW_SOURCE_POLICY_CHANGED"):
            en.verify_three_source_report(sample)
        sample=invented_report()
        sample["rows"][0]["asr_observations"].pop("parakeet")
        with self.assertRaisesRegex(ValueError,"ENGLISH_ASR_NOT_INDEPENDENT"):
            en.verify_three_source_report(sample)
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/"invented-source.json"
            file.write_text(json.dumps(invented_report()))
            with self.assertRaisesRegex(ValueError,"ENGLISH_REVIEW_SOURCE_HASH_CHANGED"):
                en.read_verified_report(file)

    def test_cli_isolated_without_dependencies_or_model_runs(self):
        path=Path(en.__file__)
        for opt in ([],["-O"],["-OO"]):
            cmd=[sys.executable,"-I","-S","-B",*opt,str(path)]
            r=subprocess.run(cmd,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(r.stdout)["admitted_training_rows"],0)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"invented.json"
            path.write_text(json.dumps(invented_report()))
            out=Path(temp)/"output.json"
            r=subprocess.run([sys.executable,"-I","-S","-B",str(Path(en.__file__)),
                  "--source-review",str(path),"--output",str(out)],
                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
            self.assertEqual(r.returncode,2)
            self.assertIn("ENGLISH_REVIEW_SOURCE_HASH_CHANGED",r.stderr)
            self.assertFalse(out.exists())


if __name__=="__main__":
    unittest.main()
