"""Tiny invented container fixtures only; no model, recognizer or real speech."""
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import wave
import zipfile

from asr_consensus_fixtures import consensus_fixture, rebind_fixture
from speech_label_admission import canonical_sha256
import asr_source_resolver as resolver


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class TrustedFixture:
    """The test installs its root with monkeypatch; production has no test switch."""
    def __init__(self, root):
        self.root = Path(root)
        self.store = self.root / "store"
        self.store.mkdir()
        self.documents = {}
        self.extra_members = {}
        self.native_captures = {}
        audio = io.BytesIO()
        with wave.open(audio, "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
            out.writeframes(b"\xe8\x03" * 160)
        self.audio = audio.getvalue()
        self.source = self.root / "invented-original.wav"
        self.source.write_bytes(self.audio)
        self.record = consensus_fixture({"file_sha256": sha(self.audio),
                                         "pcm_sha256": sha(b"\xe8\x03" * 160), "frames": 160})
        self.record["authentication_verification"] = None
        record = self.record
        rights = {"schema": "retained-asr-source-rights-v1", "training_use_allowed": True,
                  "license_id": record["source_provenance"]["license_id"],
                  "source_id": record["source_id"], "source_family_id": record["source_family_id"]}
        self.documents["rights.json"] = rights
        record["source_provenance"]["rights_evidence_sha256"] = sha(encoded(rights))
        source_policy = {"schema": "retained-asr-source-policy-v1", "allowed_roles": ["train"],
                         "human_gold": False, "source_provenance": copy.deepcopy(record["source_provenance"]),
                         "human_review_history": [], "human_review_history_sha256": canonical_sha256([])}
        for key in ("source_id", "source_family_id", "intended_text", "file_sha256", "pcm_sha256", "audio_frames"):
            source_policy[key] = record[key]
        self.documents["source-policy.json"] = source_policy
        bindings = {"root_id": "invented-trusted-fixture", "source_policy": self.ref("source-policy.json"),
                    "rights": self.ref("rights.json"), "source_audio": self.ref("source.wav"), "recognizers": []}
        record["evidence_bindings"] = bindings
        for index, identity in enumerate(record["recognizer_policy"]["recognizers"]):
            prefix = f"asr-{index}"
            model = {"schema": "retained-asr-model-lock-v1",
                     **{key: identity[key] for key in ("provider_id", "model_id", "model_revision", "model_family")},
                     "assets": [{"path": "weights.bin", "bytes": index + 1,
                                 "sha256": sha(f"invented-weight-identity-{index}".encode())}]}
            model["model_sha256"] = identity["model_sha256"] = canonical_sha256(model["assets"])
            code = {"schema": "retained-asr-source-lock-v1", "executed_head_sha": "a" * 40, "files": [
                {"path": f"{prefix}/capture.py", "bytes": 7, "sha256": sha(b"# stub\n")}]}
            native = {"schema": "retained-asr-native-six-field-config-v1", "adapter_id": "explicit-six-field-v1",
                      "parameters": copy.deepcopy(identity["decode_config"])}
            identity["native_decode_config_sha256"] = sha(encoded(native))
            self.documents[f"{prefix}/model-lock.json"] = model
            self.documents[f"{prefix}/source-lock.json"] = code
            self.documents[f"{prefix}/native-config.json"] = native
            bindings["recognizers"].append({"recognizer_id": identity["recognizer_id"],
                **{key: self.ref(f"{prefix}/{name}") for key, name in (
                    ("model_lock", "model-lock.json"), ("source_lock", "source-lock.json"),
                    ("native_decode_config", "native-config.json"), ("started", "started.json"),
                    ("receipt", "receipt.json"), ("raw_output", "raw-output.json"))}})
        self.rebind_captures()
        self.rebuild()

    def rebind_captures(self):
        """Bind invented starts/completions after changing invented identities."""
        record = self.record
        record["recognizer_policy_sha256"] = canonical_sha256(record["recognizer_policy"])
        for index, (identity, observation) in enumerate(zip(record["recognizer_policy"]["recognizers"], record["observations"])):
            prefix = f"asr-{index}"
            observation["recognizer_identity_sha256"] = canonical_sha256(identity)
            observation["recognizer_policy_sha256"] = record["recognizer_policy_sha256"]
            output = {"schema": "retained-asr-raw-transcript-v1", "request_id": observation["request_id"],
                      "actual_text": observation["actual_text"], "output_scope": "complete-decoder-transcript", "truncated": False}
            self.documents[f"{prefix}/raw-output.json"] = output
            observation["raw_output_sha256"] = sha(encoded(output))
            common = {key: observation[key] for key in (
                "source_id", "file_sha256", "pcm_sha256", "audio_frames", "request_id", "recognizer_id",
                "recognizer_identity_sha256", "recognizer_policy_sha256")}
            common.update(model_lock_sha256=sha(encoded(self.documents[f"{prefix}/model-lock.json"])),
                          source_lock_sha256=sha(encoded(self.documents[f"{prefix}/source-lock.json"])),
                          native_decode_config_sha256=identity["native_decode_config_sha256"])
            started = {"schema": "retained-asr-started-v1", **common}
            self.documents[f"{prefix}/started.json"] = started
            self.documents[f"{prefix}/receipt.json"] = {"schema": "retained-asr-completed-v1", **common,
                "started_sha256": sha(encoded(started)), "raw_output_sha256": sha(encoded(output)),
                "status": "success", "decoder_completed": True, "processed_start_frame": 0, "processed_end_frame": 160,
                "loaded_model_assets": copy.deepcopy(self.documents[f"{prefix}/model-lock.json"]["assets"]),
                "executed_source_files": copy.deepcopy(self.documents[f"{prefix}/source-lock.json"]["files"])}


            if index in self.native_captures:
                self.documents[f"{prefix}/receipt.json"]["native_capture"] = copy.deepcopy(self.native_captures[index])

    def ref(self, member):
        return {"archive": "capture", "member": member}

    def rebuild(self):
        """Repin only the test-controlled root after invented-evidence mutations."""
        members = {name: encoded(value) for name, value in self.documents.items()}
        members.update(self.extra_members)
        members["source.wav"] = self.audio
        for index in range(3): members[f"asr-{index}/capture.py"] = b"# stub\n"
        container = io.BytesIO()
        with zipfile.ZipFile(container, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, raw in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)); info.external_attr = 0o100644 << 16
                archive.writestr(info, raw)
        raw = container.getvalue(); self.archive = self.store / f"{sha(raw)}.zip"; self.archive.write_bytes(raw)
        self.provenance = {"schema": "invented-retained-run-provenance-v1", "repository": "jiying2007/kws-pipeline",
            "executed_head_sha": "a" * 40, "run_id": 1,
            "archives": {"capture": {"sha256": sha(raw), "bytes": len(raw), "exact_members": {
                name: {"sha256": sha(data), "bytes": len(data)} for name, data in members.items()}}}}
        data = encoded(self.provenance); self.provenance_path = self.root / "provenance.json"; self.provenance_path.write_bytes(data)
        self.trust_root = resolver._SourceRoot("invented-trusted-fixture", "b" * 40, "provenance.json", sha(data),
                                               "train", resolver.CAPTURE_CONTRACT)
        return self.trust_root

    def use_native_v2(self):
        from asr_native_fixtures import native_fixture, rebind_native_fixture
        from asr_native_contracts import QWEN_ADAPTER, SENSE_ADAPTER, WHISPER_ADAPTER
        from speech_asr_consensus import CONSENSUS_CLASS_V2, POLICY_CLASS_V2, RESULT_CLASS_V2
        record = self.record
        record.update(schema_version=2, evidence_class=CONSENSUS_CLASS_V2,
            acoustic_verbatim_completeness="unknown", training_use_prohibited=False,
            label_selection_policy="trusted-three-asr-preferred-training-v2", human_observation_history=[])
        record["recognizer_policy"].update(schema_version=2, evidence_class=POLICY_CLASS_V2)
        policy = self.documents["source-policy.json"]
        policy.update(training_use_prohibited=False, label_selection_policy=record["label_selection_policy"],
                      human_observation_history=[], human_observation_history_sha256=canonical_sha256([]))
        for index, adapter in enumerate((QWEN_ADAPTER, SENSE_ADAPTER, WHISPER_ADAPTER)):
            prefix = f"asr-{index}"
            identity = record["recognizer_policy"]["recognizers"][index]
            identity.pop("decode_config"); identity.pop("decode_config_sha256")
            config, capture = native_fixture(adapter)
            identity["model_family"] = config["model_family"]
            model = self.documents[f"{prefix}/model-lock.json"]
            model["model_family"] = config["model_family"]
            effective = config["effective_config"]
            dependencies = {}
            for key in list(effective):
                if not key.endswith("_sha256"):
                    continue
                member = f"{prefix}/{key}.json"
                if key == "resolved_generation_config_sha256":
                    value = {field: copy.deepcopy(effective[field]) for field in (
                        "num_beams", "num_return_sequences", "forced_bos_token_id", "forced_eos_token_id",
                        "suppress_tokens", "begin_suppress_tokens", "bad_words_ids", "force_words_ids",
                        "sequence_bias", "stop_strings")}
                    value.update({field: config["native_request"][field] for field in (
                        "do_sample", "max_new_tokens", "eos_token_id")})
                else:
                    value = {"invented_dependency": key, "adapter": adapter}
                body = encoded(value); self.extra_members[member] = body
                effective[key] = sha(body); dependencies[key] = self.ref(member)
                if key != "resolved_generation_config_sha256":
                    model["assets"].append({"path": member, "bytes": len(body), "sha256": sha(body)})
            model["model_sha256"] = identity["model_sha256"] = canonical_sha256(model["assets"])
            config["source_lock_sha256"] = sha(encoded(self.documents[f"{prefix}/source-lock.json"]))
            config["asset_lock_sha256"] = sha(encoded(model))
            capture["source_lock_sha256"] = config["source_lock_sha256"]
            capture["asset_lock_sha256"] = config["asset_lock_sha256"]
            rebind_native_fixture(config, capture)
            identity["native_decode_config"] = config
            identity["native_decode_config_sha256"] = canonical_sha256(config)
            self.documents[f"{prefix}/native-config.json"] = config
            self.native_captures[index] = capture
            record["evidence_bindings"]["recognizers"][index]["native_dependencies"] = dependencies
            observation = record["observations"][index]
            for field in ("transcript_complete", "coverage_complete", "speech_present"):
                observation.pop(field)
            observation.update(schema_version=2, evidence_class=RESULT_CLASS_V2,
                               input_coverage_complete=True, decoder_output_complete=True)
        self.rebind_captures()
        return self.rebuild()

    def add_raw_observation(self, text=None, certainty="unknown", original=b"invented unstructured note", **extra):
        history = self.record["human_observation_history"]
        revision = len(history) + 1
        raw_member = f"private-note-{revision}.txt"
        self.extra_members[raw_member] = original
        observation = {"schema_version": 1, "evidence_class": "speech-raw-human-observation-v1",
            "observation_revision": revision,
            "supersedes_observation_sha256": history[-1]["observation_sha256"] if history else None,
            **{field: self.record[field] for field in ("source_id", "file_sha256", "pcm_sha256")},
            "original_observation_artifact": self.ref(raw_member), "original_observation_sha256": sha(original),
            "observed_text": text, "certainty": certainty, **extra}
        member = f"human-observation-{revision}.json"
        self.documents[member] = observation
        reference = {"schema_version": 1, "evidence_class": "speech-raw-human-observation-reference-v1",
            "observation_revision": revision, "supersedes_observation_sha256": observation["supersedes_observation_sha256"],
            "observation_sha256": canonical_sha256(observation), "artifact": self.ref(member)}
        history.append(reference)
        policy = self.documents["source-policy.json"]
        policy["human_observation_history"] = copy.deepcopy(history)
        policy["human_observation_history_sha256"] = canonical_sha256(history)
        from speech_asr_consensus import raw_human_conflicts, human_conflicts
        wrappers = [self.documents[f"human-observation-{i}.json"] for i in range(1, revision + 1)]
        typed_reviews, typed_conflicts = [], {}
        for wrapper in wrappers:
            if wrapper.get("original_format") == "human-review-v2":
                typed_reviews.append(json.loads(self.extra_members[wrapper["original_observation_artifact"]["member"]]))
                typed_conflicts[wrapper["observation_revision"]] = human_conflicts(typed_reviews, self.record)[0]
        current, historical = raw_human_conflicts(wrappers, self.record, typed_conflicts=typed_conflicts)
        self.record.update(human_conflict=current, historical_human_conflict=historical)
        return self.rebuild()

    def admission(self):
        from speech_asr_consensus import admission_from_consensus
        context = resolver.resolve_source_context(self.record, evidence_dir=self.store)
        return admission_from_consensus(self.record, "f" * 64, verification_context=context)


@contextmanager
def trusted_fixture():
    with tempfile.TemporaryDirectory(prefix="invented-asr-source-fixture-") as directory:
        fixture = TrustedFixture(directory)
        with patch.object(resolver, "_REPO_ROOT", fixture.root), patch.object(resolver, "_SOURCE_ROOTS", (fixture.trust_root,)):
            yield fixture
