"""Offline exact source-closure freeze; no model, network or release authority.

The source freeze excludes itself, never follows an arbitrary directory walk,
and never imports a retained host/campaign. It measures the explicitly declared
files only. Execution authority and actual input identities are separate files.
"""
from pathlib import Path
import ast
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from campaign_contract import canonical, digest, identity, read_bytes, require, write_once

NEW = "research/k1-punctuation6-asr18-v1/"
OLD = "research/source-screen32-v1/"
WHISPER = "research/source-screen32-whisper-v1/"
INFERENCE_NEW = (
    "asr18_metadata.py", "campaign_contract.py", "worker_receipts.py", "native_worker.py",
    "cpu_budget.py", "cpu_gate.py", "runtime_scope.py", "setup_adapter.py", "native_locks.py",
    "whisper_binding.py", "optional_tools.py", "execution-release.json",
)
HOST_NEW = (
    "input_binding.py", "raw_data_reader.py", "hosted_run.py", "resource_host.py",
    "evidence_transport.py", "evidence_publisher.py", "prepare_freeze.py", "workflow.yml", "runtime_session.py",
    "source-reuse-lock.json", "input-freeze.json", "tts-source-authority.json", "publication-release.json",
)
# The parent host supplies the reviewed exact verifier roster here. No recursive
# host-tts tree, original TTS outputs or historical ASR inputs are ever inferred.
HOST_TTS_VERIFIER = (
    'host-tts-verifier-source/.github/workflows/k1-punctuation6-tts-v1.yml',
    'host-tts-verifier-source/research/k1-punctuation6-v1/IMPLEMENTATION.md',
    'host-tts-verifier-source/research/k1-punctuation6-v1/README.md',
    'host-tts-verifier-source/research/k1-punctuation6-v1/contract.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/coverage.proposed.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/data_publisher.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/data_transport.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/entry.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/execution-release.disabled.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/model-provenance.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/plan.proposed.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/policy-amendment.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/publication-release.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/source-reuse-lock.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_contract.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_data_publisher.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_tts_finalize.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_tts_host.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_tts_public_schema.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_tts_publish.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_tts_semantics.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/test_tts_worker.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts-release.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts-source-freeze.json',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_common.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_finalize.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_gate.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_host.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_public_schema.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_publish.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_semantics.py',
    'host-tts-verifier-source/research/k1-punctuation6-v1/tts_worker.py',
    'host-tts-verifier-source/research/qwen6_tts/generate_six.py',
    'host-tts-verifier-source/research/qwen6_tts/import_preflight.py',
    'host-tts-verifier-source/research/qwen6_tts/model-lock.json',
    'host-tts-verifier-source/research/qwen6_tts/model-locks.json',
    'host-tts-verifier-source/research/qwen6_tts/runtime-lock.json',
    'host-tts-verifier-source/research/qwen6_tts/runtime_gate.py',
    'host-tts-verifier-source/research/qwen6_tts/setup_diagnostics.py',
    'host-tts-verifier-source/research/qwen6_tts/setup_locked.py',
    'host-tts-verifier-source/research/qwen6_tts/supervision.py',
    'host-tts-verifier-source/research/qwen6_tts/test_supervision.py',
    'host-tts-verifier-source/research/qwen6_tts/wheel_identity.py',
)


def retained_pins(path):
    """Read a literal public source roster without importing its module."""
    tree = ast.parse(read_bytes(path, cap=1024 ** 2).decode("utf-8"))
    candidates = [node.value for node in tree.body if isinstance(node, ast.Assign) and
                  any(isinstance(target, ast.Name) and target.id == "RETAINED_PINS" for target in node.targets)]
    require(len(candidates) == 1, "one literal retained source roster required")
    pins = ast.literal_eval(candidates[0])
    require(type(pins) is dict and bool(pins), "nonempty retained source pins")
    for name, sha in pins.items():
        require(type(name) is str and not name.startswith("/") and
                all(part not in ("", ".", "..") for part in name.split("/")), "safe source member")
        require(type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "source SHA256")
    return pins


def declared_rosters(root=ROOT):
    root = Path(root)
    pins = retained_pins(root / OLD / "asr_worker.py")
    common = {NEW + name for name in INFERENCE_NEW}
    common |= {OLD + name for name in ("asr_worker.py", "runtime_scope.py", "sense_adapter.py", "native_v2_adapter.py")}
    common |= {"research/qwen6_asr/" + name for name in pins}
    common |= {WHISPER + name for name in ("adapter.py", "common.py", "runtime-lock.json", "source-lock.json", "setup_runtime.py")}
    common |= {"tools/" + name for name in ("asr_native_contracts.py", "speech_asr_consensus.py", "speech_label_admission.py")}
    setup = common | {WHISPER + "attempt-3/" + name for name in
        ("bounded_host.py", "bounded_connect.py", "pip_acquire.py", "pip_compatibility.py", "phased_setup.py")}
    host = setup | {NEW + name for name in HOST_NEW} | set(HOST_TTS_VERIFIER)
    # Original host/contract are observed public-resource helper sources only;
    # no original execute/run/collect/cohort logic belongs in the inference tree.
    host |= {OLD + name for name in ("container-lock.json", "hosted_run.py", "contract.py")}
    forbidden = ("research/qwen12-native-admission-v1/", "/evidence/", "/output/", "/results/")
    for names in (common, setup, host):
        require(NEW + "execution-freeze.json" not in names, "self-excluding source freeze")
        require(not any(any(value in name for value in forbidden) for name in names), "old campaign/result cannot be staged")
        require(not any(name.endswith((".wav", ".pt", ".safetensors", ".bin", ".pyc")) for name in names),
                "source-only closure, no audio/model/binary/cache")
    require(not any(name.startswith("host-tts-verifier-source/") for name in common | setup), "host-only TTS verifier")
    return {"inference_code": common, "setup_code": setup, "host_code": host}


def build(root=ROOT):
    root = Path(root)
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "unaliasable source root")
    rosters = declared_rosters(root)
    pins = retained_pins(root / OLD / "asr_worker.py")
    for name, sha in pins.items():
        require(identity(root / "research/qwen6_asr" / name)["sha256"] == sha, "unchanged retained source pin required")
    result = {"schema": "k1-asr18-source-freeze-v1"}
    for role, names in rosters.items():
        result[role] = {name: identity(root / name) for name in sorted(names)}
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(argv == [], "fixed source/output paths only")
    return write_once(HERE / "execution-freeze.json", build())


if __name__ == "__main__":
    main()
