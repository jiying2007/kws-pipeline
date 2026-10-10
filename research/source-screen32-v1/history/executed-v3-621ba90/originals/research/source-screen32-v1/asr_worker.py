#!/usr/bin/env python3
"""One blind recognizer process for 0..16 generated WAVs; no producer imports.

The two scientific loader bodies below are copied exactly from the pinned ASR
source and AST-checked by tests. Only _preload is new: it binds the smaller
interface to a variable audio-only subset, rather than relaxing old six guards.
The retained inference functions and PCM/asset/runtime guards are unchanged.
The container launcher owns hard memory/network/filesystem scope and outer wall
termination. This worker verifies that live scope before imports and each call.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys
import time

HERE = Path(__file__).resolve().parent
RETAINED = HERE.parent / "qwen6_asr"
MAX_FILES = 16
MAX_FRAMES = 192000
MAX_INPUT_BYTES = 8 * 1024**2
MAX_JOB_BYTES = 65536
SHA = re.compile(r"[0-9a-f]{64}\Z")
MODEL_IDS = {"qwen06": "Qwen/Qwen3-ASR-0.6B", "sensevoice": "FunAudioLLM/SenseVoiceSmall"}
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "ORT_DISABLE_TELEMETRY": "1", "CUDA_VISIBLE_DEVICES": ""}
CACHE_KEYS = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
              "NUMBA_CACHE_DIR", "TORCH_HOME", "MPLCONFIGDIR", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR")
RETAINED_PINS = {
    "FUNASR_MODEL_LICENSE": "7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8",
    "NOTICE.md": "a01fd4891fd878f2bbb4de93e13dbef67dd10b127974240046304803c96685a7",
    "core/asr6_contract.py": "d1c3368133e06a3de9758b23f136c3a37a74a63057f317ece863470ad0304b10",
    "core/asr_stage/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "core/asr_stage/adapters.py": "85512f20280cfa0aeb06b50699664e360db39bbf5b58034d3a2ede5cde8ff6ca",
    "core/asr_stage/architecture.py": "d3f0d67bcf724c34763ac4666a1570b02dfd979182e6183403ef91e0897d58ab",
    "core/asr_stage/assets.py": "91a8c7e75f37872fc1b07cabc2afd1790901bf44a3fda1e48fb5abc2b129ff6e",
    "core/asr_stage/decoding.py": "fd401bd85e4ca3d871e5a9354753d3599facbb32b9df22e34f05e5476e74b503",
    "core/asr_stage/execution.py": "149befff56ef9da603d8b27174ca1a1d4eae638cc86f356cc5a1b8412ac06c9f",
    "core/asr_stage/runtime_support.py": "f3a6ee85e30dc5a1bc2351a76843c44bc988aed83745773e543e1b69cf4a6d90",
    "core/pcm/pcm_binding.py": "1f8cb1e8fba2c99dfd969b7f33a12011243aa68f0a6e1eb0dbf45d90fc9b1684",
    "core/runtime_version_gate.py": "c6d2e35c69f3ba34e3a7f92bd9b486eb00476f76ed5c8687319f4382b23bd512",
    "core/sense_binding.py": "1a2fd8474d347591939b3ce9ab7411aa9a8f9ce0ffa732d93a95402b677503d2",
    "core/vendor/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "core/vendor/checkpoint_guard/__init__.py": "ca1bfd88d26591efeb70bb58155294be4ebc2af7a6e45cc5f09f3e90dc5c541e",
    "core/vendor/checkpoint_guard/checkpoint_contract.py": "80cd48b0ce8beb2f7473372a3d904ba42af316c7df4a3c44f8a8b54dd50896c4",
    "core/vendor/checkpoint_guard/exact_apply.py": "10bc8634053a602b1669a007865c9d18955577b8aab4b7e7f14dfdcc6ce9e22b",
    "core/vendor/qwen_loading_gate_v2.py": "1b0cbd7b96e24925bfcf72438edea9dfb286860d2a990cdbad48104ae595d3fb",
    "import_preflight.py": "30cfe60347ccc6074d98a21406fccee3e3be5c54822aa8d874aae3655693ae1c",
    "model-locks.json": "8b0813aba3c2447c7dc30b47e856e4ce9a619a98d32ed6dfa9f44dc6646714e4",
    "runtime-lock.json": "2d496a76bd0e044a6812bfc9afdd8965474bf135bd21f3a24f71695638d1183f",
    "sense_parser.py": "878278e46ae9af349a93e37dddffa83e916f447ef82fb6dd61fa8948bb8261d7",
    "setup_diagnostics.py": "1b98e253d7054d0bfcd5ec128cb3a331bb05de48776330e84ea348694b3a1bf2",
    "setup_locked.py": "d7c580b59b0bd249bc62c9093b9de5f8842170510cf11e7e61c29057e6287af4",
    "wheel_identity.py": "623ee07a0d8f85ba1d7218a56fcd0e270df8c6e2c3e353dd52b3184ca915d8d5"
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(raw, maximum=MAX_JOB_BYTES):
    require(type(raw) is bytes and len(raw) <= maximum, "JSON size/type")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def invalid(_):
        raise ValueError("Nonfinite JSON")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def read_regular(path, maximum):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
            "Unaliased bounded regular file required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    after = path.lstat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink, s.st_mode)
    require(stamp(before) == stamp(after) and len(raw) == before.st_size, "File changed during read")
    return raw


def verify_sources():
    for name, sha in RETAINED_PINS.items():
        require(digest(read_regular(RETAINED / name, 1024**2)) == sha, "Retained ASR source drift: " + name)
    return dict(RETAINED_PINS)


def verify_scope():
    # -I omits the script directory; import only the separately frozen sibling.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from runtime_scope import verify_runtime_scope
    return verify_runtime_scope()


def prepare_caches():
    """Fixed writable scratch paths, only after the live kernel scope succeeds."""
    verify_scope()
    scratch = Path("/scratch")
    require(scratch.is_dir() and not scratch.is_symlink(), "Expected scratch mount")
    for key in CACHE_KEYS:
        path = scratch / key.lower()
        path.mkdir(exist_ok=True)
        require(path.is_dir() and not path.is_symlink(), "Unaliased scratch cache directory")
        os.environ[key] = str(path)
    home = scratch / "home"
    home.mkdir(exist_ok=True)
    require(home.is_dir() and not home.is_symlink(), "Unaliased scratch home")
    os.environ["HOME"] = str(home)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def import_helpers():
    # This function is called only after the live hard-scope and source checks.
    sys.path.insert(0, str(RETAINED))
    sys.path.insert(0, str(RETAINED / "core"))
    from asr_stage import adapters
    from asr_stage import execution
    from pcm.pcm_binding import WaveExpectation, bind_wave, validate_pcm_descriptor
    # Explicit aliases in this NEW module only; retained module globals and
    # its exact-six validators are never changed or replaced.
    names = ("QWEN", "SENSE", "PreparedRuntime", "canonical_sha", "qwen06_schema",
             "sensevoice_schema", "verify_assets", "require_runtime_versions",
             "verify_source_members", "validate_asset_lock", "read_small_locked",
             "validate_qwen_geometry", "validate_sense_geometry", "qgate",
             "claim_fresh_model_process", "describe_cpu_tensor", "describe_storage",
             "assert_effective_qwen_attention", "verify_tokenizer", "exact_apply",
             "SENSEVOICE_METADATA_RECIPE")
    globals().update({name: getattr(adapters, name) for name in names})
    globals().update(WaveExpectation=WaveExpectation, bind_wave=bind_wave,
                     validate_pcm_descriptor=validate_pcm_descriptor)
    return adapters, execution


def validate_job(raw):
    job = decode(raw)
    require(type(job) is dict and set(job) == {"schema", "clips"}
            and job["schema"] == "blind-source-screen32-job-v1", "Blind job schema")
    rows = job["clips"]
    require(type(rows) is list and 0 <= len(rows) <= MAX_FILES, "Bounded blind subset")
    hashes = []
    for index, row in enumerate(rows, 1):
        require(type(row) is dict and set(row) == {"audio_id", "audio_path", "wav_sha256"}, "Audio-only fields")
        sha = row["wav_sha256"]
        require(type(sha) is str and SHA.fullmatch(sha), "WAV hash")
        require(row["audio_id"] == f"clip-{index:06d}" and row["audio_path"] == "audio/" + sha + ".wav", "Opaque ID/path")
        hashes.append(sha)
    require(hashes == sorted(set(hashes)), "Unique content-hash order")
    return job


def bind_inputs(root):
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), "Input directory")
    require({p.name for p in root.iterdir()} == {"job.json", "audio"}, "Only blind job and audio accepted")
    audio = root / "audio"
    require(not audio.is_symlink() and audio.is_dir(), "Audio directory")
    raw = read_regular(root / "job.json", MAX_JOB_BYTES)
    job = validate_job(raw)
    expected = {row["wav_sha256"] + ".wav" for row in job["clips"]}
    require({p.name for p in audio.iterdir()} == expected, "Extra or missing audio")
    clips = []
    total = len(raw)
    for row in job["clips"]:
        path = root / row["audio_path"]
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 46 <= info.st_size <= 44 + 2 * MAX_FRAMES,
                "WAV type/size")
        total += info.st_size
        require(total <= MAX_INPUT_BYTES, "Aggregate blind input cap")
        bound = bind_wave(str(audio.absolute()), WaveExpectation(row["audio_id"], path.name,
                          row["wav_sha256"], info.st_size, (info.st_size - 44) // 2))
        clips.append({"opaque_id": bound.opaque_id, "binding_sha256": bound.descriptor_sha256, "descriptor": bound.descriptor})
    decoder = {"schema": "bounded-source-screen-asr-decoder-v1", "scope": "audio_only_no_plan_context",
               "source_manifest_sha256": digest(raw), "order": [row["audio_id"] for row in job["clips"]], "clips": clips}
    decoder_raw = canonical(decoder)
    validate_decoder(decoder_raw, digest(decoder_raw))
    return job, raw, decoder_raw


def validate_decoder(raw, expected_sha):
    require(type(expected_sha) is str and SHA.fullmatch(expected_sha) and digest(raw) == expected_sha, "Decoder hash")
    value = decode(raw, 1024**2)
    require(type(value) is dict and set(value) == {"schema", "scope", "source_manifest_sha256", "order", "clips"}, "Decoder fields")
    require(value["schema"] == "bounded-source-screen-asr-decoder-v1" and value["scope"] == "audio_only_no_plan_context", "Decoder scope")
    require(type(value["source_manifest_sha256"]) is str and SHA.fullmatch(value["source_manifest_sha256"]), "Job identity")
    require(type(value["clips"]) is list and len(value["clips"]) <= MAX_FILES, "Decoder subset")
    require(value["order"] == [f"clip-{i:06d}" for i in range(1, len(value["clips"]) + 1)], "Decoder IDs")
    index, hashes = {}, []
    for oid, row in zip(value["order"], value["clips"]):
        require(type(row) is dict and set(row) == {"opaque_id", "binding_sha256", "descriptor"} and row["opaque_id"] == oid, "Decoder row")
        descriptor = row["descriptor"]
        validate_pcm_descriptor(descriptor)
        require(descriptor["opaque_id"] == oid and digest(canonical(descriptor)) == row["binding_sha256"], "PCM descriptor binding")
        require(1 <= descriptor["raw_pcm16"]["frame_count"] <= MAX_FRAMES, "PCM duration")
        hashes.append(descriptor["wav"]["sha256"])
        index[oid] = row
    require(hashes == sorted(set(hashes)), "Decoder hash order")
    return index


def execution_contract(model, model_lock, decoder_raw):
    decoder = decode(decoder_raw, 1024**2)
    return {"schema": "bounded-source-screen-asr-execution-v1", "model_id": MODEL_IDS[model],
            "run_id": "bounded-source-screen-" + model + "-" + decoder["source_manifest_sha256"],
            "asset_lock_sha256": model_lock["asset_lock_sha256"], "source_lock_sha256": model_lock["source_lock_sha256"],
            "expected_schema_sha256": model_lock["expected_schema_sha256"], "decoder_manifest_sha256": digest(decoder_raw),
            "source_manifest_sha256": decoder["source_manifest_sha256"], "candidate_count": len(decoder["clips"]),
            "model_dtype": "float32", "batch_size": 1, "context": "", "hotwords": [], "language": "auto",
            "scope": "single_decode_each_blind_audio_no_plan_context", "human_gold": False, "training_admitted": False}


def _preload(model_id, root, assets, asset_sha256, sources, source_sha256,
             declaration, declaration_sha256, decoder_manifest_bytes):
    verify_scope()
    model = next(name for name, value in MODEL_IDS.items() if value == model_id)
    locks = decode(read_regular(RETAINED / "model-locks.json", 65536))
    expected = execution_contract(model, locks[model], decoder_manifest_bytes)
    require(canonical(declaration) == canonical(expected) and digest(canonical(declaration)) == declaration_sha256, "Bounded execution contract")
    require(type(declaration["candidate_count"]) is int and 1 <= declaration["candidate_count"] <= MAX_FILES, "Nonempty bounded model call budget")
    require(asset_sha256 == expected["asset_lock_sha256"] and source_sha256 == expected["source_lock_sha256"], "Locked model association")
    inputs = validate_decoder(decoder_manifest_bytes, expected["decoder_manifest_sha256"])
    schema = qwen06_schema() if model_id == QWEN else sensevoice_schema()
    schema_sha = canonical_sha(schema)
    require(schema_sha == expected["expected_schema_sha256"], "Independent schema identity")
    asset_receipt = verify_assets(root, assets, asset_sha256, model_id)
    versions = require_runtime_versions()
    verify_source_members(sources, source_sha256, model_id)
    return schema, schema_sha, declaration, asset_receipt, versions, inputs


# These scientific loader bodies MUST remain AST-identical to the retained source.
def load_qwen_after_approval(root,assets,asset_sha256,sources,source_sha256,
                             declaration,declaration_sha256,decoder_manifest_bytes):
    schema,schema_sha,dec,asset_receipt,versions,inputs = _preload(QWEN,root,assets,asset_sha256,
        sources,source_sha256,declaration,declaration_sha256,decoder_manifest_bytes)
    files = validate_asset_lock(assets,asset_sha256,QWEN)
    config_data = json.loads(read_small_locked(root,"config.json",files))
    validate_qwen_geometry(config_data)
    evidence = qgate.inspect_safetensors(Path(root)/"model.safetensors",schema,schema_sha,
        files["model.safetensors"]["sha256"],approved_aliases=[])
    claim_fresh_model_process()
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoConfig,AutoModel,AutoProcessor
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = AutoConfig.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
    # Explicit nested propagation. The top-level hint alone is not sufficient.
    for cfg in (config,config.thinker_config,config.thinker_config.audio_config,config.thinker_config.text_config):
        cfg._attn_implementation = "eager"
        # Nested _from_config calls consult their own dtype. The official
        # thinker BF16 hint otherwise overrides a top-level FP32 argument.
        cfg.dtype = torch.float32
    result = AutoModel.from_pretrained(str(root),config=config,dtype=torch.float32,
        device_map="cpu",attn_implementation="eager",local_files_only=True,
        trust_remote_code=False,output_loading_info=True,ignore_mismatched_sizes=False,
        use_safetensors=True,low_cpu_mem_usage=True)
    if type(result) is not tuple or len(result)!=2: raise RuntimeError("Qwen loader must return model/loading_info")
    model,loading_info = result
    qgate.clear_verification(model)
    qgate.validate_loading_info(loading_info)
    model.eval()
    attention = assert_effective_qwen_attention(model)
    callback = lambda value: describe_cpu_tensor(value,True)
    load_receipt = qgate.verify_loaded_model(model,loading_info,evidence,callback,describe_storage)
    processor = AutoProcessor.from_pretrained(str(root),fix_mistral_regex=True,
        local_files_only=True,trust_remote_code=False)
    wrapper = Qwen3ASRModel(backend="transformers",model=model,processor=processor,
        forced_aligner=None,max_inference_batch_size=1,max_new_tokens=256)
    receipt = {"assets":asset_receipt,"versions":versions,"load":load_receipt,
               "effective_attention":attention,"dtype":"float32","device":"cpu"}
    return PreparedRuntime(QWEN,wrapper,receipt,dec,set(),inputs)


def load_sense_after_approval(root,assets,asset_sha256,sources,source_sha256,
                              declaration,declaration_sha256,decoder_manifest_bytes):
    schema,schema_sha,dec,asset_receipt,versions,inputs = _preload(SENSE,root,assets,asset_sha256,
        sources,source_sha256,declaration,declaration_sha256,decoder_manifest_bytes)
    files = validate_asset_lock(assets,asset_sha256,SENSE)
    claim_fresh_model_process()
    import torch
    import yaml
    from funasr import AutoModel
    from funasr.utils import fbank
    if fbank._HAS_TORCHAUDIO or fbank._knf is None:
        raise RuntimeError("Exact native-fbank backend is required; no backend fallback")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = yaml.safe_load(read_small_locked(root,"config.yaml",files))
    validate_sense_geometry(config)
    names = ("model","model_conf","encoder","encoder_conf","tokenizer","tokenizer_conf",
             "frontend","frontend_conf","specaug","specaug_conf")
    kwargs = {key:config[key] for key in names if key in config}
    if config.get("tokenizer")!="SentencepiecesTokenizer" or config.get("frontend")!="WavFrontend":
        raise RuntimeError("Unreviewed SenseVoice components")
    kwargs["tokenizer_conf"] = dict(kwargs["tokenizer_conf"],bpemodel=str(Path(root)/"chn_jpn_yue_eng_ko_spectok.bpe.model"))
    kwargs["frontend_conf"] = dict(kwargs["frontend_conf"],cmvn_file=str(Path(root)/"am.mvn"),
                                   dither=1.0,snip_edges=True,upsacle_samples=True)
    kwargs.update(model_path=str(root),init_param=None,device="cpu",ncpu=1,batch_size=1,seed=0,
        disable_update=True,disable_pbar=True,disable_log=True,trust_remote_code=False,fp16=False,bf16=False)
    wrapper = AutoModel(**kwargs)
    if wrapper.kwargs["tokenizer"].get_vocab_size()!=25055:
        raise RuntimeError("Locked tokenizer vocabulary differs from independent schema")
    tokenizer_identity = verify_tokenizer(wrapper.kwargs["tokenizer"].sp,
        Path(root)/"chn_jpn_yue_eng_ko_spectok.bpe.model")
    checkpoint = torch.load(Path(root)/"model.pt",map_location="cpu",weights_only=True)
    loaded = exact_apply(wrapper.model,checkpoint,describe_cpu_tensor,
                         expected_schema=schema,expected_schema_sha256=schema_sha,
                         metadata_recipe=SENSEVOICE_METADATA_RECIPE)
    del checkpoint
    wrapper.model.eval()
    receipt = {"assets":asset_receipt,"versions":versions,"load":loaded,
        "dtype":"float32","device":"cpu","frontend_backend":"kaldi-native-fbank",
        "dither":1.0,"dither_rng_reproducibility":"not established by torch/numpy seed",
        "no_vad":True,"use_itn":False,"tokenizer_identity":tokenizer_identity}
    return PreparedRuntime(SENSE,wrapper,receipt,dec,set(),inputs)



class ClipDeadline(TimeoutError):
    pass


def alarm(_signal, _frame):
    raise ClipDeadline("Per-clip deadline")


def progress(output_root, stage, index):
    """Atomic fixed-shape heartbeat for the host's whole-container watchdog."""
    require(stage in ("load", "cell") and type(index) is int
            and ((stage == "load" and index == 0) or (stage == "cell" and 1 <= index <= MAX_FILES)), "Progress state")
    root = Path(output_root)
    temporary = root / (".progress-" + str(index) + ".tmp")
    raw = canonical({"stage": stage, "index": index, "started_monotonic": time.monotonic()})
    with temporary.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(root / "progress.json")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(model, input_root, output_root, runtime_root):
    require(model in MODEL_IDS, "Model choice")
    scope = verify_scope()
    os.environ.update(THREAD_ENV)
    os.environ.update(OFFLINE_ENV)
    prepare_caches()
    pins = verify_sources()
    helpers, evidence = import_helpers()
    job, job_raw, decoder_raw = bind_inputs(input_root)
    output_root = Path(output_root)
    require(not output_root.is_symlink() and output_root.is_dir(), "Output directory")
    require(not any(output_root.iterdir()), "Each recognizer requires its own empty output directory")
    directory = output_root / model
    directory.mkdir(exist_ok=False)  # Durable single-model claim, including failed loads.
    retain = lambda name, value: evidence.retain(directory, name, value)
    models = decode(read_regular(RETAINED / "model-locks.json", 65536))
    lock = models[model]
    model_identity = {"model_id": MODEL_IDS[model], "revision": lock["asset_lock"]["revision"]}
    declaration = execution_contract(model, lock, decoder_raw)
    contract_sha = retain("contract.json", declaration)
    retain("blind-job.json", job)
    retain("decoder-inputs.json", decode(decoder_raw, 1024**2))
    retain("source-identity.json", {"files": pins, "scientific_loader_bodies": "AST-identical; bounded _preload only"})
    retain("scope.json", scope)
    outcomes = [{"opaque_id": row["audio_id"], "wav_sha256": row["wav_sha256"], "status": "not_run", "raw_text": None,
                 "completeness": "unknown", "quality_flags": [], "execution_receipt_sha256": None} for row in job["clips"]]
    retain("outcomes.initial.json", outcomes)
    attempted, failure, started = 0, None, time.monotonic()
    try:
        if not outcomes:
            retain("not-run.json", {"status": "empty_generated_subset", "model_constructions": 0, "decode_attempts": 0})
        else:
            progress(output_root, "load", 0)
            verify_scope()
            environment = evidence.require_offline_flags()
            environment["thread_environment"] = dict(THREAD_ENV)
            retain("environment.json", environment)
            from setup_adapter import verify_runtime
            verify_runtime(runtime_root, "asr")
            loader = load_qwen_after_approval if model == "qwen06" else load_sense_after_approval
            infer = helpers.infer_qwen_once_after_approval if model == "qwen06" else helpers.infer_sense_once_after_approval
            retain("model-load-started.json", {"model": model_identity, "execution_contract_sha256": contract_sha})
            loaded = loader(str(Path(runtime_root) / "models" / model), lock["asset_lock"], lock["asset_lock_sha256"],
                            lock["source_lock"], lock["source_lock_sha256"], declaration, contract_sha, decoder_raw)
            retain("model-load.json", loaded.receipt)
            inputs = validate_decoder(decoder_raw, digest(decoder_raw))
            for index, outcome in enumerate(outcomes):
                verify_scope()
                oid = outcome["opaque_id"]
                descriptor = inputs[oid]["descriptor"]
                bound = bind_wave(str((Path(input_root) / "audio").absolute()), WaveExpectation(oid,
                                  outcome["wav_sha256"] + ".wav", outcome["wav_sha256"], descriptor["wav"]["bytes"], descriptor["raw_pcm16"]["frame_count"]))
                require(bound.descriptor == descriptor and bound.descriptor_sha256 == inputs[oid]["binding_sha256"], "Per-call PCM identity")
                progress(output_root, "cell", index + 1)
                attempted += 1
                retain(oid + ".started.json", {"model": model_identity, "opaque_id": oid, "wav_sha256": outcome["wav_sha256"],
                       "input_binding_sha256": bound.descriptor_sha256, "execution_contract_sha256": contract_sha,
                       "decoder_manifest_sha256": digest(decoder_raw), "attempt": 1})
                previous = signal.signal(signal.SIGALRM, alarm)
                signal.setitimer(signal.ITIMER_REAL, 300)
                began = time.monotonic()
                forward_completed = False
                try:
                    row, raw_evidence = infer(loaded, bound, digest(decoder_raw))
                    require(type(row) is dict and set(row) == {"raw_text", "completeness", "quality_flags"}, "Adapter result")
                    require(type(row["raw_text"]) is str and len(row["raw_text"]) <= 32768
                            and row["completeness"] in ("complete", "incomplete", "unknown") and type(row["quality_flags"]) is list
                            and all(type(v) is str for v in row["quality_flags"]), "Adapter result types")
                    completed = dict(outcome, **row, status="success")
                    forward_completed = True
                except Exception as exc:
                    completed = dict(outcome, status="timeout" if isinstance(exc, ClipDeadline) else "error")
                    raw_evidence = {"exception_type": type(exc).__name__, "forward_completion_confirmed": False}
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                decoder_sha = retain(oid + ".decoder.json", {"model": model_identity, "opaque_id": oid,
                                     "wav_sha256": outcome["wav_sha256"], "input_binding_sha256": bound.descriptor_sha256,
                                     "wall_seconds": time.monotonic() - began, "evidence": raw_evidence})
                receipt = {k: completed[k] for k in ("opaque_id", "wav_sha256", "status", "raw_text", "completeness", "quality_flags")}
                receipt.update(schema="bounded-source-screen-asr-clip-receipt-v1", model=model_identity,
                               attempt_started=True, model_forward_completed=forward_completed, decoder_evidence_sha256=decoder_sha,
                               input_binding_sha256=bound.descriptor_sha256, decoder_manifest_sha256=digest(decoder_raw),
                               execution_contract_sha256=contract_sha)
                completed["execution_receipt_sha256"] = retain(oid + ".receipt.json", receipt)
                outcome.update(completed)
                retain(f"outcomes.{index + 1:04d}.json", outcomes)
                if outcome["status"] != "success":
                    break
    except Exception as exc:
        failure = {"exception_type": type(exc).__name__, "decode_attempts": attempted,
                   "remaining_outputs": "not_run", "retries": 0}
        retain("stage-failure.json", failure)
    finally:
        retain("outcomes.json", outcomes)
        status = "empty_generated_subset" if not outcomes else ("complete" if failure is None and all(row["status"] == "success" for row in outcomes) else "stopped")
        summary = {"schema": "bounded-source-screen-asr-summary-v1", "model": model_identity, "status": status,
                   "requested_clips": len(outcomes), "attempted": attempted, "success": sum(row["status"] == "success" for row in outcomes),
                   "not_run": sum(row["status"] == "not_run" for row in outcomes), "max_model_calls": len(outcomes),
                   "blind_job_sha256": digest(job_raw), "decoder_manifest_sha256": digest(decoder_raw), "wall_seconds": time.monotonic() - started,
                   "retries": 0, "context": "", "hotwords": [], "human_gold": False, "training_admitted": False,
                   "completeness_scope": "Decoder status only; acoustic completeness remains unknown"}
        retain("summary.json", summary)
        retain("raw-freeze.json", {"schema": "bounded-source-screen-asr-raw-freeze-v1", "labels_joined": False,
                                  "files": {path.name: digest(path.read_bytes()) for path in directory.iterdir() if path.is_file()}})
    return summary



def _terminal_write(path, value):
    raw = canonical(value)
    require(len(raw) <= 8 * 1024**2, "Terminal result cap")
    with Path(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def terminal_outcomes(input_root, output_root, model):
    """Host-only accounting after process exit; never imports helpers or models.

    This reads only the blind job, immutable start/receipt/evidence bytes and
    their contract bindings. Aggregate snapshots are deliberately ignored.
    Existing raw files/freezes are never rewritten. A possibly consumed attempt
    cannot become not_run merely because its final receipt was interrupted.
    """
    require(model in MODEL_IDS, "Model choice")
    input_root, output_root = Path(input_root), Path(output_root)
    job_raw = read_regular(input_root / "job.json", MAX_JOB_BYTES)
    job = validate_job(job_raw)
    output_root.mkdir(parents=True, exist_ok=True)
    require(output_root.is_dir() and not output_root.is_symlink(), "Terminal output directory")
    require(not os.path.lexists(output_root / "terminal-outcomes.json")
            and not os.path.lexists(output_root / "terminal-summary.json"), "Terminal accounting already exists")
    directory = output_root / model
    context_error = None
    context = None
    frozen = None
    if os.path.lexists(directory):
        try:
            require(directory.is_dir() and not directory.is_symlink(), "Raw model directory")
            contract_raw = read_regular(directory / "contract.json", MAX_JOB_BYTES)
            decoder_raw = read_regular(directory / "decoder-inputs.json", 1024**2)
            declaration, decoder = decode(contract_raw), decode(decoder_raw, 1024**2)
            require(type(decoder) is dict and set(decoder) == {"schema", "scope", "source_manifest_sha256", "order", "clips"}
                    and decoder["schema"] == "bounded-source-screen-asr-decoder-v1"
                    and decoder["scope"] == "audio_only_no_plan_context"
                    and decoder["source_manifest_sha256"] == digest(job_raw), "Terminal decoder/job binding")
            require(decoder["order"] == [row["audio_id"] for row in job["clips"]]
                    and type(decoder["clips"]) is list and len(decoder["clips"]) == len(job["clips"]), "Terminal full subset")
            locks = decode(read_regular(RETAINED / "model-locks.json", MAX_JOB_BYTES))
            require(digest(read_regular(RETAINED / "model-locks.json", MAX_JOB_BYTES)) == RETAINED_PINS["model-locks.json"], "Terminal model lock pin")
            require(canonical(declaration) == canonical(execution_contract(model, locks[model], decoder_raw)), "Terminal execution contract")
            identity = {"model_id": MODEL_IDS[model], "revision": locks[model]["asset_lock"]["revision"]}
            bindings = {}
            for expected, row in zip(job["clips"], decoder["clips"]):
                require(type(row) is dict and set(row) == {"opaque_id", "binding_sha256", "descriptor"}
                        and row["opaque_id"] == expected["audio_id"], "Terminal decoder row")
                descriptor = row["descriptor"]
                require(type(descriptor) is dict and descriptor["opaque_id"] == expected["audio_id"]
                        and descriptor["wav"]["sha256"] == expected["wav_sha256"]
                        and digest(canonical(descriptor)) == row["binding_sha256"], "Terminal PCM identity")
                bindings[expected["audio_id"]] = row["binding_sha256"]
            context = {"model": identity, "execution_contract_sha256": digest(contract_raw),
                       "decoder_manifest_sha256": digest(decoder_raw), "bindings": bindings}
            freeze_path = directory / "raw-freeze.json"
            if os.path.lexists(freeze_path):
                freeze = decode(read_regular(freeze_path, 1024**2), 1024**2)
                require(type(freeze) is dict and set(freeze) == {"schema", "labels_joined", "files"}
                        and freeze["schema"] == "bounded-source-screen-asr-raw-freeze-v1"
                        and freeze["labels_joined"] is False and type(freeze["files"]) is dict, "Terminal raw freeze")
                frozen = freeze["files"]
                for name, sha in frozen.items():
                    require(type(name) is str and Path(name).name == name and name not in (".", "..")
                            and type(sha) is str and SHA.fullmatch(sha), "Terminal freeze membership")
                for name, raw in (("contract.json", contract_raw), ("decoder-inputs.json", decoder_raw)):
                    require(frozen.get(name) == digest(raw), "Terminal frozen contract binding")
        except (OSError, ValueError, TypeError, KeyError):
            context_error = "invalid_or_incomplete_context"
    rows = []
    for expected in job["clips"]:
        oid = expected["audio_id"]
        names = {kind: oid + suffix for kind, suffix in (("start", ".started.json"), ("receipt", ".receipt.json"), ("decoder", ".decoder.json"))}
        present = {kind: os.path.lexists(directory / name) for kind, name in names.items()}
        row = {"opaque_id": oid, "wav_sha256": expected["wav_sha256"], "status": "not_run", "raw_text": None,
               "completeness": "unknown", "quality_flags": [], "execution_receipt_sha256": None,
               "started_marker_present": present["start"], "validation": "untouched",
               "human_gold": False, "training_admitted": False}
        recorded_attempt = any(present.values()) or (frozen is not None and any(name in frozen for name in names.values()))
        if recorded_attempt:
            row.update(status="failed_no_retry", validation="started_or_orphan_evidence_without_valid_receipt")
            try:
                require(context is not None and context_error is None and all(present.values()), "Incomplete attempt evidence")
                retained = {kind: read_regular(directory / name, 8 * 1024**2) for kind, name in names.items()}
                if frozen is not None:
                    require(all(frozen.get(names[kind]) == digest(raw) for kind, raw in retained.items()), "Frozen attempt evidence changed")
                start, receipt, evidence = (decode(retained[kind], 8 * 1024**2) for kind in ("start", "receipt", "decoder"))
                common = {"model": context["model"], "opaque_id": oid, "wav_sha256": expected["wav_sha256"],
                          "input_binding_sha256": context["bindings"][oid]}
                require(canonical(start) == canonical({**common, "execution_contract_sha256": context["execution_contract_sha256"],
                        "decoder_manifest_sha256": context["decoder_manifest_sha256"], "attempt": 1}), "Start identity")
                fields = set(common) | {"schema", "status", "raw_text", "completeness", "quality_flags", "attempt_started",
                         "model_forward_completed", "decoder_evidence_sha256", "decoder_manifest_sha256", "execution_contract_sha256"}
                require(type(receipt) is dict and set(receipt) == fields, "Receipt fields")
                require(all(receipt[key] == value for key, value in common.items())
                        and receipt["schema"] == "bounded-source-screen-asr-clip-receipt-v1"
                        and receipt["attempt_started"] is True
                        and receipt["decoder_evidence_sha256"] == digest(retained["decoder"])
                        and receipt["decoder_manifest_sha256"] == context["decoder_manifest_sha256"]
                        and receipt["execution_contract_sha256"] == context["execution_contract_sha256"], "Receipt bindings")
                require(type(evidence) is dict and set(evidence) == set(common) | {"wall_seconds", "evidence"}
                        and all(evidence[key] == value for key, value in common.items())
                        and type(evidence["wall_seconds"]) in (int, float) and math.isfinite(evidence["wall_seconds"])
                        and evidence["wall_seconds"] >= 0 and type(evidence["evidence"]) is dict, "Decoder evidence")
                require(receipt["status"] in ("success", "error", "timeout")
                        and receipt["completeness"] in ("complete", "incomplete", "unknown")
                        and type(receipt["quality_flags"]) is list
                        and all(type(flag) is str for flag in receipt["quality_flags"]), "Receipt outcome types")
                success = receipt["status"] == "success"
                require(receipt["model_forward_completed"] is success
                        and ((success and type(receipt["raw_text"]) is str and len(receipt["raw_text"]) <= 32768)
                             or (not success and receipt["raw_text"] is None)), "Receipt completion")
                row.update(status="success" if success else "failed_no_retry", raw_text=receipt["raw_text"],
                           completeness=receipt["completeness"], quality_flags=receipt["quality_flags"],
                           execution_receipt_sha256=digest(retained["receipt"]), validation="validated_receipt")
            except (OSError, ValueError, TypeError, KeyError):
                pass  # Conservative consumed-attempt status; raw bytes stay untouched.
        rows.append(row)
    attempted = sum(row["status"] != "not_run" for row in rows)
    success = sum(row["status"] == "success" for row in rows)
    summary = {"schema": "bounded-source-screen-asr-terminal-summary-v1", "model": model,
               "blind_job_sha256": digest(job_raw), "requested_clips": len(rows), "attempted": attempted,
               "success": success, "failed_no_retry": attempted - success, "not_run": len(rows) - attempted,
               "status": "empty_generated_subset" if not rows else ("complete" if success == len(rows) else ("not_run" if not attempted else "stopped")),
               "context_validation": context_error or ("validated" if context else "stage_never_started"),
               "raw_preserved": True, "snapshots_used": False, "human_gold": False, "training_admitted": False}
    _terminal_write(output_root / "terminal-outcomes.json", rows)
    _terminal_write(output_root / "terminal-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(argv)
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.12.14"
            and platform.machine() == "x86_64" and sys.platform == "linux" and sys.byteorder == "little", "Exact runtime host required")
    result = run(args.model, args.input, args.output, args.runtime)
    return 0 if result["status"] in ("complete", "empty_generated_subset") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "blocked", "exception_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
