"""Root-pinned public projection of the existing complete prepare posterior.

No network, model execution, audio, private inventory or custody fingerprint.
The projection's statements are not remote authentication. Only the separately
reviewed release pin authorizes these exact bytes after actual archive custody
acceptance. Synthetic local tests never become actual prepare evidence.
"""
from pathlib import Path
import campaign_contract as c
import asr18_metadata as metadata

HERE = Path(__file__).resolve().parent
NEW = 'research/k1-punctuation6-asr18-v1/'
EXPERIMENT = 'k1-punctuation6-prepare-diagnostic-r2'
REPOSITORY = 'jiying2007/kws-pipeline'
WORKFLOW = '.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml'
INPUT_KEYS = ('actual_input_freeze_sha256', 'actual_tts_authority_sha256',
    'actual_screening_receipt_sha256', 'actual_worker_binding_sha256', 'actual_opaque_order_sha256')
EXCLUDED = frozenset(NEW + name for name in ('campaign_contract.py', 'execution-release.json',
    'prepare_prerequisite.py', 'prepare-prerequisite.json'))
SCIENTIFIC_MEMBERS = ('research/k1-punctuation6-asr18-v1/asr18_metadata.py', 'research/k1-punctuation6-asr18-v1/cpu_budget.py', 'research/k1-punctuation6-asr18-v1/cpu_gate.py', 'research/k1-punctuation6-asr18-v1/native_locks.py', 'research/k1-punctuation6-asr18-v1/native_worker.py', 'research/k1-punctuation6-asr18-v1/optional_tools.py', 'research/k1-punctuation6-asr18-v1/runtime_scope.py', 'research/k1-punctuation6-asr18-v1/setup_adapter.py', 'research/k1-punctuation6-asr18-v1/whisper_binding.py', 'research/k1-punctuation6-asr18-v1/worker_receipts.py', 'research/qwen6_asr/FUNASR_MODEL_LICENSE', 'research/qwen6_asr/NOTICE.md', 'research/qwen6_asr/core/asr6_contract.py', 'research/qwen6_asr/core/asr_stage/__init__.py', 'research/qwen6_asr/core/asr_stage/adapters.py', 'research/qwen6_asr/core/asr_stage/architecture.py', 'research/qwen6_asr/core/asr_stage/assets.py', 'research/qwen6_asr/core/asr_stage/decoding.py', 'research/qwen6_asr/core/asr_stage/execution.py', 'research/qwen6_asr/core/asr_stage/runtime_support.py', 'research/qwen6_asr/core/pcm/pcm_binding.py', 'research/qwen6_asr/core/runtime_version_gate.py', 'research/qwen6_asr/core/sense_binding.py', 'research/qwen6_asr/core/vendor/__init__.py', 'research/qwen6_asr/core/vendor/checkpoint_guard/__init__.py', 'research/qwen6_asr/core/vendor/checkpoint_guard/checkpoint_contract.py', 'research/qwen6_asr/core/vendor/checkpoint_guard/exact_apply.py', 'research/qwen6_asr/core/vendor/qwen_loading_gate_v2.py', 'research/qwen6_asr/import_preflight.py', 'research/qwen6_asr/model-locks.json', 'research/qwen6_asr/runtime-lock.json', 'research/qwen6_asr/sense_parser.py', 'research/qwen6_asr/setup_diagnostics.py', 'research/qwen6_asr/setup_locked.py', 'research/qwen6_asr/wheel_identity.py', 'research/source-screen32-v1/asr_worker.py', 'research/source-screen32-v1/native_v2_adapter.py', 'research/source-screen32-v1/runtime_scope.py', 'research/source-screen32-v1/sense_adapter.py', 'research/source-screen32-whisper-v1/adapter.py', 'research/source-screen32-whisper-v1/common.py', 'research/source-screen32-whisper-v1/runtime-lock.json', 'research/source-screen32-whisper-v1/setup_runtime.py', 'research/source-screen32-whisper-v1/source-lock.json', 'tools/asr_native_contracts.py', 'tools/speech_asr_consensus.py', 'tools/speech_label_admission.py')
ACCEPTANCE = {'resource_budget': 'verified_complete_archive',
    'package': 'verified_complete_archive_and_git', 'custody': 'verified_root_reviewed_original_bytes',
    'no_audio_mount': 'verified_fixed_prepare_entrypoints'}
POSTERIOR_KEYS = ('evidence_kind', 'identity', 'numeric_job_id', 'result_commit', 'archive_sha256',
    'outcome', 'run_conclusion', 'job_conclusion', 'step_conclusions', 'host_receipt_status',
    'verified_prepared_families', 'verified_completed_model_constructions', 'diagnostic_forward_calls',
    'zero_forward_scope')
BINDING_KEYS = ('source_ref', 'result_ref', 'workflow', 'prepare_freeze_sha256', 'posterior_policy_sha256',
    'input_authorities', 'scientific_source_sha256', 'acceptance')
SEALED_RUNS = (38207026177, 38219171568, 38234429437, 38251170134)
SEALED_SOURCES = ('0f3cc89a3235c4b3af294dc6b5816ff9877c0f42',
    '054213c551eb83494b9a4ba8cd6d3ca4c1f488dd', '54ee9ee80e88c172cd04611474ded2613d956745',
    '97141cea41cc5cc18fac3f5a434777991bfd4864', 'd6ca13151f98a8fc80c2fbb1f21832a9524191db',
    'f25fd13109aa7e3b6e8bea8df61cff3ec2e74c2d')


def disabled_receipt():
    return {'schema': 'k1-asr18-prepare-prerequisite-v1', 'prepare_experiment': EXPERIMENT,
        **{key: None for key in POSTERIOR_KEYS + BINDING_KEYS},
        'source_ref': 'research/' + EXPERIMENT, 'result_ref': 'evidence/' + EXPERIMENT,
        'workflow': WORKFLOW, 'input_authorities': {key: None for key in INPUT_KEYS}}


def _scientific_identity(freeze, read):
    c.require(type(freeze) is dict and freeze.get('schema') == 'k1-asr18-source-freeze-v1' and
        type(freeze.get('inference_code')) is dict, 'fixed scientific inference closure required')
    roster = freeze['inference_code']
    c.require(set(roster) == set(SCIENTIFIC_MEMBERS) | EXCLUDED, 'exact scientific/control source roster')
    measured = {}
    for name in SCIENTIFIC_MEMBERS:
        raw = read(name)
        c.require(type(raw) is bytes and len(raw) <= 8 * c.MIB, 'bounded scientific source bytes')
        row = {'bytes': len(raw), 'sha256': c.digest(raw)}
        c.require(c.exact_json(row, roster[name]), 'verified scientific source byte identity')
        measured[name] = row
    projection = {'schema': 'k1-asr18-scientific-source-binding-v1',
        'metadata_experiment': metadata.EXPERIMENT, 'families': list(c.FAMILIES),
        'cells': list(c.CELLS), 'limits': c.LIMITS, 'source_files': measured}
    return c.digest(c.canonical(projection))


def scientific_identity(root, freeze):
    """Remeasure the explicit source roster, with no activation-control self-reference."""
    root = Path(root)
    return _scientific_identity(freeze, lambda name: c.read_bytes(root / name, cap=8*c.MIB))


def scientific_identity_from_members(members, freeze):
    """Same measurement over already validated complete archive members, never hashes alone."""
    c.require(type(members) is dict, 'actual archive byte members required')
    return _scientific_identity(freeze, lambda name: members['source/' + name])


def _validate_projection(value):
    c.require(type(value) is dict and set(value) == set(disabled_receipt()), 'exact public prerequisite allowlist')
    fixed = {'schema': 'k1-asr18-prepare-prerequisite-v1', 'prepare_experiment': EXPERIMENT,
        'evidence_kind': 'actual_read_only', 'source_ref': 'research/' + EXPERIMENT,
        'result_ref': 'evidence/' + EXPERIMENT, 'workflow': WORKFLOW,
        'outcome': 'complete_prepare_verified', 'run_conclusion': 'success', 'job_conclusion': 'success',
        'step_conclusions': {'contracts': 'success', 'host': 'success', 'publisher': 'success'},
        'host_receipt_status': 'all3_prepared_no_recognition', 'verified_prepared_families': list(c.FAMILIES),
        'verified_completed_model_constructions': 3, 'diagnostic_forward_calls': 0,
        'zero_forward_scope': 'fixed_reviewed_prepare_entrypoints_and_no_audio_mount_only',
        'acceptance': ACCEPTANCE}
    c.require(all(c.exact_json(value[key], expected) for key, expected in fixed.items()),
        'complete actual new prepare prerequisite required')
    identity = value['identity']
    c.require(type(identity) is dict and set(identity) ==
        {'repository', 'run_id', 'run_attempt', 'job_key', 'executed_head_sha'}, 'exact actual prepare identity')
    for key, expected in {'repository': REPOSITORY, 'run_attempt': 1, 'job_key': 'prepare_diagnostic'}.items():
        c.require(c.exact_json(identity[key], expected), 'fixed actual prepare identity')
    for number in (identity['run_id'], value['numeric_job_id']):
        c.require(type(number) is int and 0 < number < 10**20 and number not in SEALED_RUNS,
            'new numeric prepare run/job required')
    for head in (identity['executed_head_sha'], value['result_commit']):
        c.require(type(head) is str and c.HEAD.fullmatch(head) and head not in SEALED_SOURCES,
            'new actual prepare source/result required')
    c.require(identity['executed_head_sha'] != value['result_commit'], 'distinct actual prepare result commit')
    for key in ('archive_sha256', 'prepare_freeze_sha256', 'posterior_policy_sha256', 'scientific_source_sha256'):
        c.sha(value[key])
    inputs = value['input_authorities']
    c.require(type(inputs) is dict and set(inputs) == set(INPUT_KEYS), 'exact unchanged input authority binding')
    for key in INPUT_KEYS: c.sha(inputs[key])
    return value


def project_posterior(posterior, *, public_binding=None):
    """Allowlist an already verified full-success posterior; this does not release ASR.

    The reviewed posterior verifier emits public_prerequisite_binding only after
    complete archive, Git, jobs, resource and custody acceptance. Root independently
    reviews this output before pinning the receipt in a later ASR release.
    """
    c.require(type(posterior) is dict and posterior.get('schema') == 'k1-prepare-posterior-result-v1',
        'existing reviewed posterior output required')
    if public_binding is None: public_binding = posterior.get('public_prerequisite_binding')
    c.require(type(public_binding) is dict and set(public_binding) == set(BINDING_KEYS),
        'complete verified public posterior binding required')
    for key, expected in (('publisher_step_success', True), ('recognition_success_established', False),
            ('retry_authorized', False), ('consumed_asr_attempts', None),
            ('sealed_original18_consumed_asr_attempts', None), ('sealed_r1_18_consumed_asr_attempts', None)):
        c.require(key in posterior and c.exact_json(posterior[key], expected), 'posterior fixed nonrecognition scope')
    value = {'schema': 'k1-asr18-prepare-prerequisite-v1', 'prepare_experiment': EXPERIMENT,
        **{key: posterior[key] for key in POSTERIOR_KEYS}, **public_binding}
    return _validate_projection(value)


def validate_receipt(raw, release, *, source_freeze=None, source_root=None, source_members=None):
    c.require(type(raw) is bytes and len(raw) <= 65536, 'bounded canonical prerequisite bytes')
    c.validate_release_shape(release)
    expected = release['prepare_prerequisite_sha256']
    c.require(expected is not None, 'new prepare prerequisite is unestablished')
    c.require(c.digest(raw) == c.sha(expected), 'root-pinned exact prepare prerequisite bytes')
    value = _validate_projection(c.decode_json(raw))
    c.require(all(value['input_authorities'][key] == release[key] for key in INPUT_KEYS),
        'same fixed six inputs across prepare and ASR')
    if source_freeze is not None:
        actual = (scientific_identity_from_members(source_members, source_freeze) if source_members is not None
            else scientific_identity(HERE.parents[1] if source_root is None else source_root, source_freeze))
        c.require(value['scientific_source_sha256'] == actual, 'unchanged scientific source and parameters')
    return value
