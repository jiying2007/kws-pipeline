"""Metadata-only launcher for a fixed three-constructor sequence, never ASR.

The parent hosted diagnostic owns real-six admission, one-use host claim, setup,
capacity diagnostics and publication. No argument permits a decode mode/callback.
"""
from pathlib import Path
import json
import os
import re
import sys
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare_contract as c


def _shared(name):
    return c.fixed_module(c.OLD + name + '.py', name)


def build_metadata_packet(host_verified_input_dir, release, binding, plan, run_identity):
    """Rebind actual six host WAVs, then return only immutable metadata bytes.

    Caller must first finish the original complete source/screening admission.
    This function never substitutes test WAVs or trusts a supplied descriptor.
    """
    c.validate_release(release)
    contract = _shared('campaign_contract')
    worker = c.fixed_module('research/source-screen32-v1/asr_worker.py', 'prepare_retained_worker')
    core = c.ROOT / 'research/qwen6_asr/core'
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from pcm.pcm_binding import WaveExpectation, bind_wave, validate_pcm_descriptor
    worker.WaveExpectation, worker.bind_wave = WaveExpectation, bind_wave
    worker.validate_pcm_descriptor = validate_pcm_descriptor
    job, job_raw, decoder_raw = worker.bind_inputs(host_verified_input_dir)
    contract.validate_input_binding(binding, job, plan)
    files = {'job.json': job_raw, 'decoder-inputs.json': decoder_raw,
        'worker-input-binding.json': c.canonical(binding), 'execution-plan.json': c.canonical(plan)}
    files['host-binding.json'] = c.canonical({'schema': 'k1-prepare-host-binding-v1',
        'identity': run_identity, 'binding_pins': c.BINDING_PINS,
        'members': {name: {'bytes': len(raw), 'sha256': c.digest(raw)} for name, raw in sorted(files.items())},
        'wav_files_validated': 6, 'worker_audio_mounted': False, 'asr_permission_granted': False})
    c.validate_metadata_packet(files, release)
    return files


def worker_arguments(*, name, image, code, runtime, output, metadata, authority, family,
                     run_identity, freeze_sha, claim_sha):
    c.release_first()
    c.require(family in c.FAMILIES, 'fixed_prepare_family')
    session = _shared('runtime_session')
    # Reuse resource flags unchanged; replace the UNLAUNCHED input placeholder.
    stage = 'sensevoice' if family == 'whisper' else family
    args = session.container_args(name, image, code, runtime, output, 'asr', stage, metadata)
    index = args.index(image)
    args = args[:index]
    mounts = [i for i, value in enumerate(args) if value == '--mount' and ',dst=/input,' in args[i + 1]]
    c.require(len(mounts) == 1, 'single_unlaunched_metadata_mount')
    args[mounts[0] + 1] = args[mounts[0] + 1].replace(',dst=/input,', ',dst=/metadata,')
    args += ['--mount', f'type=bind,src={authority},dst=/authority,readonly']
    values = {'KWS_EXECUTED_HEAD_SHA': run_identity['executed_head_sha'],
        'KWS_RUN_ID': str(run_identity['run_id']), 'KWS_PREPARE_FREEZE_SHA256': freeze_sha,
        'KWS_PREPARE_CLAIM_SHA256': claim_sha}
    for key, value in values.items():
        c.require(type(value) is str and re.fullmatch('[a-zA-Z0-9_-]{1,128}', value), 'fixed_worker_environment')
        args += ['--env', key + '=' + value]
    return args + [image, '/runtime/venv/bin/python', '-I', '-B', '/code/' + c.REL + 'prepare_worker.py', family]


def validate_launch(record, args):
    """Inspect exact final mounted metadata, source, resources, gate and command."""
    config, host = record['Config'], record['HostConfig']
    image = config['Image']
    c.require(args.count(image) == 1 and args[0] == 'create', 'exact_prepare_image')
    index = args.index(image)
    command = args[index + 1:]
    worker = ['/runtime/venv/bin/python', '-I', '-B', '/code/' + c.REL + 'prepare_worker.py']
    gate = ['python', '-I', '-B', '/code/' + c.OLD + 'cpu_gate.py']
    c.require(len(command) == 9 and command[:4] == gate and command[4:8] == worker and
        command[8] in c.FAMILIES and config.get('Entrypoint') in (None, []) and
        config['Cmd'] == command, 'exact_prepare_command_no_mode')
    expected = {}
    for i, arg in enumerate(args[:index]):
        if arg == '--mount':
            fields = args[i + 1].split(',')
            spec = dict(part.split('=', 1) for part in fields if '=' in part)
            c.require(spec['dst'] not in expected, 'unique_mount_destination')
            expected[spec['dst']] = (spec['src'], 'readonly' not in fields)
    c.require(set(expected) == {'/code', '/runtime', '/output', '/metadata', '/authority', '/gate'} and
        all(expected[d][1] is False for d in ('/code', '/runtime', '/metadata', '/authority')) and
        all(expected[d][1] is True for d in ('/output', '/gate')), 'metadata_only_mount_contract')
    binds = [row for row in record['Mounts'] if row['Type'] == 'bind']
    c.require(len(binds) == len(expected) and {m['Destination'] for m in binds} == set(expected), 'exact_mount_roster')
    for row in binds:
        c.require(expected[row['Destination']] == (row['Source'], row['RW']) and
            row.get('Propagation') == 'rprivate', 'observed_mount_identity')
    c.require(all(m['Type'] in ('bind', 'tmpfs') for m in record['Mounts']), 'no_extra_volume')
    c.require({m['Destination'] for m in record['Mounts'] if m['Type'] == 'tmpfs'} == {'/scratch', '/tmp'},
              'fixed_tmpfs_roster')
    c.require(config.get('User') == args[args.index('--user') + 1] and
        config['User'].split(':')[0].isdigit() and int(config['User'].split(':')[0]) != 0,
        'nonroot_effective_worker_user')
    c.require(host.get('Memory') == 12 * 1024 ** 3 and host.get('MemorySwap') == 12 * 1024 ** 3 and
        host.get('NanoCpus') == 4_000_000_000 and host.get('PidsLimit') == 256 and
        host.get('NetworkMode') == 'none', 'prepare_resource_network_boundary')
    c.require(host.get('ReadonlyRootfs') is True and host.get('Privileged') is False and
        host.get('CgroupnsMode') == 'private' and host.get('PidMode') == '' and
        host.get('IpcMode') == 'private' and host.get('CapAdd') in (None, []) and
        set(host.get('CapDrop') or []) == {'ALL'} and
        any(x in ('no-new-privileges', 'no-new-privileges:true') for x in host.get('SecurityOpt', [])),
        'prepare_privilege_boundary')
    c.require(host.get('LogConfig') == {'Type': 'local', 'Config': {'max-size': '1m', 'max-file': '1',
        'compress': 'false'}}, 'bounded_private_local_logs')
    c.require(not any(host.get(key) for key in ('Binds', 'VolumesFrom', 'Devices', 'DeviceRequests',
        'ExtraHosts', 'Links', 'GroupAdd', 'PortBindings', 'PublishAllPorts')), 'no_extra_host_resource')
    supplied = {args[i + 1].split('=', 1)[0]: args[i + 1].split('=', 1)[1]
        for i, arg in enumerate(args[:index]) if arg == '--env'}
    observed = {}
    for entry in config['Env']:
        key, value = entry.split('=', 1)
        c.require(key not in observed, 'unique_worker_environment')
        observed[key] = value
    c.require(set(observed) <= set(supplied) | {'PATH', 'LANG', 'GPG_KEY', 'PYTHON_VERSION', 'PYTHON_SHA256'} and
        all(observed.get(k) == v for k, v in supplied.items()), 'no_secret_or_label_environment')
    return True


def prepare_supervise(docker, args, output, *, family, deadline):
    """Retained supervision algorithm, with the metadata-only launch verifier."""
    c.release_first()
    session = _shared('runtime_session')
    # These are already imported by fixed runtime_session, not arbitrary callbacks.
    from resource_host import gated_arguments, atomic, safe_error
    name = args[args.index('--name') + 1]
    args = gated_arguments(args, name, docker, '/code/' + c.OLD + 'cpu_gate.py')
    began = time.monotonic()
    row = {'name': name, 'family': family, 'mode': 'prepare', 'status': 'started',
        'worker_exit_success': False, 'primary_error': None, 'cleanup_error': None,
        'wall_seconds': 0.0, 'gate_result': None}
    docker.supervisions.append(row)
    docker.supervision_failure = None
    primary = cleanup = retention = None
    try:
        docker(*args)
        validate_launch(json.loads(docker('inspect', name))[0], args)
        docker('start', name)
        while not docker.completed(name):
            c.require(time.monotonic() < deadline, 'fixed_prepare_wall_ceiling')
            docker.sample_bytes(output, 'phase_output', 32 * c.MIB)
            time.sleep(.2)
        row['worker_exit_success'] = True
    except BaseException as error:
        primary = error
        row['primary_error'] = safe_error(error)
    finally:
        try:
            if name in docker.names and docker.names[name]['create_attempted']:
                docker.remove(name)
        except BaseException as error:
            cleanup = error
            row['cleanup_error'] = safe_error(error)
        row.update(wall_seconds=time.monotonic() - began, gate_result=docker.gate_result(name),
            status='complete' if primary is None and cleanup is None else 'failed')
        try:
            atomic(docker.root / 'supervision-receipt.json',
                {'schema': 'k1-asr18-supervision-v1', 'stages': docker.supervisions})
        except BaseException as error:
            retention = error
            row['status'] = 'failed'
    error = primary or cleanup or retention
    if error is not None:
        docker.supervision_failure = (error, row)
        raise error
    return row['wall_seconds']


def validate_result(value, family, run_identity, freeze_sha):
    expected = {'schema', 'identity', 'family', 'status', 'model_loads_started', 'model_loads_completed',
        'maximum_new_asr_calls', 'native_decode_attempts', 'audio_forward_calls', 'audio_mounted',
        'recognition_result', 'legacy_execution_contract_grants_asr', 'metadata_packet_sha256',
        'source_freeze_sha256', 'guard', 'wall_seconds'}
    c.require(type(value) is dict and set(value) == expected, 'exact_prepare_result')
    fixed = {'schema': 'k1-prepare-worker-result-v1', 'family': family, 'status': 'prepared_configuration_only',
        'model_loads_started': 1, 'model_loads_completed': 1, 'maximum_new_asr_calls': 0,
        'native_decode_attempts': 0, 'audio_forward_calls': 0, 'audio_mounted': False,
        'recognition_result': False, 'legacy_execution_contract_grants_asr': False,
        'source_freeze_sha256': freeze_sha}
    c.require(all(type(value[k]) is type(v) and value[k] == v for k, v in fixed.items()) and
        c.canonical(value['identity']) == c.canonical(run_identity) and
        type(value['metadata_packet_sha256']) is str and c.SHA.fullmatch(value['metadata_packet_sha256']), 'prepare_result_binding')
    c.require(c.canonical(value['guard']) == c.canonical({'schema': 'k1-prepare-forward-guard-v1', 'forbidden_attempts': 0,
        'python_entry_guard': 'active_through_constructor_and_config',
        'coverage': 'reviewed_fixed_python_native_and_module_forward_entrypoints',
        'universal_native_extension_proof': False}), 'completed_guard_observation')
    c.require(type(value['wall_seconds']) in (int, float) and 0 <= value['wall_seconds'] <= 600,
        'prepare_result_wall')
    return value


def run_prepares(root, setup, docker, run_identity, freeze_sha, claim_sha, setup_seconds):
    """Fixed Qwen→SenseVoice→Whisper sequence. There is no post-prepare callback."""
    c.release_first()
    root = Path(root)
    contract = _shared('campaign_contract')
    complete = []
    spent = setup_seconds
    for family in c.FAMILIES:
        print(json.dumps({'schema': 'k1-prepare-progress-v1', 'stage': 'prepare',
            'family': family, 'status': 'started'}, sort_keys=True), flush=True)
        remaining = min(600, 1800 - spent)
        c.require(remaining > 0, 'combined_setup_prepare_ceiling')
        output = contract.durable_mkdir(root / 'output' / family)
        args = worker_arguments(name=c.EXPERIMENT + '-' + str(run_identity['run_id']) + '-' + family,
            image=setup['image'], code=setup['code'], runtime=setup['runtimes'][family], output=output,
            metadata=root / 'metadata', authority=root / 'authority', family=family, run_identity=run_identity,
            freeze_sha=freeze_sha, claim_sha=claim_sha)
        docker.stage = 'setup'
        began = time.monotonic()
        try:
            prepare_supervise(docker, args, output, family=family,
                deadline=min(docker.deadline, began + remaining))
        finally:
            used = time.monotonic() - began
            spent += used
        c.require(used <= remaining, 'prepare_wall_threshold')
        receipt = c.regular(output / 'prepare-receipt.json')
        c.require(receipt == c.regular(output / 'worker-complete.json'), 'matching_prepare_terminal_bytes')
        value = validate_result(c.decode(receipt), family, run_identity, freeze_sha)
        claim = c.decode(c.regular(root / 'authority' / 'prepare-claim.json'))
        c.require(value['metadata_packet_sha256'] == claim['metadata_packet_sha256'], 'same_claimed_metadata')
        complete.append(family)
        print(json.dumps({'schema': 'k1-prepare-progress-v1', 'stage': 'prepare',
            'family': family, 'status': 'complete'}, sort_keys=True), flush=True)
    return {'prepared_families': complete, 'combined_setup_seconds': spent, 'asr_forward_calls': 0}
