#!/usr/bin/env python3
"""Verify the retained fourth-run ZIP and saved observations; never run a model.

Only standard-library code and the repository's existing offline validators are
used. Archive members remain in memory. --write-zip writes the original carrier
exclusively after every check succeeds; it never extracts executable members.
"""
import argparse
import ast
import base64
import binascii
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import struct
import wave
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = SOURCE.parents[1]
RUN = HERE / 'run-38126849922'
ZIP_SHA256 = 'b938916aeee60210863fb2f86bf7d1de89663dc9a1775e7910d702a64af14cf4'
ZIP_BYTES = 1843987
MAX_ARCHIVE = 4 * 1024**2
MAX_LOGICAL = 8 * 1024**2
MAX_MEMBER = 3 * 1024**2
MAX_FILES = 500
MAX_CHUNK = 1024**2
BOUNDARY = {'source_role': 'source_screen_only', 'training_admitted': False,
            'shipping_approved': False, 'human_gold': False,
            'acoustic_verbatim_completeness': 'unknown'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(data):
    return {'bytes': len(data), 'sha256': sha(data)}


def read_json(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON value')
    def finite(value):
        number = float(value)
        require(math.isfinite(number), 'nonfinite JSON number')
        return number
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite)


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and all(32 <= ord(c) < 127 for c in name), 'invalid path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and
            all(x not in ('', '.', '..') for x in name.split('/')), 'unsafe path')
    return name


def file_bytes(root, name, cap):
    safe_name(name)
    path = root / name
    require(not root.is_symlink() and root.is_dir(), 'archive root type')
    for index in range(1, len(Path(name).parts) + 1):
        require(not root.joinpath(*Path(name).parts[:index]).is_symlink(), 'symlink archive path')
    require(path.is_file() and path.stat().st_size <= cap, 'file type or size')
    with path.open('rb') as stream:
        data = stream.read(cap + 1)
    require(len(data) <= cap, 'file grew beyond cap')
    return data


def valid_identity(row, cap):
    require(type(row['bytes']) is int and 0 <= row['bytes'] <= cap, 'size cap')
    require(isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'SHA256 format')



def exact_member_stream(data, info):
    """Bound raw decompression independently of ZipFile's advertised-size truncation."""
    offset = info.header_offset
    require(0 <= offset <= len(data) - 30, 'ZIP local header bounds')
    header = struct.unpack_from('<4s5H3I2H', data, offset)
    require(header[0] == b'PK\x03\x04', 'ZIP local signature')
    flags, method, crc, compressed, decoded, name_size, extra_size = (
        header[2], header[3], header[6], header[7], header[8], header[9], header[10])
    require(flags == info.flag_bits and method == info.compress_type, 'ZIP local metadata')
    if not flags & 8:
        require((crc, compressed, decoded) == (info.CRC, info.compress_size, info.file_size), 'ZIP local sizes/CRC')
    start = offset + 30 + name_size + extra_size
    end = start + info.compress_size
    require(0 <= info.compress_size <= MAX_ARCHIVE and offset + 30 <= start <= end <= len(data), 'ZIP compressed bounds')
    payload = data[start:end]
    if info.compress_type == zipfile.ZIP_STORED:
        raw = payload
    else:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(payload, info.file_size + 1)
        require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data, 'ZIP deflate EOF/tails')
    require(len(raw) == info.file_size, 'ZIP raw expansion size')
    require(binascii.crc32(raw) & 0xffffffff == info.CRC, 'ZIP raw CRC')
    return raw


def decode_archive(data, manifest):
    """Closed, bounded generic parser also exercised with invented adversarial ZIPs."""
    require(set(manifest) == {'schema', 'archive_bytes', 'archive_sha256', 'logical_bytes', 'chunks', 'members'}, 'manifest fields')
    require(manifest['schema'] == 'whisper-screen-compact-archive-v1', 'archive schema')
    require(type(manifest['archive_bytes']) is int and 0 < manifest['archive_bytes'] <= MAX_ARCHIVE, 'archive cap')
    require(type(manifest['logical_bytes']) is int and 0 < manifest['logical_bytes'] <= MAX_LOGICAL, 'logical cap')
    require(len(data) == manifest['archive_bytes'] and sha(data) == manifest['archive_sha256'], 'archive identity')
    rows = manifest['members']
    require(isinstance(rows, list) and 0 < len(rows) <= MAX_FILES, 'member count cap')
    expected = {}
    for row in rows:
        require(set(row) == {'path', 'bytes', 'sha256'}, 'member fields')
        name = safe_name(row['path'])
        require(name not in expected, 'duplicate member manifest')
        valid_identity(row, MAX_MEMBER)
        expected[name] = {k: row[k] for k in ('bytes', 'sha256')}
    require(sum(row['bytes'] for row in rows) == manifest['logical_bytes'], 'logical total')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(expected), 'ZIP member count')
        names = [safe_name(info.orig_filename) for info in infos]
        require(all(info.orig_filename == info.filename for info in infos), 'ZIP normalized filename mismatch')
        require(len(set(names)) == len(names) and set(names) == set(expected), 'ZIP closed member set')
        result = {}
        for info in infos:
            mode = (info.external_attr >> 16) & 0xffff
            require(not info.is_dir() and not info.flag_bits & 1 and
                    stat.S_IFMT(mode) in (0, stat.S_IFREG), 'ZIP non-regular or encrypted member')
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), 'ZIP compression method')
            require(info.file_size == expected[info.filename]['bytes'], 'ZIP member size')
            raw = exact_member_stream(data, info)
            # ZipFile additionally validates local filename/header consistency and overlap.
            require(archive.read(info) == raw, 'ZIP local member consistency')
            require(identity(raw) == expected[info.filename], 'ZIP member hash')
            if info.filename.endswith('.json'):
                read_json(raw)
            result[info.filename] = raw
    return result


def read_archive(directory=RUN):
    directory = Path(directory)
    manifest = read_json(file_bytes(directory, 'ARCHIVE.json', 256 * 1024))
    chunks = manifest['chunks']
    require(isinstance(chunks, list) and 0 < len(chunks) <= 4, 'chunk count cap')
    data = []
    names = []
    for index, row in enumerate(chunks):
        require(set(row) == {'path', 'bytes', 'sha256'}, 'chunk fields')
        require(row['path'] == f'chunks/terminal.zip.part-{index:03d}', 'chunk order/path')
        valid_identity(row, MAX_CHUNK)
        require(row['bytes'] > 0, 'empty chunk')
        raw = file_bytes(directory, row['path'], MAX_CHUNK)
        require(identity(raw) == {k: row[k] for k in ('bytes', 'sha256')}, 'chunk identity')
        data.append(raw)
        names.append(row['path'])
    actual = list((directory / 'chunks').rglob('*'))
    require(not any(p.is_symlink() or not p.is_file() for p in actual), 'chunk tree kind')
    require({p.relative_to(directory).as_posix() for p in actual} == set(names), 'chunk closed set')
    archive = b''.join(data)
    # The immutable downloaded carrier is anchored independently of its manifest.
    require(len(archive) == ZIP_BYTES and sha(archive) == ZIP_SHA256, 'fourth-run ZIP identity')
    return archive, manifest, decode_archive(archive, manifest)


def check_boundary(record):
    require(all(type(record.get(k)) is type(v) and record[k] == v for k, v in BOUNDARY.items()), 'source-screen boundary')


def verify_records(files):
    """Recheck saved receipt/trace bindings. Does not claim runtime re-execution."""
    sys.path.insert(0, str(ROOT / 'tools'))
    sys.path.insert(0, str(SOURCE))
    from asr_native_contracts import validate_native_capture
    from adapter import termination
    data = {name: read_json(raw) for name, raw in files.items() if name.endswith('.json')}
    freeze = data['artifact-freeze.json']
    check_boundary(freeze)
    require(freeze['schema'] == 'whisper-screen-artifact-freeze-v1', 'artifact freeze schema')
    require(freeze['files'] == {name: identity(raw) for name, raw in files.items()
                               if name != 'artifact-freeze.json'}, 'artifact freeze exact membership')
    host = data['host-receipt.json']
    check_boundary(host)
    require(host['run_id'] == '38126849922' and host['head_sha'] == '1986acbeb790833a92f5014fccb88a931c0f3e7f'
            and host['run_attempt'] == host['run_number'] == 1, 'run identity')
    require(host['status'] == 'all_sixteen_completed' and host['attempts'] == host['completed'] == 16
            and host['terminal_validation_errors'] == 0, 'host completion')
    source_freeze = SOURCE / 'attempt-4/execution-freeze.json'
    require(host['source_freeze_sha256'] == sha(source_freeze.read_bytes()), 'source freeze identity')
    frozen = read_json(source_freeze.read_bytes())
    require(len(frozen['files']) == 262, 'source closure count')
    for name, digest in frozen['files'].items():
        require(sha(file_bytes(ROOT, name, 8 * 1024**2)) == digest, 'frozen source closure')
    limits = host['limits']
    require(limits['job_seconds'] == 2400 and 0 < host['wall_seconds'] < 2400
            and limits['setup_seconds'] == 1200 and limits['clips'] == 16
            and limits['new_whisper_calls'] == 16 and limits['download_bytes'] == 1610612736
            and all(limits[k] == 0 for k in ('new_tts_calls', 'new_qwen_calls', 'new_sensevoice_calls')), 'fixed budgets')
    require(host['campaign']['campaign_attempt'] == 4 and host['campaign']['automatic_retry'] is False
            and host['campaign']['same_model_and_input'] is True, 'campaign scope')
    for name in ('runtime-lock.json', 'source-lock.json'):
        require(files['setup/' + name] == (SOURCE / name).read_bytes(), 'original lock identity')
    source_lock = data['setup/source-lock.json']
    source_files = {name.removeprefix('setup/official-source/'): raw for name, raw in files.items()
                    if name.startswith('setup/official-source/')}
    # Official upstream support/source only. No wheels, checkpoint, audio or logs.
    require(len(source_files) == 22 and set(source_files) == {r['path'] for r in source_lock['files']}, 'official source set')
    for row in source_lock['files']:
        require(identity(source_files[row['path']]) == {'bytes': row['size_bytes'], 'sha256': row['sha256']}, 'official source hash and size')
    require(all(name.endswith('.json') or name.startswith('setup/official-source/') for name in files), 'public member scope')
    require(not any(name.endswith(('.whl', '.pt', '.pth', '.wav', '.flac', '.log')) for name in files), 'excluded runtime bytes')
    setup = data['setup/setup-receipt.json']
    check_boundary(setup)
    require(setup['status'] == 'complete' and setup['model_calls'] == setup['source_builds'] == 0, 'setup result')
    for name in ('asset-lock', 'installed-inventory', 'runtime-lock', 'source-lock'):
        key = name.replace('-', '_') + '_sha256'
        require(setup[key] == sha(files[f'setup/{name}.json']), 'setup inventory binding')
    for phase in ('install', 'source', 'model', 'finalize'):
        require(data[f'setup/phase-{phase}-complete.json']['status'] == 'complete', 'phase completion')
    acquisition = data['setup/bounded-acquisition/receipt.json']
    require(acquisition['status'] == 'complete' and acquisition['aggregate_cap'] == 1610612736
            and acquisition['total_read_bytes'] == 774057552 and len(acquisition['assets']) == 24, 'acquisition summary')
    require(sum(r['budget_charge_bytes'] for r in acquisition['assets']) == acquisition['total_read_bytes'], 'acquisition total')
    previous = 0
    assets = [f'wheel-{n:02d}' for n in range(1, 23)] + ['source', 'model']
    for index, (name, event) in enumerate(zip(assets, acquisition['assets'])):
        require(event['asset_id'] == name and event == data[f'setup/bounded-acquisition/budget-{index:03d}-final.json'], 'budget sequence')
        broker = data[f'setup/bounded-acquisition/{name}-broker.json']
        require(broker['prior_charged_bytes'] == previous and broker['charged_bytes'] == event['budget_charge_bytes']
                and broker['status'] == 'complete' and broker['failure_code'] is None and broker['plan_complete'] is True, 'broker chain')
        asset = broker['assets'][name]
        require(asset['complete'] is True and asset['connections_started'] == asset['transport_closed'] == (22 if name == 'source' else 1)
                and asset['downstream_read_bytes'] + asset['upstream_read_bytes'] == asset['total_read_bytes'] == broker['charged_bytes'], 'broker counters')
        previous += event['budget_charge_bytes']
        require(previous == broker['total_read_bytes'] <= acquisition['aggregate_cap'], 'broker cumulative charge')
    require(all(r['status'] == 'verified' and r['charge_kind'] == 'observed' and
                r['budget_charge_bytes'] == r['observed_ciphertext_bytes'] <= r['allowance'] for r in acquisition['assets']), 'observed acquisition rows')
    wheels = data['setup/runtime-lock.json']['files']
    require(len(wheels) == 22, 'fixed wheel set')
    for index, wheel in enumerate(wheels):
        result = data[f'setup/bounded-acquisition/wheel-{index+1:02d}-worker-result.json']
        require(result['schema'] == 'bounded-pip-worker-result-v2' and result['status'] == 'ok'
                and result['stage'] == 'complete' and result['pip_version'] == '25.0.1'
                and result['help_exit_code'] == 0 and result['missing_options'] == ['--resume-retries']
                and result['resume_policy'] == 'audited-25.0.1-no-resume'
                and result['reviewed_source_matches'] is True and result['asset_index'] == index
                and result['sha256'] == wheel['sha256'] and result['size_bytes'] == wheel['size_bytes'], 'pip observed compatibility and wheel result')
    pip_source = ast.parse((SOURCE / 'attempt-3/pip_compatibility.py').read_bytes())
    pip_hashes = next(ast.literal_eval(node.value) for node in pip_source.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == 'SOURCE_SHA256' for target in node.targets))
    require(len(pip_hashes) == 11, 'frozen pip audit definition')
    verify_scopes(data)
    cleanups = [r for name, r in data.items() if '/bounded-host-cleanup-' in name]
    require(len(cleanups) == 26 and all(r['status'] == 'complete' for r in cleanups), 'cleanup receipts')
    containers = [c for r in cleanups for c in r['containers']]
    require(len(containers) == len({c['container'] for c in containers}) == 50 and all(c['status'] == 'removed' and c['deadline_overrun'] is False for c in containers), 'bounded containers removed')
    diagnostic = data['container-sensevoice.json']
    require(diagnostic['status'] == 'complete' and diagnostic['cleanup_failure'] is None
            and diagnostic['container_state'] == {'ExitCode': 0, 'OOMKilled': False, 'Running': False}, 'inference container cleanup')
    config = data['whisper/native-config.json']
    original_job = file_bytes(ROOT, 'research/source-screen32-v1/evidence/run-38018029787/blind/job.json', 65536)
    input_lock = read_json(file_bytes(ROOT, 'research/source-screen32-v1/continuation-lock.json', 65536))
    ranks = {int(rank): base64.b64decode(token) for token, rank in
             (line.split() for line in files['setup/official-source/whisper/assets/multilingual.tiktoken'].splitlines())}
    require(len(ranks) == 50257, 'tokenizer rank count')
    rows = data['terminal-outcomes.json']
    job = data['whisper/blind-job.json']
    require(job == read_json(original_job), 'original blind job semantic identity')
    require(len(rows) == len(job['clips']) == len(input_lock['clips']) == 16 and [r['audio_id'] for r in rows] ==
            [f'clip-{n:06d}' for n in range(1, 17)], 'fixed 16-row denominator')
    for row, clip, locked in zip(rows, job['clips'], input_lock['clips']):
        check_boundary(row)
        prefix = 'whisper/' + row['audio_id'] + '/'
        raw = data[prefix + 'raw-output.json']
        done = data[prefix + 'completed.json']
        start = data[prefix + 'started.json']
        require(row['audio_id'] == clip['audio_id'] == done['audio_id'] == start['audio_id'] and
                row['wav_sha256'] == clip['wav_sha256'] == done['wav_sha256'] == start['wav_sha256'], 'clip identity')
        require(row['status'] == 'completed_native_eot' and row['attempts'] == start['attempt'] == done['attempt'] == 1
                and row['validation_errors'] == [] and row['stop_reason'] == 'eot', 'one completed attempt')
        for key, name in [('started_sha256', 'started.json'), ('input_capture_sha256', 'input-capture.json'),
                          ('pre_finalize_sha256', 'pre-finalize.json'), ('raw_output_sha256', 'raw-output.json')]:
            require(done[key] == sha(files[prefix + name]), 'completion byte binding')
        require(row['raw_output_sha256'] == sha(files[prefix + 'raw-output.json']) and row['raw_text'] == raw['raw_text'], 'terminal raw binding')
        require(start['native_config_sha256'] == sha(files['whisper/native-config.json']) and
                start['input_job_sha256'] == sha(original_job), 'started input/config binding')
        require(raw['decoder_completion_validated'] is True and raw['full_input_processed'] is True, 'native completion')
        validate_native_capture(config, raw['native_capture'])
        count = raw['native_capture']['termination']['step_count']
        names = [prefix + f'update-{n:03d}.json' for n in range(1, count + 1)]
        require(set(names) == {n for n in files if n.startswith(prefix + 'update-')}, 'exact durable updates')
        steps = [data[n] for n in names]
        require(termination(steps, config['effective_config']) == raw['native_capture']['termination'], 'natural EOT from durable trace')
        require(data[prefix + 'pre-finalize.json'] == {'token_ids': steps[-1]['token_ids'],
                'last_update_completed': True, 'trace_count': count}, 'pre-finalize trace binding')
        require(raw['returned_token_ids'] == raw['native_capture']['termination']['generated_token_ids'][:-1], 'returned natural EOT suffix')
        captured = data[prefix + 'input-capture.json']
        require(captured['native_input'] == raw['native_capture']['input'], 'native input binding')
        decoded = b''.join(ranks[t] for t in raw['returned_token_ids']).decode('utf-8', errors='replace').strip()
        require(decoded == raw['raw_text'], 'independent token decoding')
        require(locked['audio_id'] == clip['audio_id'] and locked['wav_sha256'] == clip['wav_sha256'], 'original input lock')
        wav = file_bytes(ROOT, input_lock['reference_directory'] + '/' + locked['producer_path'], 1024**2)
        require(identity(wav) == {'bytes': locked['bytes'], 'sha256': locked['wav_sha256']}, 'original WAV identity')
        with wave.open(io.BytesIO(wav), 'rb') as stream:
            frames = stream.getnframes()
            require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
                    == (16000, 1, 2, 'NONE') and frames == locked['frames'], 'original WAV format')
            pcm = stream.readframes(frames + 1)
        require(len(pcm) == frames * 2 and sha(pcm) == locked['pcm_sha256'], 'original PCM identity')
        floats = b''.join(struct.pack('<f', value[0] / 32768.0) for value in struct.iter_unpack('<h', pcm))
        require(captured['pcm16_sha256'] == sha(pcm) and captured['float32_source_sha256'] == sha(floats)
                and captured['float32_padded_sha256'] == sha(floats + bytes((480000 - frames) * 4)), 'PCM float32 padding identity')
        require(raw['native_capture']['input']['audio_frames'] == frames and
                0 < raw['wall_seconds'] < 300 and raw['process_peak_rss_kib'] * 1024 < 12 * 1024**3, 'clip input and budgets')
    require(data['whisper/worker-complete.json']['calls'] == 16, 'worker call count')
    return {'archive_members': len(files), 'completed_native_eot': 16, 'attempts': 16,
            'acquisition_counted_bytes': acquisition['total_read_bytes'], 'source_role': 'source_screen_only',
            'training_admitted': False, 'shipping_approved': False, 'human_gold': False}


def verify_scopes(data):
    """Check retained resource/download records; no claim about absent kernels."""
    for asset in [f'wheel-{n:02d}' for n in range(1, 23)] + ['source', 'model']:
        for role in ('broker', 'worker'):
            scope = data[f'setup/bounded-acquisition/{asset}-{role}-scope.json']
            limits = ({'cpus': 1, 'memory': 268435456, 'network': 'bridge', 'pids': 16} if role == 'broker'
                      else {'cpus': 3, 'memory': 12616466432, 'network': 'none', 'pids': 240})
            require(scope['role'] == role and scope['status'] == 'verified' and scope['socket_mounted'] is True
                    and scope['limits'] == limits, 'container inspect resource scope')
            name = f'setup/bounded-acquisition/{asset}-{role}-kernel.json'
            if role == 'worker' and asset in ('source', 'model'):
                require(name not in data, 'absent source/model worker kernel remains explicit')
                continue
            kernel = data[name]
            require(kernel['role'] == role and kernel['uid'] == 1001 and kernel['cgroup'] == '0::/'
                    and kernel['readonly_root'] is True and kernel['private_readonly_cgroup'] is True
                    and kernel['interfaces'] == (['eth0', 'lo'] if role == 'broker' else ['lo']), 'kernel isolation scope')
            require(kernel['limits'] == {'cpu.max': str(limits['cpus'] * 100000) + ' 100000',
                    'memory.max': str(limits['memory']), 'memory.swap.max': '0', 'pids.max': str(limits['pids'])}, 'kernel resource scope')
            require(kernel['status'] == {**{k: '0000000000000000' for k in ('CapAmb', 'CapBnd', 'CapEff', 'CapInh', 'CapPrm')},
                    'NSpid': '1', 'NoNewPrivs': '1', 'Seccomp': '2'}, 'kernel privilege scope')
    for phase in ('install', 'finalize'):
        scope = data[f'setup/bounded-acquisition/{phase}-offline-scope.json']
        require(scope['role'] == 'offline' and scope['status'] == 'verified' and scope['socket_mounted'] is False
                and scope['limits'] == {'cpus': 4, 'memory': 12884901888, 'network': 'none', 'pids': 256}, 'offline phase scope')
    wheels = data['setup/runtime-lock.json']['files']
    sources = data['setup/source-lock.json']['files']
    model = data['setup/asset-lock.json']['files']['small.pt']
    require(model == {'bytes': 483617219, 'sha256': '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794'}, 'fixed checkpoint identity receipt')
    expected = {'wheels/' + r['filename']: {'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in wheels}
    expected.update({'source/' + r['path']: {'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in sources})
    expected['model/small.pt'] = model
    require(data['setup/download-receipt.json']['files'] == expected, 'exact 45 locked download receipts')
    for phase, rows in [('source', sources), ('model', [{'size_bytes': model['bytes'], 'sha256': model['sha256']}])]:
        progress = data['setup/phase-' + phase + '-progress.json']
        require(progress['retry_count'] == 0 and len(progress['assets']) == progress['verified_assets'] == len(rows), 'source/model no retry')
        for index, (record, row) in enumerate(zip(progress['assets'], rows), 1):
            require(record == {'expected_bytes': row['size_bytes'], 'expected_sha256': row['sha256'], 'group': phase,
                    'http_status': 200, 'index': index, 'observed_sha256': row['sha256'],
                    'response_body_bytes_observed': row['size_bytes'], 'status': 'verified'}, 'source/model body receipt')
        require(progress['response_body_bytes_observed'] == sum(r['size_bytes'] for r in rows), 'source/model body total')


def verify(directory=RUN, output=None):
    archive, manifest, files = read_archive(directory)
    require(len(files) == 431 and sum(map(len, files.values())) == 4636835, 'fourth-run logical identity')
    provenance = read_json(file_bytes(Path(directory), 'provenance.json', 256 * 1024))
    require(provenance['schema'] == 'whisper-screen-fourth-run-retention-v1', 'provenance schema')
    require(provenance['archive_manifest_sha256'] == sha(file_bytes(Path(directory), 'ARCHIVE.json', 256 * 1024)), 'provenance manifest binding')
    require(provenance['zip_sha256'] == ZIP_SHA256 and provenance['zip_bytes'] == ZIP_BYTES, 'provenance carrier identity')
    check_boundary(provenance)
    expected_receipts = {'verification-receipt.json': '4e40bff753892fef4cd5754f786fa35a759f6f1dbfdb59243ceacf0dbe09845e',
                         'scope-verification.json': '04228e6597ea7e30a614604facc02f648a8ca004f7d99d00f94b8121011ee6a2'}
    require(provenance['verification_receipts'] == expected_receipts, 'verification receipt identity set')
    receipts = {}
    for name, digest in expected_receipts.items():
        raw = file_bytes(Path(directory), name, 256 * 1024)
        require(sha(raw) == digest, 'independent verification receipt identity')
        receipts[name] = read_json(raw)
    recorded = receipts['verification-receipt.json']
    supplemental = receipts['scope-verification.json']
    require(provenance['limitations'] == recorded['limitations'] + [supplemental['source_model_worker_kernel_record_limitation']], 'uncertainty retention')
    observed_keys = ('conclusion', 'denominator', 'audio_attempts', 'audio_completed', 'terminal_validation_errors',
        'natural_pre_finalize_eot_verified', 'original_wav_and_pcm_float32_padding_verified',
        'native_token_to_raw_text_independently_decoded', 'wheel_count', 'official_source_file_count',
        'pip_version', 'pip_resume_policy', 'pip_reviewed_source_hash_count', 'pip_reviewed_source_matches_records',
        'host_wall_seconds', 'workflow_execution_step_seconds', 'aggregate_setup_tunnel_bytes',
        'aggregate_setup_tunnel_cap_bytes', 'aggregate_setup_tunnel_scope', 'removed_setup_containers',
        'cleanup_records', 'setup_cleanup_deadline_overruns', 'inference_cleanup_complete', 'inference_container_seconds',
        'container_inspect_resource_scopes_verified', 'kernel_resource_scopes_verified',
        'download_receipt_exact_locked_members_verified', 'official_source_http_200_body_records_verified',
        'model_http_200_body_records_verified')
    require(provenance['observed'] == {key: recorded.get(key, supplemental.get(key)) for key in observed_keys}, 'observed provenance binding')
    require(provenance['run_id'] == 38126849922 and provenance['head_sha'] == recorded['head_sha']
            and provenance['executed_tree'] == recorded['executed_tree'] and provenance['source_entry_count'] == 262
            and provenance['source_freeze_sha256'] == recorded['source_freeze_sha256']
            and provenance['artifact_id'] == 11696898706 and provenance['run_attempt'] == provenance['run_number'] == 1
            and provenance['campaign_attempt'] == 4 and provenance['prior_run_ids'] == [38105471896, 38107706348, 38112558973], 'provenance run/source binding')
    result = verify_records(files)
    if output is not None:
        # Exclusive creation also refuses existing symlinks. No member extraction.
        with Path(output).open('xb') as stream:
            stream.write(archive)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-zip', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(output=args.write_zip), sort_keys=True))
