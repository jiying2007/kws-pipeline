"""Fail-closed, stdlib-only public preparation artifact gate.

This module never downloads, uploads, computes features, loads models, or installs.
The CLI is a no-execution plan unless --package is explicitly supplied.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
from urllib.parse import unquote, urlsplit

PACKAGE_RELATIVE = 'research/token_preparation'
MIB = 1024 * 1024
PAYLOAD_CAP = 7 * MIB
TAR_CAP = 15 * MIB // 2
GZIP_CAP = 8 * MIB
ARCHIVE_NAME = 'token-preparation-artifact.tar.gz'
RECEIPT_NAME = ARCHIVE_NAME + '.sha256'
MANIFEST_NAME = 'artifact-manifest.json'
JSON_NAMES = frozenset(name + '.json' for name in (
    'native-features', 'official-features', 'feature-comparison', 'initial-control',
    'preparation-result', 'synthetic-ctc', 'input-reconstitution', 'build',
    'environment', 'resources', 'dependencies', 'final-status',
    'source-identities', 'rights-and-provenance'))
FAILURE_REQUIRED = frozenset(('final-status.json', 'source-identities.json',
                              'rights-and-provenance.json'))
JSON_CAPS = {'resources.json': MIB, 'dependencies.json': MIB // 2}
DEFAULT_JSON_CAP = 128 * 1024
RIGHTS_BASENAMES = frozenset(('QWEN-PUBLIC-RIGHTS.md','QWEN-PUBLIC-CATALOG.json',
    'A20-CORE-LICENSE-SCOPE.md','Apache-2.0.txt','BSD-2-Clause-torchaudio.txt','D20-SOURCE-NOTICE.md'))
PHASES = ('input_reconstitution', 'dependencies', 'build', 'features_control')
PHASE_STATUSES = frozenset(('SUCCEEDED', 'FAILED', 'NOT_RUN', 'INCOMPLETE', 'UNKNOWN'))
# These are public metadata identities/geometry, never samples or computed features.
GEOMETRY = (
 ('tts-candidate-011',42), ('tts-candidate-016',36), ('tts-candidate-017',74),
 ('tts-candidate-018',52), ('tts-candidate-020',74), ('tts36-v1-001',28),
 ('tts36-v1-002',49), ('tts36-v1-004',87), ('tts36-v1-006',34),
 ('tts36-v1-007',26), ('tts36-v1-008',39), ('tts36-v1-011',58),
 ('tts36-v1-012',31), ('voice54-v1-002',82), ('voice54-v1-027',39),
 ('voice54-v1-016',55), ('voice54-v1-003',55), ('voice54-v1-006',36),
 ('voice54-v1-031',84), ('voice54-v1-035',36), ('qwen3-kw1-dylan',63),
 ('qwen3-kw1-uncle_fu',60), ('qwen3-kw1-vivian',60), ('qwen3-kw2-dylan',63),
 ('qwen3-kw2-uncle_fu',74), ('qwen3-kw2-vivian',68),
 ('qwen3-repeat-nihao-dylan',58), ('qwen3-repeat-nihao-uncle_fu',84),
 ('qwen3-repeat-nihao-vivian',52), ('qwen3-suffix-kw2-dylan',52),
 ('qwen3-suffix-kw2-uncle_fu',95), ('qwen3-suffix-kw2-vivian',66))
RAW_SHAPES = {f'{group}/{name}.f32le': [rows,400]
              for group, items in (('native', GEOMETRY), ('official', GEOMETRY[:20]))
              for name, rows in items}
RAW_SHAPES['a20-initial-logits.f32le'] = [32,95,6]
ALLOWLIST = JSON_NAMES | frozenset(RAW_SHAPES)
SHA256 = re.compile(r'^[0-9a-f]{64}$')
URL_RE = re.compile(r'[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>"\x27]+')
SECRET_RE = re.compile(r'(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|'
    r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|'
    r'AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{16,})\b|'
    r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b|'
    r'\bBearer\s+[A-Za-z0-9._~+/-]{8,}|'
    r'\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+)', re.I)
BAD_KEY = re.compile(r'(?:^|_)(?:password|passwd|secret|secrets|credential|credentials|'
    r'authorization|cookie|cookies|hostname|host_name|host_id|source_host_id|runner_name|runner_id|runner_tracking_id|fqdn|nodename|mac_address|ip_address|boot_id|machine_id|'
    r'username|user_name|home_dir|home_directory|environ|environment_variables|'
    r'env|env_vars|env_dump|private_url|signed_url|stdout|stderr|traceback|raw_log|raw_logs|access_token|refresh_token|'
    r'auth_token|api_key|bearer_token)(?:_|$)', re.I)
ABS_PATH = re.compile(r'(?<![A-Za-z0-9])(?:/(?:[A-Za-z0-9_.~-]+/?)+|[A-Za-z]:[\\/]|\\\\|~/)')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
IPV4 = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')
UUID = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I)
PUBLIC_HOSTS = frozenset(('github.com', 'raw.githubusercontent.com', 'www.apache.org',
                          'apache.org', 'opensource.org', 'download-r2.pytorch.org',
                          'download.pytorch.org', 'files.pythonhosted.org', 'pypi.org'))

class GateError(ValueError):
    """The candidate must not be published; messages never echo private data."""

def require(ok, code):
    if not ok:
        raise GateError(code)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')

def _unique(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'duplicate_json_key')
        obj[key] = value
    return obj

def parse_json(raw):
    try:
        obj = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
             parse_constant=lambda _: (_ for _ in ()).throw(GateError('nonfinite_json')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise GateError('invalid_json') from exc
    require(type(obj) is dict, 'json_root_not_object')
    return obj

def _relative(name):
    require(type(name) is str and name and len(name) <= 200, 'invalid_member_path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and
            all(p not in ('', '.', '..') for p in path.parts) and
            '\\' not in name and re.fullmatch(r'[A-Za-z0-9_.\-/]+', name), 'invalid_member_path')
    return path

def _no_symlink_ancestors(path):
    p = Path(path).absolute()
    for part in (p, *p.parents):
        require(not part.is_symlink(), 'symlink_path')
    return p

def read_regular(root, relative, cap):
    """Use no-follow descriptor traversal and snapshot bytes exactly once."""
    path = _relative(relative)
    root = _no_symlink_ancestors(root)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(root, flags | os.O_DIRECTORY)
    try:
        for part in path.parts[:-1]:
            nextfd = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd); fd = nextfd
        filefd = os.open(path.parts[-1], flags, dir_fd=fd)
        try:
            before = os.fstat(filefd)
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'nonregular_or_hardlinked_member')
            require(0 <= before.st_size <= cap, 'member_size_cap')
            chunks, count = [], 0
            while True:
                data = os.read(filefd, min(65536, cap + 1 - count))
                if not data:
                    break
                chunks.append(data); count += len(data)
                require(count <= cap, 'member_size_cap')
            after = os.fstat(filefd)
            require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns) and
                    count == before.st_size, 'member_changed_during_read')
            return b''.join(chunks)
        finally:
            os.close(filefd)
    except OSError as exc:
        raise GateError('member_open_failed') from exc
    finally:
        os.close(fd)

def _urls(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _urls(key); yield from _urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from _urls(child)
    elif isinstance(value, str):
        yield from URL_RE.findall(value)

def _public_url(url):
    try:
        p = urlsplit(url)
        return (p.scheme == 'https' and p.hostname in PUBLIC_HOSTS and
                p.port is None and not p.username and not p.password and
                not p.query and not p.fragment and '%' not in re.sub(r'%2[bB]', '', url) and
                not any(x in ('.', '..') for x in p.path.split('/')) and len(p.path) > 1)
    except ValueError:
        return False

def privacy_check(obj, allowed_urls=frozenset()):
    """Reject, never silently redact: source producers must supply clean evidence."""
    nodes = 0
    def walk(value, depth=0):
        nonlocal nodes
        nodes += 1
        require(nodes <= 150000 and depth <= 24, 'json_complexity_cap')
        if isinstance(value, dict):
            for key, child in value.items():
                require(type(key) is str and not BAD_KEY.search(key.replace('-', '_')),
                        'private_metadata_key')
                walk(key, depth + 1); walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)
        elif isinstance(value, str):
            require(len(value) <= 8192 and not any(ord(c) < 32 and c not in '\n\r\t' for c in value),
                    'opaque_or_control_text')
            decoded = unquote(value)
            require(not SECRET_RE.search(decoded), 'secret_like_value')
            require(not EMAIL.search(decoded) and not UUID.search(decoded), 'personal_or_host_identifier')
            urls = list(URL_RE.finditer(value))
            for match in urls:
                url = match.group()
                require(url in allowed_urls and _public_url(url), 'unapproved_url')
            scrubbed = unquote(URL_RE.sub('', value))
            require(not re.search(r'(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{80,}(?![A-Za-z0-9+/=_-])', scrubbed),
                    'opaque_encoded_value')
            require(not ABS_PATH.search(scrubbed), 'private_absolute_path')
            require(not IPV4.search(scrubbed), 'network_identifier')
            require(not re.search(r'\b(?:localhost|[a-z0-9.-]+\.(?:internal|local|corp|lan))\b', scrubbed, re.I),
                    'private_hostname')
        elif type(value) is float:
            require(math.isfinite(value), 'nonfinite_json')
        else:
            require(value is None or type(value) in (bool, int), 'unsupported_json_type')
    walk(obj)

def _source_member(name):
    path = _relative(name)
    allowed_suffixes = {'.py','.c','.h','.json','.md','.txt','.lock','.yml','.yaml','.sh'}
    if name.startswith('.github/workflows/'):
        return len(path.parts) == 3 and path.suffix in {'.yml','.yaml'}
    prefix = PACKAGE_RELATIVE + '/'
    if not name.startswith(prefix):
        return False
    inner = PurePosixPath(name[len(prefix):])
    return (inner.suffix in allowed_suffixes and (len(inner.parts) == 1 or
            inner.parts[0] in {'src','metadata','tests'}))

def validate_sourcecontext(context):
    """Validate exact source-freeze bytes and every closed source member.

    context has source_root, source_freeze_sha256, rights_privacy_reviewed=True.
    Optional release_sha256 is identity-only. Context paths are never published.
    """
    require(type(context) is dict and set(context) <= {'source_root', 'source_freeze_sha256',
        'release_sha256', 'rights_privacy_reviewed'}, 'sourcecontext_keys')
    require(context.get('rights_privacy_reviewed') is True, 'rights_privacy_review_required')
    pin = context.get('source_freeze_sha256')
    require(type(pin) is str and SHA256.fullmatch(pin), 'source_freeze_pin_required')
    root = context.get('source_root')
    require(isinstance(root, (str, os.PathLike)), 'source_root_required')
    freeze_raw = read_regular(root, PACKAGE_RELATIVE + '/SOURCE-FREEZE.json', MIB)
    require(digest(freeze_raw) == pin, 'source_freeze_pin_mismatch')
    freeze = parse_json(freeze_raw)
    entries = freeze.get('files', freeze.get('entries'))
    require(type(entries) is list and 1 <= len(entries) <= 256, 'source_freeze_entries')
    blobs, total = {}, 0
    for entry in entries:
        require(type(entry) is dict and {'path', 'bytes', 'sha256'} <= set(entry), 'source_entry_schema')
        name = entry['path']; require(_source_member(name), 'source_member_scope')
        require(name not in blobs and name != PACKAGE_RELATIVE + '/SOURCE-FREEZE.json', 'source_entry_duplicate_or_self')
        require(type(entry['bytes']) is int and 0 <= entry['bytes'] <= 2*MIB and
                type(entry['sha256']) is str and SHA256.fullmatch(entry['sha256']), 'source_entry_identity')
        raw = read_regular(root, name, entry['bytes'])
        require(len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], 'source_entry_mismatch')
        total += len(raw); require(total <= 8*MIB, 'source_freeze_payload_cap')
        blobs[name] = raw
    required = (PACKAGE_RELATIVE + '/metadata/TRAIN32.json',
                PACKAGE_RELATIVE + '/metadata/DERIVED-OUTPUT-RIGHTS-PROVENANCE.json')
    require(all(name in blobs for name in required), 'required_source_metadata_not_frozen')
    train = parse_json(blobs[required[0]])
    rows = train.get('rows')
    require(type(rows) is list and len(rows) == 32 and
            [(r.get('recording'), r.get('model_rows')) for r in rows] == list(GEOMETRY), 'frozen_geometry_mismatch')
    require(train.get('development_audio_included') is False and
            all(r.get('role') == 'train' for r in rows), 'source_scope_mismatch')
    rights = parse_json(blobs[required[1]])
    require(type(rights.get('source_metadata')) is list and rights['source_metadata'] and
            type(rights.get('required_output_notices')) is list and rights['required_output_notices'],
            'source_rights_missing')
    seen_rights = set()
    for entry in rights['source_metadata']:
        require(type(entry) is dict and type(entry.get('file')) is str, 'rights_entry_schema')
        logical = _relative(entry['file'])
        basename = logical.name
        require(basename in RIGHTS_BASENAMES and basename not in seen_rights, 'rights_entry_name')
        name = PACKAGE_RELATIVE + '/metadata/rights/' + basename
        require(name in blobs and type(entry.get('bytes')) is int and len(blobs[name]) == entry['bytes'] and
                entry.get('sha256') == digest(blobs[name]), 'rights_source_identity_mismatch')
        seen_rights.add(basename)
    require(seen_rights == RIGHTS_BASENAMES, 'required_rights_missing')
    urls = set()
    for name, raw in blobs.items():
        if name.endswith('.json'):
            for url in _urls(parse_json(raw)):
                if _public_url(url):
                    urls.add(url)
    require(freeze.get('schema') == 'a20-execution-source-freeze-v1', 'source_freeze_schema')
    identity = {'source_freeze_sha256': pin,
        'train32_sha256': digest(blobs[required[0]]),
        'rights_provenance_sha256': digest(blobs[required[1]])}
    if 'release_sha256' in context:
        release = context['release_sha256']
        require(type(release) is str and SHA256.fullmatch(release), 'release_identity')
        identity['release_sha256'] = release
    return {'identity': identity, 'rights': rights, 'allowed_urls': frozenset(urls)}

def _inventory(root):
    root = _no_symlink_ancestors(root)
    require(root.is_dir(), 'input_directory_required')
    found = []
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), 'symlink_member')
            if stat.S_ISDIR(mode):
                require(relative in ('native', 'official'), 'unknown_directory')
            else:
                require(stat.S_ISREG(mode), 'nonregular_member')
                require(relative in ALLOWLIST, 'unknown_member')
                found.append(relative)
    return sorted(found)

def _shape_equal(value, expected):
    return type(value) is list and all(type(v) is int for v in value) and value == expected

def _fp32(raw, shape):
    require(len(raw) == math.prod(shape)*4, 'raw_shape_or_length_mismatch')
    require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)), 'nonfinite_fp32')

def _validate_features(group, evidence, blobs, status):
    names = sorted(name for name in blobs if name.startswith(group + '/'))
    if evidence is None:
        require(not names, 'raw_without_feature_evidence')
        return
    rows = evidence.get('rows')
    require(type(rows) is list, 'feature_evidence_rows')
    indexed = {}
    for row in rows:
        require(type(row) is dict, 'feature_row_schema')
        name = row.get('file')
        require(type(name) is str and name in RAW_SHAPES and name.startswith(group + '/') and
                name not in indexed, 'feature_row_name')
        require(name in blobs, 'feature_row_without_raw')
        raw = blobs[name]
        require(row.get('recording') == Path(name).stem and _shape_equal(row.get('shape'), RAW_SHAPES[name]) and
                type(row.get('bytes')) is int and row['bytes'] == len(raw) and
                row.get('sha256') == digest(raw), 'feature_evidence_identity')
        indexed[name] = row
    require(sorted(indexed) == names, 'feature_evidence_incomplete')
    if status == 'SUCCESS':
        require(len(names) == (32 if group == 'native' else 20), 'required_feature_missing')
        require(type(evidence.get('frontend_passes')) is int and evidence['frontend_passes'] == len(names)
                and type(evidence.get('model_calls')) is int and evidence['model_calls'] == 0,
                'feature_execution_counts_mismatch')

def validate_payload(blobs, status, source):
    require(status in ('SUCCESS', 'FAILED'), 'invalid_status')
    require(set(blobs) <= ALLOWLIST, 'unknown_member')
    required = JSON_NAMES if status == 'SUCCESS' else FAILURE_REQUIRED
    require(required <= set(blobs), 'required_evidence_missing')
    require(sum(map(len, blobs.values())) <= PAYLOAD_CAP, 'payload_cap')
    docs = {}
    for name, raw in blobs.items():
        if name in JSON_NAMES:
            require(len(raw) <= JSON_CAPS.get(name, DEFAULT_JSON_CAP), 'json_size_cap')
            docs[name] = parse_json(raw)
            privacy_check(docs[name], source['allowed_urls'])
        else:
            _fp32(raw, RAW_SHAPES[name])
    if status == 'SUCCESS':
        require(set(RAW_SHAPES) <= set(blobs), 'required_raw_missing')
    _validate_features('native', docs.get('native-features.json'), blobs, status)
    _validate_features('official', docs.get('official-features.json'), blobs, status)
    raw = blobs.get('a20-initial-logits.f32le')
    control = docs.get('initial-control.json')
    if raw is not None:
        require(control is not None and control.get('raw_file') == 'a20-initial-logits.f32le' and
                _shape_equal(control.get('raw_shape'), [32,95,6]) and control.get('raw_sha256') == digest(raw),
                'control_evidence_identity')
    elif control is not None:
        require(control.get('status') in ('NOT_RUN', 'FAILED', 'INCOMPLETE', 'UNKNOWN') and
                not any(key in control for key in ('raw_file', 'raw_shape', 'raw_sha256')),
                'control_raw_missing')
    final = docs['final-status.json']
    counts = {'native_feature_count': sum(n.startswith('native/') for n in blobs),
              'official_feature_count': sum(n.startswith('official/') for n in blobs),
              'control_logits_count': int(raw is not None)}
    require(final.get('status') == status and final.get('training_authorized') is False,
            'final_status_mismatch')
    for key, count in counts.items():
        require(type(final.get(key)) is int and final[key] == count, 'final_artifact_count_mismatch')
    require('model_forward_count' in final and
            (final['model_forward_count'] is None or type(final['model_forward_count']) is int
             and final['model_forward_count'] in (0,1)), 'model_forward_count_required')
    if raw is not None:
        require(final['model_forward_count'] == 1, 'control_execution_count_mismatch')
    phases = final.get('phases')
    require(type(phases) is dict and set(phases) == set(PHASES) and
            all(v in PHASE_STATUSES for v in phases.values()), 'explicit_phase_status_required')
    require(final.get('training_update_count') == 0 and type(final['training_update_count']) is int,
            'training_not_zero')
    if status == 'SUCCESS':
        require(all(v == 'SUCCEEDED' for v in phases.values()) and final['model_forward_count'] == 1,
                'success_phase_incomplete')
    else:
        require(any(v != 'SUCCEEDED' for v in phases.values()) and
                type(final.get('failure_code')) is str and
                re.fullmatch(r'[A-Z][A-Z0-9_]{0,95}', final['failure_code']), 'failure_status_incomplete')
    identities = docs['source-identities.json']
    require(all(identities.get(k) == v for k,v in source['identity'].items()), 'source_identity_mismatch')
    rights = docs['rights-and-provenance.json']
    require(rights.get('commercial_output_license') == 'not-established' and
            rights.get('distribution_scope') == 'bounded-public-research-preparation-only' and
            rights.get('new_license_grant') is False and
            rights.get('source_provenance_sha256') == source['identity']['rights_provenance_sha256'] and
            rights.get('source_metadata') == source['rights']['source_metadata'] and
            rights.get('required_output_notices') == source['rights']['required_output_notices'],
            'rights_scope_or_provenance_mismatch')
    if status == 'SUCCESS':
        preparation = docs['preparation-result.json']
        expected = {'native_waveform_passes':32, 'official_waveform_passes':20, 'model_forwards':1,
                    'optimizer_constructions':0, 'backwards':0, 'updates':0, 'decoder_calls':0,
                    'development_calls':0, 'F_arm_calls':0}
        require(all(type(preparation.get(k)) is int and preparation[k] == v for k,v in expected.items())
                and preparation.get('training_authorized') is False and
                preparation.get('source_freeze_sha256') == source['identity']['source_freeze_sha256'],
                'preparation_scope_mismatch')
        require(docs['synthetic-ctc.json'].get('status') == 'PASS' and
                type(docs['synthetic-ctc.json'].get('model_calls')) is int and
                docs['synthetic-ctc.json']['model_calls'] == 0, 'synthetic_ctc_status')
        require(type(control.get('model_forwards')) is int and type(control.get('cmvn_calls')) is int and
                control['model_forwards'] == control['cmvn_calls'] == 1 and
                all(type(control.get(k)) is int and control[k] == 0 for k in
                    ('optimizer_constructions', 'backwards', 'optimizer_steps', 'dropout_calls')),
                'control_scope_mismatch')
    return docs, counts

def build_archive(blobs, status, source):
    """Pure function: validate bytes; create and self-verify deterministic tar/gzip."""
    _, counts = validate_payload(blobs, status, source)
    manifest = {'schema': 'a20-public-preparation-artifact-manifest-v1', 'status': status,
        'source_identities': source['identity'], 'artifact_counts': counts,
        'missing_raw_members': sorted(set(RAW_SHAPES) - set(blobs)),
        'missing_evidence_members': sorted(JSON_NAMES - set(blobs)),
        'missing_means': 'Not included; never proof of a negative result or that execution did not happen.',
        'publication': {'retention_days': 1, 'visibility': 'public', 'maximum_upload_bytes': GZIP_CAP,
                        'requires_immediate_download_hash_and_private_persistence': True},
        'members': [{'path': n, 'bytes': len(blobs[n]), 'sha256': digest(blobs[n])}
                    for n in sorted(blobs)]}
    members = dict(blobs)
    members[MANIFEST_NAME] = canonical(manifest)
    require(len(members[MANIFEST_NAME]) <= DEFAULT_JSON_CAP, 'manifest_size_cap')
    require(sum(map(len, members.values())) <= PAYLOAD_CAP, 'payload_cap')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name]); info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ''
            archive.addfile(info, io.BytesIO(members[name]))
    tar = buffer.getvalue()
    require(len(tar) <= TAR_CAP, 'tar_cap')
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, filename='', mode='wb', mtime=0, compresslevel=9) as stream:
        stream.write(tar)
    packed = compressed.getvalue()
    receipt = f'{digest(packed)}  {ARCHIVE_NAME}\n'.encode('ascii')
    require(len(packed) + len(receipt) <= GZIP_CAP, 'publication_total_cap')
    # Read back exactly the in-memory bytes destined for disk, with no extraction.
    require(gzip.decompress(packed) == tar, 'gzip_readback_mismatch')
    with tarfile.open(fileobj=io.BytesIO(tar), mode='r:') as archive:
        infos = archive.getmembers()
        require([i.name for i in infos] == sorted(members), 'tar_members_mismatch')
        for info in infos:
            require(info.isfile() and not info.pax_headers and info.uid == info.gid == info.mtime == 0 and
                    info.uname == info.gname == '' and info.mode == 0o644 and
                    info.size == len(members[info.name]), 'tar_metadata_mismatch')
            stream = archive.extractfile(info)
            require(stream is not None and digest(stream.read()) == digest(members[info.name]),
                    'tar_member_digest_mismatch')
    return packed, receipt, manifest, len(tar)

def package_artifact(inputdir, outdir, status, sourcecontext):
    """Publish-ready local package only. Does not authorize or perform upload.

    outdir must be absent/empty, outside inputdir. Returns only
    filenames, hashes and counts; no private paths. Any failure produces no final
    output pair. Never downgrade a privacy/schema error into an uploadable result.
    """
    source = validate_sourcecontext(sourcecontext)
    root, output = _no_symlink_ancestors(inputdir), _no_symlink_ancestors(outdir)
    source_root = _no_symlink_ancestors(sourcecontext['source_root'])
    require(output != source_root, 'output_directory_overlap')
    for protected in (root,):
        require(output != protected and not output.is_relative_to(protected) and
                not protected.is_relative_to(output), 'output_directory_overlap')
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), 'output_directory_not_empty')
    names = _inventory(root)
    blobs = {name: read_regular(root, name, JSON_CAPS.get(name, DEFAULT_JSON_CAP)
             if name in JSON_NAMES else math.prod(RAW_SHAPES[name])*4) for name in names}
    packed, receipt, manifest, tar_bytes = build_archive(blobs, status, source)
    output.mkdir(parents=True, exist_ok=True)
    archive_path, receipt_path = output/ARCHIVE_NAME, output/RECEIPT_NAME
    created = []
    try:
        for path, raw in ((archive_path, packed), (receipt_path, receipt)):
            with path.open('xb') as stream:
                created.append(path)
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        require(archive_path.stat().st_size == len(packed) and
                digest(read_regular(output, ARCHIVE_NAME, GZIP_CAP)) == digest(packed), 'saved_archive_mismatch')
        require(read_regular(output, RECEIPT_NAME, 256) == receipt, 'saved_receipt_mismatch')
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {'schema': 'a20-public-preparation-package-receipt-v1', 'status': status,
        'archive': ARCHIVE_NAME, 'sha256': digest(packed), 'gzip_bytes': len(packed),
        'sha256_receipt': RECEIPT_NAME, 'publication_total_bytes': len(packed)+len(receipt),
        'tar_bytes': tar_bytes, 'payload_bytes': sum(map(len, blobs.values())) + len(canonical(manifest)),
        'member_count': len(manifest['members'])+1, 'artifact_counts': manifest['artifact_counts'],
        'retention_days': 1, 'upload_performed': False}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='store_true')
    parser.add_argument('--inputdir', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--status', choices=('SUCCESS', 'FAILED'))
    parser.add_argument('--sourcecontext', type=Path)
    args = parser.parse_args(argv)
    if not args.package:
        print(json.dumps({'status': 'PREPARATION_ONLY_NO_EXECUTION', 'upload_performed': False,
            'package_performed': False, 'allowed_member_count': len(ALLOWLIST),
            'payload_cap': PAYLOAD_CAP, 'tar_cap': TAR_CAP, 'publication_cap': GZIP_CAP}, sort_keys=True))
        return 0
    if any(value is None for value in (args.inputdir,args.outdir,args.status,args.sourcecontext)):
        parser.error('--package requires --inputdir, --outdir, --status and --sourcecontext')
    try:
        context = parse_json(read_regular(args.sourcecontext.parent, args.sourcecontext.name, 16*1024))
        print(json.dumps(package_artifact(args.inputdir,args.outdir,args.status,context), sort_keys=True))
    except (GateError, OSError) as exc:
        # OSError may contain private paths; never print repr/str or a traceback.
        print(json.dumps({'status':'BLOCKED_NO_PUBLICATION','error': str(exc) if isinstance(exc,GateError)
                          else 'filesystem_error','upload_performed':False}, sort_keys=True))
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
