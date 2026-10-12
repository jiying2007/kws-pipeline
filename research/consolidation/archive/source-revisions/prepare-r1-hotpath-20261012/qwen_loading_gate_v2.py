"""Preparation-only Qwen3-ASR-0.6B single-file gate; stdlib, no runtime imports.
A successful fixture is not model compatibility, execution permission, or an approved
expected-state schema. Actual use additionally requires the surrounding stage locks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import weakref

REQUIRED_LOADING_FIELDS = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
MAX_HEADER_BYTES = 16 * 1024 * 1024
# This intentionally supports only exact integer/boolean identity and lossless
# F32 identity / BF16->F32. Any other conversion needs another reviewed recipe.
DTYPES = {
    'F32': (4, 'torch.float32'), 'BF16': (2, 'torch.float32'),
    'I64': (8, 'torch.int64'), 'I32': (4, 'torch.int32'),
    'I16': (2, 'torch.int16'), 'I8': (1, 'torch.int8'),
    'U8': (1, 'torch.uint8'), 'BOOL': (1, 'torch.bool'),
}
_VERIFIED = {}


def _forget(model):
    entry = _VERIFIED.get(id(model))
    if entry is not None and entry[0]() is model:
        del _VERIFIED[id(model)]


def _seal(model, receipt_sha256):
    key = id(model)
    def expired(dead_ref):
        entry = _VERIFIED.get(key)
        if entry is not None and entry[0] is dead_ref:
            del _VERIFIED[key]
    _VERIFIED[key] = (weakref.ref(model, expired), receipt_sha256)


def _same_instance_seal(model):
    entry = _VERIFIED.get(id(model))
    return entry[1] if entry is not None and entry[0]() is model else None


class QwenLoadGateError(RuntimeError):
    def __init__(self, message, *, loading_info_raw=None, evidence=None):
        super().__init__(message)
        # Keep the complete original object, including malformed/unknown fields.
        # Do not stringify/truncate it or accidentally claim it is JSON evidence.
        self.loading_info_raw = loading_info_raw
        self.evidence = evidence


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _json_copy(value):
    # Reject custom classes and non-JSON values without invoking their methods.
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if value != value or value in (float('inf'), -float('inf')):
            raise QwenLoadGateError('Nonfinite loading_info evidence')
        return value
    if type(value) is list:
        return [_json_copy(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _json_copy(v) for k, v in value.items()}
    raise QwenLoadGateError('loading_info must contain only plain JSON data')


def validate_loading_info(loading_info):
    """All four lists must exist, be actual lists, and be empty. No filtering."""
    try:
        evidence = _json_copy(loading_info)
        if type(loading_info) is not dict:
            raise QwenLoadGateError('loading_info must be a plain dict')
        for name in REQUIRED_LOADING_FIELDS:
            if name not in loading_info:
                raise QwenLoadGateError('loading_info missing required field: ' + name)
            if type(loading_info[name]) is not list:
                raise QwenLoadGateError('loading_info field must be a list: ' + name)
            if loading_info[name]:
                raise QwenLoadGateError('loading_info field is nonempty: ' + name)
        return evidence
    except (QwenLoadGateError, RecursionError) as exc:
        raise QwenLoadGateError(str(exc), loading_info_raw=loading_info,
                               evidence=locals().get('evidence')) from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QwenLoadGateError('Duplicate JSON header key: ' + key)
        result[key] = value
    return result


def _sha(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise QwenLoadGateError('Missing/malformed SHA256: ' + name)


def validate_expected_schema(schema, approved_sha256):
    """The digest MUST originate in a separately approved architecture/state lock.
    Creating this schema from the checkpoint or loaded model is circular evidence.
    """
    _sha(approved_sha256, 'expected-state schema')
    if type(schema) is not dict or not schema:
        raise QwenLoadGateError('Expected-state schema is missing or empty')
    for key, row in schema.items():
        if type(key) is not str or not key or type(row) is not dict or set(row) != {'shape', 'dtype'}:
            raise QwenLoadGateError('Malformed expected-state schema row')
        if type(row['shape']) is not list or any(type(n) is not int or n < 0 for n in row['shape']):
            raise QwenLoadGateError('Malformed expected-state shape')
        if type(row['dtype']) is not str or row['dtype'] not in {entry[1] for entry in DTYPES.values()}:
            raise QwenLoadGateError('Unsupported expected-state dtype')
    if digest(schema) != approved_sha256:
        raise QwenLoadGateError('Expected-state schema SHA256 mismatch')
    return json.loads(canonical_bytes(schema))


def validate_aliases(aliases, approved_sha256, schema):
    """Only predeclared omitted aliases, never heuristic prefix or tie discovery.
    The alias digest and its source/config provenance belong in the execution lock.
    """
    if type(aliases) is not list:
        raise QwenLoadGateError('Alias mapping must be a plain list')
    if aliases:
        _sha(approved_sha256, 'approved alias mapping')
        if digest(aliases) != approved_sha256:
            raise QwenLoadGateError('Approved alias mapping SHA256 mismatch')
    elif approved_sha256 not in (None, digest([])):
        raise QwenLoadGateError('Empty alias mapping SHA256 mismatch')
    mapping = {}
    for row in aliases:
        if type(row) is not dict or set(row) != {'alias', 'source'}:
            raise QwenLoadGateError('Malformed alias mapping')
        alias, source = row['alias'], row['source']
        if type(alias) is not str or type(source) is not str:
            raise QwenLoadGateError('Malformed alias/source name')
        if alias == source or alias in mapping or alias not in schema or source not in schema:
            raise QwenLoadGateError('Duplicate, unknown or self-mapped alias')
        if schema[alias] != schema[source]:
            raise QwenLoadGateError('Alias source and destination schemas conflict')
        mapping[alias] = source
    if any(source in mapping for source in mapping.values()):
        raise QwenLoadGateError('Alias chains/cycles are not allowed')
    return mapping


def _converted_chunk(raw, dtype):
    if dtype == 'BF16':
        # Exact little-endian IEEE BF16 to F32 expansion, no rounding.
        if any((u & 0x7f80) == 0x7f80 for (u,) in struct.iter_unpack('<H', raw)):
            raise QwenLoadGateError('Nonfinite BF16 checkpoint tensor')
        expanded = bytearray(2 * len(raw))
        expanded[2::4] = raw[0::2]
        expanded[3::4] = raw[1::2]
        return expanded
    if dtype == 'F32':
        if any((u & 0x7f800000) == 0x7f800000 for (u,) in struct.iter_unpack('<I', raw)):
            raise QwenLoadGateError('Nonfinite F32 checkpoint tensor')
    if dtype == 'BOOL' and any(v not in (0, 1) for v in raw):
        raise QwenLoadGateError('Noncanonical BOOL checkpoint tensor')
    return raw


def inspect_safetensors(path, expected_schema, approved_schema_sha256,
                        expected_file_sha256, *, approved_aliases=None, approved_aliases_sha256=None):
    """Stream complete local header/payload, detect holes/overlap/extra bytes,
    and fingerprint every necessary tensor after the exact approved conversion.
    NO tensor runtime, remote lookup, deserialization, alias or prefix fallback.
    """
    schema = validate_expected_schema(expected_schema, approved_schema_sha256)
    if approved_aliases is None:
        approved_aliases = []
    aliases = validate_aliases(approved_aliases, approved_aliases_sha256, schema)
    _sha(expected_file_sha256, 'checkpoint file')
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise QwenLoadGateError('Checkpoint must be a regular nonsymlink file')
    stat_before = path.stat()
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise QwenLoadGateError('Truncated safetensors prefix')
        header_len = struct.unpack('<Q', prefix)[0]
        if header_len == 0 or header_len > MAX_HEADER_BYTES or header_len % 8:
            raise QwenLoadGateError('Invalid/unapproved safetensors header length')
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len or not header_raw.startswith(b'{'):
            raise QwenLoadGateError('Truncated/malformed safetensors header')
        try:
            header = json.loads(header_raw.decode('utf-8'), object_pairs_hook=_unique_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise QwenLoadGateError('Invalid safetensors JSON') from exc
        if type(header) is not dict:
            raise QwenLoadGateError('Safetensors header must be an object')
        metadata = header.get('__metadata__', {})
        if type(metadata) is not dict or any(type(k) is not str or type(v) is not str for k, v in metadata.items()):
            raise QwenLoadGateError('Malformed safetensors metadata')
        rows = {k: v for k, v in header.items() if k != '__metadata__'}
        if set(rows) != set(schema) - set(aliases):
            raise QwenLoadGateError('Checkpoint keys do not exactly equal expected keys minus approved omitted aliases')
        payload_len = stat_before.st_size - 8 - header_len
        segments = []
        for key, row in rows.items():
            if type(row) is not dict or set(row) != {'dtype', 'shape', 'data_offsets'}:
                raise QwenLoadGateError('Malformed safetensors tensor row: ' + key)
            dtype, shape, offsets = row['dtype'], row['shape'], row['data_offsets']
            if type(dtype) is not str or dtype not in DTYPES:
                raise QwenLoadGateError('Unsupported checkpoint dtype: ' + key)
            if type(shape) is not list or any(type(n) is not int or n < 0 for n in shape):
                raise QwenLoadGateError('Malformed checkpoint shape: ' + key)
            if shape != schema[key]['shape'] or DTYPES[dtype][1] != schema[key]['dtype']:
                raise QwenLoadGateError('Checkpoint shape/dtype does not match expected state: ' + key)
            if type(offsets) is not list or len(offsets) != 2 or any(type(n) is not int for n in offsets):
                raise QwenLoadGateError('Malformed tensor offsets: ' + key)
            begin, end = offsets
            count = 1
            for n in shape:
                count *= n
            if begin < 0 or end < begin or end > payload_len or end - begin != count * DTYPES[dtype][0]:
                raise QwenLoadGateError('Tensor byte extent/shape mismatch: ' + key)
            segments.append((begin, end, key))
        position = 0
        for begin, end, key in sorted(segments):
            if begin != position:
                raise QwenLoadGateError('Safetensors payload contains hole or overlap')
            position = end
        if position != payload_len:
            raise QwenLoadGateError('Safetensors payload has unaccounted bytes')
        whole_hash = hashlib.sha256(prefix + header_raw)
        tensor_evidence = {}
        for begin, end, key in sorted(segments):
            raw_hash, loaded_hash = hashlib.sha256(), hashlib.sha256()
            remaining = end - begin
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk or len(chunk) % DTYPES[rows[key]['dtype']][0]:
                    raise QwenLoadGateError('Truncated tensor payload')
                whole_hash.update(chunk)
                raw_hash.update(chunk)
                loaded_hash.update(_converted_chunk(chunk, rows[key]['dtype']))
                remaining -= len(chunk)
            tensor_evidence[key] = {
                'shape': rows[key]['shape'], 'checkpoint_dtype': rows[key]['dtype'],
                'dtype': schema[key]['dtype'], 'checkpoint_content_sha256': raw_hash.hexdigest(),
                'sha256': loaded_hash.hexdigest(), 'data_offsets': rows[key]['data_offsets'],
                'conversion': 'bf16_to_f32_exact' if rows[key]['dtype'] == 'BF16' else 'identity',
            }
        if handle.read(1):
            raise QwenLoadGateError('Unexpected trailing checkpoint bytes')
    stat_after = path.stat()
    if (stat_before.st_dev, stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_dev, stat_after.st_ino, stat_after.st_size, stat_after.st_mtime_ns):
        raise QwenLoadGateError('Checkpoint changed while inspected')
    if whole_hash.hexdigest() != expected_file_sha256:
        raise QwenLoadGateError('Checkpoint file SHA256 mismatch')
    return {'schema': 'qwen-safetensors-content-evidence-v1',
            'checkpoint_file_sha256': whole_hash.hexdigest(), 'checkpoint_bytes': stat_before.st_size,
            'expected_state_schema_sha256': approved_schema_sha256,
            'expected_state_schema': schema, 'approved_aliases': approved_aliases,
            'approved_aliases_sha256': digest(approved_aliases),
            'header': header, 'header_sha256': hashlib.sha256(header_raw).hexdigest(),
            'tensors': tensor_evidence}


def inspect_ignore_hazards(model):
    """Inspect every runtime module, including inherited attributes. Tied-weight
    declarations are recorded; omitted aliases still fail exact-key comparison.
    """
    evidence = []
    names = ('_keys_to_ignore_on_load_missing', '_keys_to_ignore_on_load_unexpected',
             '_tied_weights_keys', '_dynamic_tied_weights_keys')
    for module_name, module in model.named_modules():
        row = {'module': module_name, 'class': type(module).__module__ + '.' + type(module).__qualname__}
        for name in names:
            value = getattr(module, name, None)
            if value is not None and (type(value) not in (list, tuple) or any(type(v) is not str for v in value)):
                raise QwenLoadGateError('Malformed runtime load-ignore/tie attribute: ' + name)
            row[name] = None if value is None else list(value)
        evidence.append(row)
    if not evidence or evidence[0]['module'] != '':
        raise QwenLoadGateError('Incomplete runtime module hazard inventory')
    for row in evidence:
        if row['_keys_to_ignore_on_load_missing'] or row['_keys_to_ignore_on_load_unexpected']:
            raise QwenLoadGateError('Runtime ignore-key regex is nonempty', evidence={'ignore_hazards': evidence})
    return evidence


def clear_verification(model):
    _forget(model)
    model._kws_qwen_weights_verified = False
    model._kws_qwen_load_receipt = None


def verify_loaded_model(model, loading_info, checkpoint_evidence, describe_tensor, describe_storage=None):
    """Compare every loaded state tensor to full checkpoint-derived evidence.
    describe_tensor must be the reviewed runtime callback, never user-selected
    arbitrary metadata. Its implementation must prove shape/dtype/content/device.
    """
    clear_verification(model)
    full_info = None
    try:
        full_info = validate_loading_info(loading_info)
        hazards = inspect_ignore_hazards(model)
        if type(checkpoint_evidence) is not dict or checkpoint_evidence.get('schema') != 'qwen-safetensors-content-evidence-v1':
            raise QwenLoadGateError('Missing full checkpoint evidence')
        tensors = checkpoint_evidence.get('tensors')
        if type(tensors) is not dict or not tensors:
            raise QwenLoadGateError('Missing necessary checkpoint tensor evidence')
        schema = validate_expected_schema(checkpoint_evidence.get('expected_state_schema'),
                                          checkpoint_evidence.get('expected_state_schema_sha256'))
        aliases = validate_aliases(checkpoint_evidence.get('approved_aliases'),
                                   checkpoint_evidence.get('approved_aliases_sha256'), schema)
        if set(tensors) != set(schema) - set(aliases):
            raise QwenLoadGateError('Checkpoint evidence lacks required tensor coverage')
        state = model.state_dict()
        if set(state) != set(schema):
            raise QwenLoadGateError('Loaded state keys differ from complete expected schema')
        verified = {}
        for key in sorted(state):
            got = describe_tensor(state[key])
            source = aliases.get(key, key)
            wanted = {name: tensors[source][name] for name in ('shape', 'dtype', 'sha256')}
            if type(got) is not dict or set(got) != set(wanted) or got != wanted:
                raise QwenLoadGateError('Loaded state shape/dtype/content mismatch: ' + key)
            verified[key] = got
        alias_proofs = []
        if aliases and not callable(describe_storage):
            raise QwenLoadGateError('Approved aliases require actual shared-storage proof')
        for alias, source in sorted(aliases.items()):
            left, right = describe_storage(state[alias]), describe_storage(state[source])
            required = {'storage_pointer', 'storage_nbytes', 'storage_offset', 'stride', 'device'}
            for item in (left, right):
                if type(item) is not dict or set(item) != required:
                    raise QwenLoadGateError('Malformed alias storage proof')
                if (type(item['storage_pointer']) is not int or item['storage_pointer'] <= 0
                        or type(item['storage_nbytes']) is not int or item['storage_nbytes'] <= 0
                        or type(item['storage_offset']) is not int or item['storage_offset'] < 0
                        or type(item['stride']) is not list
                        or any(type(n) is not int or n < 0 for n in item['stride'])
                        or item['device'] != 'cpu'):
                    raise QwenLoadGateError('Invalid alias storage proof')
            if left != right or len(left['stride']) != len(schema[alias]['shape']):
                raise QwenLoadGateError('Alias lacks identical storage/offset/stride proof')
            alias_proofs.append({'alias': alias, 'source': source, 'shared_storage': left,
                                 'shared_content_sha256': verified[alias]['sha256']})
        receipt = {'schema': 'qwen-complete-load-receipt-v1',
                   'loading_info': full_info, 'checkpoint_evidence': checkpoint_evidence,
                   'runtime_ignore_hazards': hazards, 'verified_loaded_state': verified,
                   'verified_alias_proofs': alias_proofs,
                   'complete_checkpoint_verification_passed': True,
                   'execution_authorized': False}
        # Canonical copy severs mutable caller references before storing evidence.
        receipt = json.loads(canonical_bytes(receipt))
        receipt_sha = digest(receipt)
        _seal(model, receipt_sha)
        model._kws_qwen_load_receipt = receipt
        model._kws_qwen_weights_verified = True
        return json.loads(canonical_bytes(receipt))
    except Exception as exc:
        clear_verification(model)
        if isinstance(exc, QwenLoadGateError):
            raise QwenLoadGateError(str(exc), loading_info_raw=loading_info,
                                   evidence={'loading_info': full_info, 'checkpoint_evidence': checkpoint_evidence,
                                             'failure_evidence': exc.evidence}) from exc
        raise


def require_verified_receipt(model):
    """A boolean alone, copied receipt, altered receipt or another model cannot pass.
    This is an in-process safety interlock, NOT tamper resistance or execution auth.
    Immutable model ownership after verification is a required launcher obligation.
    """
    receipt = getattr(model, '_kws_qwen_load_receipt', None)
    if (getattr(model, '_kws_qwen_weights_verified', False) is not True
            or type(receipt) is not dict or not _same_instance_seal(model)
            or digest(receipt) != _same_instance_seal(model)
            or receipt.get('complete_checkpoint_verification_passed') is not True):
        raise QwenLoadGateError('Complete Qwen checkpoint/content verification receipt required before inference')
    return receipt
