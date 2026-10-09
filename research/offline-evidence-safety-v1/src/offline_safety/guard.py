"""Saved-only evidence admission; never starts a process, model or native library."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

POLICY = {
    'schema': 'offline-evidence-policy-v1',
    'scope': 'SAVED_TRACE_REVIEW_ONLY',
    'profile': 'same-input',
    'stage_atol': 1e-4, 'stage_rtol': 1e-5,
    'cmvn': 'bitexact', 'optimized_python': 'deny',
    'old_attempt_retry_authorized': False,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require_unoptimized():
    if sys.flags.optimize != 0:
        raise PermissionError('optimized Python is not admitted')


def load_public_admission(path, expected_sha256):
    """Load a caller-pinned, trusted public helper. This is not a sandbox."""
    require_unoptimized()
    path = Path(path)
    if sha(path) != expected_sha256:
        raise PermissionError('public admission source drift')
    spec = importlib.util.spec_from_file_location('_pinned_public_admission', path)
    module = importlib.util.module_from_spec(spec)
    # Compile the verified bytes directly: do not use a stale bytecode cache.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError('public admission source changed while loading')
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def approval_binding(helper_path, ticket, identity):
    """Public integrity schema, not an authorization issuer or digital signature."""
    return {
        'schema': 'offline-review-approval-v1',
        'scope': POLICY['scope'],
        'policy_sha256': digest(POLICY),
        'guard_sha256': sha(__file__),
        'helper_sha256': sha(helper_path),
        'ticket_sha256': digest(ticket),
        'identity_sha256': digest(identity),
    }


def _snapshot_arrays(path, arrays):
    # No shape/finite validation before persistence; object arrays are refused
    # because their serialization could invoke arbitrary pickle behavior.
    if not isinstance(arrays, dict) or any(
        not isinstance(k, str) or not isinstance(v, np.ndarray) or v.dtype.hasobject
        for k, v in arrays.items()
    ):
        raise TypeError('raw evidence requires string-keyed non-object ndarrays')
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise IOError('raw evidence member mismatch')
        for key, value in arrays.items():
            copy = saved[key]
            if copy.dtype != value.dtype or copy.shape != value.shape or copy.tobytes() != value.tobytes():
                raise IOError('raw evidence readback mismatch')


def save_then_compare(destination, native, reference, identity, *, helper_path,
                      helper_sha256, ticket, work, attempt, approval=None, armed=False):
    """Persist even malformed numerical traces; re-audit before any comparison.

    Approval is an externally reviewed exact binding, not model-run permission.
    Missing/incorrect approval defaults to refusal. No historical file is edited.
    """
    dest = Path(destination)
    dest.mkdir(exist_ok=False)
    write_new(dest / 'STARTED.json', {'complete': False, 'qualified': False,
                                    'scope': POLICY['scope'], 'model_calls': 0})
    sync_directory(dest)
    sync_directory(dest.parent)
    raw_complete = False
    try:
        _snapshot_arrays(dest / 'native-raw.npz', native)
        _snapshot_arrays(dest / 'reference-raw.npz', reference)
        write_new(dest / 'identity.json', identity)
        raw_receipt = {
            'complete': True, 'qualified': False,
            'native_sha256': sha(dest / 'native-raw.npz'),
            'reference_sha256': sha(dest / 'reference-raw.npz'),
            'identity_sha256': sha(dest / 'identity.json'),
        }
        write_new(dest / 'RAW-SAVED.json', raw_receipt)
        receipt_hash = sha(dest / 'RAW-SAVED.json')
        sync_directory(dest)
        raw_complete = True
        def read_bound(name, expected):
            nonlocal raw_complete
            try:
                raw = (dest / name).read_bytes()
            except OSError:
                raw_complete = False
                raise
            if hashlib.sha256(raw).hexdigest() != expected:
                raw_complete = False
                raise IOError('saved evidence drift: ' + name)
            return raw

        identity = json.loads(read_bound('identity.json', raw_receipt['identity_sha256']))
        require_unoptimized()
        if armed is not True:
            raise PermissionError('default disarmed; explicit saved-only review approval required')
        if approval is None or approval != approval_binding(helper_path, ticket, identity):
            raise PermissionError('missing, wrong or stale saved-only approval')
        helper = load_public_admission(helper_path, helper_sha256)
        # The old standalone precondition is now on the actual saved-review path.
        current = helper.require_admission(ticket, work, attempt)
        if identity.get('source_sha256') != current['bindings']['source_freeze_sha256']:
            raise PermissionError('identity source binding mismatch')
        if identity.get('dependency_sha256') != current['bindings']['dependency_manifest_sha256']:
            raise PermissionError('identity dependency binding mismatch')
        if identity.get('admission_ticket_sha256') != digest(ticket):
            raise PermissionError('identity ticket binding mismatch')
        if identity.get('arm') != current['arm']:
            raise PermissionError('identity arm mismatch')
        records = json.loads((Path(attempt) / current['arm'] / 'REFERENCE-FROZEN.json').read_text())['records']
        matched = [r for r in records if r['name'] == identity.get('fixture')]
        if len(matched) != 1:
            raise PermissionError('fixture is not in the admitted reference')
        record = matched[0]
        admitted_path = helper.contained(Path(attempt) / current['arm'], record['path'])
        admitted_reference = helper.load_trace(admitted_path, record['rows'])
        if identity.get('reference_sha256') != helper.trace_digest(admitted_reference):
            raise PermissionError('reference is not the admitted fixture')
        # Validate snapshots, never mutable caller arrays after the persistence step.
        native_bytes = read_bound('native-raw.npz', raw_receipt['native_sha256'])
        reference_bytes = read_bound('reference-raw.npz', raw_receipt['reference_sha256'])
        with np.load(io.BytesIO(native_bytes), allow_pickle=False) as data:
            native_saved = {k: data[k] for k in data.files}
        with np.load(io.BytesIO(reference_bytes), allow_pickle=False) as data:
            reference_saved = {k: data[k] for k in data.files}
        result = helper.save_then_compare(dest / 'validated', native_saved, reference_saved, identity)
        # Detect ordinary drift during the comparison as well as before it.
        helper.require_admission(ticket, work, attempt)
        if approval != approval_binding(helper_path, ticket, identity):
            raise PermissionError('review binding changed during comparison')
        for name, expected in (('native-raw.npz', raw_receipt['native_sha256']),
                               ('reference-raw.npz', raw_receipt['reference_sha256']),
                               ('identity.json', raw_receipt['identity_sha256']),
                               ('RAW-SAVED.json', receipt_hash)):
            read_bound(name, expected)
        terminal = {'complete': True, 'qualified': bool(result['pass_gate']),
                    'status': 'PASS' if result['pass_gate'] else 'FAIL',
                    'policy_sha256': digest(POLICY), 'approval_sha256': digest(approval),
                    'model_calls': 0, 'old_attempt_retry_authorized': False}
        # Publish only bytes that have successfully flushed and fsynced.
        write_new(dest / 'TERMINAL.pending.json', terminal)
        os.link(dest / 'TERMINAL.pending.json', dest / 'TERMINAL.json')
        (dest / 'TERMINAL.pending.json').unlink()
        sync_directory(dest)
        return terminal
    except BaseException as exc:
        # If publication succeeded but the final directory fsync failed, do not
        # overwrite its valid numerical verdict or mask the original I/O error.
        # Durability is only confirmed by successful return from this function.
        if (dest / 'TERMINAL.json').exists():
            raise
        # A partial or invalid trace never acquires a qualifying terminal receipt.
        write_new(dest / 'TERMINAL.json', {
            'complete': raw_complete, 'qualified': False, 'status': 'REJECTED',
            'error_type': type(exc).__name__, 'error': str(exc), 'model_calls': 0,
            'old_attempt_retry_authorized': False,
        })
        sync_directory(dest)
        raise
