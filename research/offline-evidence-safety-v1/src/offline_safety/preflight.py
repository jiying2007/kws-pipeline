"""Explicit reusable pre-launch checks. No process launcher or authorization issuer."""
from pathlib import Path
from .guard import digest, require_unoptimized, sha

POLICY = {
    'schema': 'synthetic-preflight-policy-v1',
    'scope': 'NEW_OFFLINE_SYNTHETIC_PROCESS_ONLY_V1',
    'model_execution_authorized': False,
    'old_once_authorization_accepted': False,
    'optimized_python': 'deny',
}


def _files(paths):
    if not isinstance(paths, dict) or not paths:
        raise ValueError('nonempty named file mapping required')
    result = {}
    for name, path in paths.items():
        if not isinstance(name, str) or not name:
            raise ValueError('nonempty file identity required')
        path = Path(path)
        result[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result


def preflight_binding(*, source_files, input_files):
    """Calculate the exact proposed binding; this does NOT approve or launch it."""
    return {
        'schema': 'new-synthetic-preflight-approval-v1',
        'approved': True,
        'scope': POLICY['scope'],
        'policy_sha256': digest(POLICY),
        'preflight_sha256': sha(__file__),
        'shared_guard_sha256': sha(Path(__file__).with_name('guard.py')),
        'source_files': _files(source_files),
        'input_files': _files(input_files),
    }


def require_preflight(*, source_files, input_files, approval=None, armed=False):
    """Recompute exact new-scope bindings immediately before a caller's launch.

    Approval must come from the caller's separately authorized workflow. The
    result is integrity admission for a synthetic fixture, never model permission
    or a reusable execution ticket. This function neither launches nor consumes
    any historical once permission. Caller owns isolation and immutable inputs.
    """
    require_unoptimized()
    if armed is not True:
        raise PermissionError('preflight is disarmed')
    if approval is None:
        raise PermissionError('new exact synthetic preflight approval required')
    current = preflight_binding(source_files=source_files, input_files=input_files)
    if type(approval) is not dict or digest(approval) != digest(current):
        raise PermissionError('wrong scope, policy, source, input or verifier binding')
    return {'admitted': True, 'scope': POLICY['scope'],
            'binding_sha256': digest(current), 'model_execution_authorized': False}
