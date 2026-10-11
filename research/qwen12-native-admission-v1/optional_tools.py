"""Declare the unused external FFmpeg backend unavailable during Sense import.

This is an explicit capture execution policy, not an installation probe result.
The exact original dependency remains on disk and in the native source lock.
Only its locked import-time version probe is suppressed; all other subprocess
calls retain their original behavior under the already-installed process guard.
"""
from contextlib import contextmanager
from pathlib import Path
import errno
import subprocess
import sys
from types import CodeType

from campaign_contract import digest, require
from native_locks import read

SOURCE = 'funasr/utils/load_utils.py'
SOURCE_SHA = '55731d913c76de3959be6c6da523dff8ce73f8a7329dd20d97aa5aba8fd4092d'
MODULE = 'funasr.utils.load_utils'
POLICY = 'sensevoice-unused-external-ffmpeg-unavailable-v1'
MAX_PROBES = 32


def policy_config():
    return {'schema': 'native-optional-tool-policy-v1', 'policy': POLICY,
        'source': SOURCE, 'source_sha256': SOURCE_SHA, 'external_backend': 'ffmpeg',
        'availability': 'disabled_by_capture_policy', 'actual_installed_status': 'unknown',
        'external_process_executed': False, 'maximum_import_probes': MAX_PROBES}


@contextmanager
def unavailable_ffmpeg(runtime, guard, emit, *, register_code=None):
    """A scoped, exact-callsite policy; never suppress arbitrary PermissionError."""
    require(type(guard) is dict and guard.get('schema') == 'qwen12-inference-process-guard-v1' and
        guard.get('family') == 'sensevoice' and guard.get('fork_vfork_nonthread_clone') == 'EPERM',
        'installed Sense process guard receipt required')
    require(MODULE not in sys.modules and 'funasr' not in sys.modules, 'fresh Sense import required')
    source = Path(runtime) / 'venv/lib/python3.12/site-packages' / SOURCE
    raw = read(source, 65536)
    require(digest(raw) == SOURCE_SHA, 'locked optional-probe source drift')
    codes = [code for code in compile(raw, str(source), 'exec', dont_inherit=True,
        optimize=sys.flags.optimize).co_consts if isinstance(code, CodeType) and
        code.co_name == 'is_ffmpeg_installed']
    require(len(codes) == 1, 'single locked optional-probe function')
    expected_code = codes[0]
    if register_code is not None:
        try: register_code(expected_code)
        except BaseException: pass  # Diagnostic registration is best-effort only.
    original = subprocess.check_output
    count = 0
    completed = False
    enabled = None
    failed = False

    def check_output(*args, **kwargs):
        nonlocal count
        caller = sys._getframe(1)
        exact = (caller.f_code.co_filename == str(source) and
                 caller.f_code == expected_code and caller.f_lineno == 33 and
                 caller.f_globals.get('__name__') == MODULE and
                 caller.f_globals.get('__file__') == str(source) and
                 getattr(caller.f_globals.get('__spec__'), 'origin', None) == str(source))
        if not exact:
            return original(*args, **kwargs)
        require(len(args) == 1 and type(args[0]) is list and args[0] == ['ffmpeg', '-version'] and
            set(kwargs) == {'stderr'} and type(kwargs['stderr']) is int and
            kwargs['stderr'] == subprocess.STDOUT, 'locked optional-probe invocation drift')
        require(count < MAX_PROBES, 'bounded optional-probe invocations')
        count += 1
        # The dependency's existing unavailable branch handles this exact type.
        # This is not a report that a real subprocess returned ENOENT.
        raise FileNotFoundError(errno.ENOENT, 'external backend disabled by capture policy')

    subprocess.check_output = check_output
    try:
        yield
        module = sys.modules.get(MODULE)
        require(module is not None and getattr(module, 'use_ffmpeg', None) is False,
            'Sense external backend must remain unavailable')
        enabled = False
        require(1 <= count <= MAX_PROBES, 'locked optional probe must have been observed')
        completed = True
    except BaseException:
        failed = True
        raise
    finally:
        subprocess.check_output = original
        receipt = {**policy_config(), 'schema': 'native-optional-tool-probe-receipt-v1',
            'intercepted_probes': count, 'module_backend_enabled': enabled, 'completed': completed}
        if failed:
            try: emit(receipt)
            except BaseException: pass  # Never replace the actual loader failure.
        else:
            emit(receipt)  # Missing evidence after successful import fails closed.
