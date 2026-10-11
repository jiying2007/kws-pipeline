"""Source-only compatibility policy. Import performs no probes or networking.

The updated source worker obtains version/help with its exact interpreter and
passes the resolved pip package root in the pinned immutable runtime image.
Integration changes source only; the historical executed commit and third freeze
remain unchanged. No activation is wired or runtime authorization granted.
"""
import hashlib
from pathlib import Path
import re
import stat

UPSTREAM_COMMIT = 'bc7c88cb3de9c9af769c51517833ea48bbe70d9a'
AUDITED_VERSION = '25.0.1'
OPTIONS = ('--resume-retries', '--retries', '--require-hashes', '--no-deps',
           '--no-index', '--only-binary', '--keyring-provider')
# Exact upstream bytes; populated from reviewed pypa/pip tag 25.0.1.
SOURCE_SHA256 = {'_internal/cli/cmdoptions.py': '577041db6178feffd1907699e689d69ecce16c1b63619bcd85b9fd33434ed072',
 '_internal/cli/index_command.py': '8bfb203653e65c2e621d4698f9d9a6ac728a4e07393b8856ce2b2be40975aebd',
 '_internal/commands/download.py': 'd2a0749f2b3a6443eca20e39d650ec8cbe41c7b67deedf81f34a0564a869cca3',
 '_internal/network/auth.py': '0f88004a352baa80c5952b7a810efaeca0008efe8f532254d29b839615cd5511',
 '_internal/network/download.py': '14b38fdbd74f6040818808bb7848ef01b364cb368a36a6f28ce4f69bc1cf5bc5',
 '_internal/network/session.py': '9ac3387acd7a2e698d118364ad8ca0f1f4dcee00076ca165b5ac1f28fdbb2ce2',
 '_internal/network/utils.py': '2276b17a5f8dc41bb83d05a48f212b7677dec2c1427201e987b773475f856e86',
 '_vendor/requests/adapters.py': '27b55e571281bdac1bb655f60c4455a34e49f415d371660b30735dd4169af9b9',
 '_vendor/requests/sessions.py': 'ca44c8f145864a5b4e7c7d3b1caa25947ee44c11b0e168620556901a67244f0e',
 '_vendor/urllib3/connectionpool.py': '7b67a203035b14d08ac63e1bc0328d2bec3b1c8752cf73a633153f4c8b7e7af4',
 '_vendor/urllib3/util/retry.py': 'e8436f399f0f043ce1f24822c69aa5f6522b6f67711fe93b66605a9c9176360e'}


class CompatibilityFailure(ValueError):
    def __init__(self, evidence):
        super().__init__('pip_capability_probe')
        self.evidence = evidence


def _source_matches(pip_root):
    """Fail closed, without returning paths, bytes, exception strings or hashes."""
    try:
        root = Path(pip_root)
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            return False
        for relative, digest in SOURCE_SHA256.items():
            path = root / relative
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or path.resolve() != path or
                    not 0 < info.st_size <= 1024 * 1024):
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def _has_option(help_text, option):
    # Match complete option tokens; '--retries-extra' must not satisfy retries.
    return re.search(rb'(?<![A-Za-z0-9_-])' + re.escape(option.encode('ascii')) +
                     rb'(?![A-Za-z0-9_-])', help_text) is not None


def select_contract(version, help_exit_code, help_text, pip_root=None):
    """Return fixed retry argv and sanitized probe evidence, or fail closed.

    Unknown versions never get the 25.0.1 exemption. Their existing contract
    continues to require explicit --resume-retries 0 and all seven options.
    No promise of broader version compatibility is made.
    """
    valid_version = (type(version) is str and
                     re.fullmatch(r'[0-9]{1,6}\.[0-9]{1,6}(?:\.[0-9]{1,6})?', version))
    valid_code = type(help_exit_code) is int and -255 <= help_exit_code <= 255
    valid_help = type(help_text) is bytes and len(help_text) <= 128 * 1024
    audited = valid_version and version == AUDITED_VERSION
    required = OPTIONS[1:] if audited else OPTIONS
    missing = [flag for flag in OPTIONS if not valid_help or not _has_option(help_text, flag)]
    required_missing = [flag for flag in missing if flag in required]
    source_ok = _source_matches(pip_root) if audited else None
    evidence = {
        'schema': 'bounded-pip-capability-v1',
        'pip_version': version if valid_version else None,
        'help_exit_code': help_exit_code if valid_code else None,
        'missing_options': missing,
        'policy': 'audited-25.0.1-no-resume' if audited else 'explicit-zero-resume',
        'reviewed_source_matches': source_ok,
    }
    if not (valid_version and valid_code and valid_help and help_exit_code == 0 and
            not required_missing and (not audited or source_ok)):
        raise CompatibilityFailure(evidence)
    retry_argv = ['--retries', '0']
    if not audited:
        retry_argv += ['--resume-retries', '0']
    return {'retry_argv': retry_argv, 'evidence': evidence}
