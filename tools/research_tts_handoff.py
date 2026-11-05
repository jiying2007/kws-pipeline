#!/usr/bin/env python3
"""Read the actual ASR handoff using its separately pinned data source tree."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

DATA_SOURCES = ('automation/active_asr_weak_view.py', 'automation/research_weak_sampling.py',
    'kws_data_contract/__init__.py', 'kws_data_contract/identity.py',
    'kws_data_contract/reference_text.py', 'kws_data_contract/source_registry.py',
    'research/auto-tts-voice-clone-20261102/bulk_pool.py')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, 'HANDOFF_REQUEST_BOUND')
    request = json.loads(raw)
    source = Path(request['source_root']).absolute()
    require(not any(p.is_symlink() for p in (source, *source.parents)) and
        set(request['source_files']) == set(DATA_SOURCES), 'HANDOFF_SOURCE_ROSTER')
    require({p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()} == set(DATA_SOURCES) and
        not any(p.is_symlink() for p in source.rglob('*')), 'HANDOFF_EXTRA_SOURCE_OR_ALIAS')
    for name, expected in request['source_files'].items():
        path = source / name
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'HANDOFF_SOURCE_ALIAS')
        body = path.read_bytes()
        require(len(body) == expected['bytes'] and hashlib.sha256(body).hexdigest() == expected['sha256'],
            'HANDOFF_SOURCE_CHANGED')
    sys.path.insert(0, str(source))
    from automation import active_asr_weak_view as data
    require(Path(data.__file__).resolve() == source / 'automation/active_asr_weak_view.py', 'HANDOFF_IMPORT_SOURCE')
    require(request['expected_count'] == 232, 'HANDOFF_EXPECTED232')
    require(callable(data.verify_accounted_seal), 'HANDOFF_VERIFIER_ENTRYPOINT')
    sealed = request.get('sealed_stage')
    if request['action'] == 'verify':
        path = Path(sealed['path']).absolute()
        require(not any(p.is_symlink() for p in (path, *path.parents)) and path.stat().st_size == sealed['bytes'] and
            hashlib.sha256(path.read_bytes()).hexdigest() == sealed['sha256'], 'HANDOFF_SEALED_FILE_CHANGED')
        result = data.verify_accounted_seal(path, sealed['sha256'], expected_count=232)
        require(type(result) is dict and result.get('schema') == 'd1-accounted-selective-seal-verification-v3' and
            result.get('status') == 'verified' and result.get('verified') is True and
            result.get('accounted_prefix_count') == 232 and result.get('new_model_calls') == 0 and
            result.get('ml_imports') == 0 and result.get('full_replay_completed') is True,
            'HANDOFF_VERIFIER_EXPLICIT_SUCCESS_REQUIRED')
    else:
        require(request['action'] == 'preflight' and sealed is None, 'HANDOFF_ACTION_SCOPE')
    require(not any(name in sys.modules for name in ('torch', 'numpy', 'onnxruntime', 'qwen_tts', 'scipy', 'soundfile')),
        'HANDOFF_ML_IMPORT')
    # The full verifier raises on any mismatched source/native/queue link. This
    # stable receipt describes that executed check without importing its module
    # globals into a pipeline interpreter or fabricating a future completion.
    print(json.dumps({'schema': 'active-tts-asr232-data-verification-v2',
        'entrypoint': 'automation.active_asr_weak_view.verify_accounted_seal',
        'source_root': str(source), 'source_files': request['source_files'], 'sealed_stage': sealed,
        'expected_count': 232, 'verification_performed': request['action'] == 'verify',
        'verified': request['action'] == 'verify', 'source_preflight_passed': True,
        'ml_imports': 0, 'new_model_calls': 0,
        'pid': os.getpid(), 'pgid': os.getpgrp()}, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
