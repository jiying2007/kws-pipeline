"""One fresh-process primary run with durable per-clip evidence, no downloader.

The first clip is the canary and is never repeated in this run. A failed load
or decode ends the run; every remaining clip is explicitly not_run. The outer
supervisor owns hard deadlines/resource termination. This module's SIGALRM is
only a cooperative per-clip deadline, not a guarantee against a blocked C call.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from .adapters import (QWEN, SENSE, validate_decoder_manifest,
    load_qwen_after_approval, load_sense_after_approval,
    infer_qwen_once_after_approval, infer_sense_once_after_approval)
from .architecture import canonical_sha
from .assets import MODELS
from pcm.pcm_binding import WaveExpectation, bind_wave


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def retain(directory, name, value):
    """Exclusive write; every receipt refers to exact retained bytes."""
    raw = json_bytes(value)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Execution sidecar exceeds bounded size')
    artifact_root = Path(directory).parent
    retained = sum(p.stat().st_size for p in artifact_root.rglob('*.json') if p.is_file())
    if retained + len(raw) > 19 * 1024 * 1024:
        raise ValueError('Raw evidence cap; reserve final artifact metadata space')
    path = Path(directory) / name
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def require_offline_flags():
    expected = {'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
        'HF_DATASETS_OFFLINE':'1', 'HF_HUB_DISABLE_TELEMETRY':'1','ORT_DISABLE_TELEMETRY':'1',
        'CUDA_VISIBLE_DEVICES':''}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Required offline/CPU environment flags missing')
    return {'configured_flags': expected, 'kernel_network_isolated': False,
        'limits': 'Verified local assets and local-files-only loaders; this is not kernel network isolation'}


class ClipDeadline(TimeoutError):
    pass


def _alarm(_signum, _frame):
    raise ClipDeadline('Cooperative per-clip deadline reached')


def execute_primary(plan, directory, decoder_bytes, *, loader=None, infer=None,
                    binder=bind_wave, check_environment=True):
    """Plan carries no human labels, intended text or named source waveforms."""
    fields = {'schema','model_id','model_root','pcm_root','asset_lock',
        'asset_lock_sha256','source_lock','source_lock_sha256','contract',
        'contract_sha256','decoder_manifest_sha256','clip_seconds'}
    if type(plan) is not dict or set(plan) != fields or plan['schema'] != 'asr6-run-plan-v1':
        raise ValueError('Malformed primary plan')
    model_id = plan['model_id']
    if model_id not in (QWEN, SENSE):
        raise ValueError('Unsupported model')
    if type(plan['clip_seconds']) is not int or plan['clip_seconds'] != 120:
        raise ValueError('Invalid per-clip deadline')
    if canonical_sha(plan['contract']) != plan['contract_sha256']:
        raise ValueError('Execution contract bytes changed')
    if plan['contract']['model_id'] != model_id or plan['contract']['decoder_manifest_sha256'] != plan['decoder_manifest_sha256']:
        raise ValueError('Plan contract association mismatch')
    inputs = validate_decoder_manifest(decoder_bytes, plan['decoder_manifest_sha256'])
    manifest = json.loads(decoder_bytes)
    if (len(inputs) != plan["contract"]["candidate_count"] or manifest["experiment_id"] != plan["contract"]["experiment_id"] or
        manifest['source_manifest_sha256']!=plan['contract']['source_manifest_sha256'] or manifest['input_freeze_sha256']!=plan['contract']['input_freeze_sha256']):
        raise ValueError("Candidate plan/manifest count or experiment mismatch")
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    retain(directory, 'plan.json', {'schema':'asr6-public-plan-reference-v1',
        'runtime_plan_sha256':canonical_sha(plan), **{k:plan[k] for k in
        ('model_id','asset_lock_sha256','source_lock_sha256','decoder_manifest_sha256','clip_seconds')}})
    contract_sha = retain(directory, 'contract.json', plan['contract'])
    if contract_sha != plan['contract_sha256']:
        raise ValueError('Retained contract differs')
    with (directory / 'decoder-inputs.json').open('xb') as stream:
        stream.write(decoder_bytes)
    environment = require_offline_flags() if check_environment else {'test_only': True}
    retain(directory, 'environment.json', environment)
    model = {'model_id': model_id, 'revision': MODELS[model_id][0],
             'run_id': plan['contract']['run_id']}
    outcomes = [{'opaque_id': oid, 'wav_sha256': row['descriptor']['wav']['sha256'],
        'status':'not_run', 'raw_text':None, 'completeness':'unknown',
        'quality_flags':[], 'execution_receipt_sha256':None} for oid, row in inputs.items()]
    retain(directory, 'outcomes.initial.json', outcomes)
    started = time.monotonic()
    loader = loader or (load_qwen_after_approval if model_id == QWEN else load_sense_after_approval)
    infer = infer or (infer_qwen_once_after_approval if model_id == QWEN else infer_sense_once_after_approval)
    retain(directory, 'model-load-started.json', {'model':model,
        'execution_contract_sha256':contract_sha, 'process_id':os.getpid()})
    try:
        runtime = loader(plan['model_root'], plan['asset_lock'], plan['asset_lock_sha256'],
            plan['source_lock'], plan['source_lock_sha256'], plan['contract'],
            plan['contract_sha256'], decoder_bytes)
        retain(directory, 'model-load.json', runtime.receipt)
    except Exception as exc:
        retain(directory, 'load-failure.json', {'exception_type':type(exc).__name__,
            'detail':'Stage failed; detailed exception text is not public', 'no_model_decode_attempted':True})
        retain(directory, 'outcomes.json', outcomes)
        summary = {'status':'load_failed', 'model':model, 'attempted':0, 'success':0,
                   'not_run':len(inputs), 'wall_seconds':time.monotonic()-started}
        retain(directory, 'summary.json', summary)
        return summary
    attempted = 0
    input_failed = False
    for index, outcome in enumerate(outcomes):
        oid = outcome['opaque_id']
        descriptor = inputs[oid]['descriptor']
        # Only opaque, deterministic basenames enter this runtime.
        expectation = WaveExpectation(oid, descriptor['wav']['sha256'] + '.wav', descriptor['wav']['sha256'],
            descriptor['wav']['bytes'], descriptor['raw_pcm16']['frame_count'])
        try:
            bound = binder(plan['pcm_root'], expectation)
            if bound.descriptor != descriptor or bound.descriptor_sha256 != inputs[oid]['binding_sha256']:
                raise ValueError('Staged PCM differs from exact decoder input')
        except Exception as exc:
            retain(directory, oid + '.input-failure.json', {'exception_type':type(exc).__name__,
                'detail':'Stage failed; detailed exception text is not public', 'decode_attempted':False})
            outcome.update(status='input_error')
            input_failed = True
            break
        attempted += 1
        retain(directory, oid + '.started.json', {'schema':'asr6-clip-started-v1',
            'model':model,'opaque_id':oid,'wav_sha256':outcome['wav_sha256'],
            'input_binding_sha256':inputs[oid]['binding_sha256'],
            'decoder_manifest_sha256':plan['decoder_manifest_sha256'],
            'execution_contract_sha256':contract_sha,'process_id':os.getpid()})
        previous = signal.signal(signal.SIGALRM, _alarm)
        began = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, plan['clip_seconds'])
        try:
            row, evidence = infer(runtime, bound, plan['decoder_manifest_sha256'])
            if type(row) is not dict or set(row) != {'raw_text','completeness','quality_flags'}:
                raise RuntimeError('Malformed adapter outcome')
            outcome.update(row, status='success')
            forward_completed = True
        except Exception as exc:
            outcome.update(status='timeout' if isinstance(exc, ClipDeadline) else 'error',
                raw_text=None, completeness='unknown', quality_flags=[])
            forward_completed = False
            evidence = {'schema':'asr-decoder-failure-evidence-v1',
                'exception_type':type(exc).__name__, 'detail':'Stage failed; detailed exception text is not public',
                'forward_completion_confirmed':False}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        evidence = {'schema':'asr-decoder-evidence-sidecar-v1', 'model':model,
            'opaque_id':oid, 'wav_sha256':outcome['wav_sha256'],
            'input_binding_sha256':inputs[oid]['binding_sha256'],
            'outcome':{k:outcome[k] for k in ('status','raw_text','completeness','quality_flags')},
            'wall_seconds':time.monotonic()-began, 'evidence':evidence}
        evidence_sha = retain(directory, oid + '.decoder.json', evidence)
        receipt = {k:outcome[k] for k in ('opaque_id','wav_sha256','status','raw_text','completeness','quality_flags')}
        receipt.update(schema='asr6-clip-execution-receipt-v1', model=model,
            input_binding_sha256=inputs[oid]['binding_sha256'],
            decoder_manifest_sha256=plan['decoder_manifest_sha256'], attempt_started=True,
            model_forward_completed=forward_completed, decoder_evidence_sha256=evidence_sha,
            execution_contract_sha256=contract_sha)
        outcome['execution_receipt_sha256'] = retain(directory, oid + '.receipt.json', receipt)
        retain(directory, 'outcomes.' + str(index+1).zfill(4) + '.json', outcomes)
        print(json.dumps({'event':'primary_clip_retained','model_id':model_id,
            'opaque_id':oid,'status':outcome['status'],'canary_in_primary_run':index==0}), flush=True)
        if outcome['status'] != 'success':
            break
    retain(directory, 'outcomes.json', outcomes)
    summary = {'status':'input_failed' if input_failed else ('complete' if attempted == len(inputs) and all(r['status']=='success' for r in outcomes) else 'decode_stopped'),
        'model':model, 'attempted':attempted, 'success':sum(r['status']=='success' for r in outcomes),
        'not_run':sum(r['status']=='not_run' for r in outcomes), 'wall_seconds':time.monotonic()-started,
        'first_clip_is_primary_canary':True, 'retries':0, 'outcome_driven_repair':False}
    retain(directory, 'summary.json', summary)
    return summary
