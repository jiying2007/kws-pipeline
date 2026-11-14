#!/usr/bin/env python3
"""Public-runner-only fixed saved-weight Python/native C comparison; never train."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import contract as c
from export_weights import export, source_result

FLAGS = ['-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
         '-Wall', '-Wextra', '-Werror', '-pedantic']


def trace_dtype():
    import numpy as np
    dtype = np.dtype([('stream', '<u4'), ('available_sample', '<u8'),
                      ('features', '<f4', (32,)), ('logits', '<f4', (3,)),
                      ('probabilities', '<f4', (3,)), ('proposal', '<u4')])
    c.require(dtype.itemsize == 168, 'C_REPLAY_TRACE_LAYOUT')
    return dtype


def read_trace(path):
    import numpy as np
    raw = path.read_bytes()
    c.require(raw[:8] == b'EN63TRC1' and raw[8:16] == struct.pack('<II', 1, 168)
              and (len(raw)-16) % 168 == 0, 'C_REPLAY_TRACE_FORMAT')
    return np.frombuffer(raw, dtype=trace_dtype(), offset=16).copy()


def reference(model, kind, data, continuous, online):
    """Exactly one saved-model forward per real frame of this sequence."""
    import numpy as np
    import torch
    records = []
    if continuous:
        sequences = [[data['sources'][cell][2] for cell in sorted(data['sources'])]]
    else:
        sequences = [[data['sources'][cell][2]] for cell in sorted(data['sources'])]
    for stream, arrays in enumerate(sequences):
        framer = online.contract.BoundedFrames()
        encoder = online.IncrementalModel(model, kind)
        decoder = online.contract.ProposalDecoder()
        for samples in arrays:
            for start, end in online.chunk_plan(len(samples), 160):
                for frame, values in framer.push(samples[start:end]):
                    logmel = online.pcm.logmel(np.asarray([values], dtype=np.float32), data['bank'])[0]
                    feature = ((logmel-data['mean'])/data['std']).astype(np.float32)
                    logits = encoder.consume(feature)
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                    event = decoder.consume(probabilities.tolist())
                    records.append((stream, 400+(frame-1)*160, feature, logits.cpu().numpy(),
                                    probabilities, 0 if event is None else event['proposal_class']))
        before = (framer.frames, len(records), decoder.frame)
        c.require(framer.finish() == encoder.frames and before == (framer.frames, len(records), decoder.frame),
                  'C_REFERENCE_EOF_CHANGED')
    return np.array(records, dtype=trace_dtype())


def event_signature(trace):
    return [(int(x['stream']), int(x['available_sample']), int(x['proposal']))
            for x in trace if x['proposal']]


def compare(reference_rows, actual):
    import numpy as np
    c.require(reference_rows.shape == actual.shape, 'C_REPLAY_FRAME_COUNT')
    for field in ('stream', 'available_sample'):
        c.require(np.array_equal(reference_rows[field], actual[field]), 'C_REPLAY_REAL_CLOCK')
    errors = {}
    for field in ('features', 'logits', 'probabilities'):
        c.require(bool(np.isfinite(actual[field]).all()) and bool(np.isfinite(reference_rows[field]).all()),
                  'C_REPLAY_NONFINITE_TRACE')
        error = float(np.max(np.abs(reference_rows[field].astype(np.float64)-actual[field])))
        c.require(error <= c.TOLERANCES[field], 'C_REPLAY_NUMERIC:'+field+':'+str(error))
        errors[field] = error
    c.require(event_signature(reference_rows) == event_signature(actual), 'C_REPLAY_EVENT_CLOCK_OR_CLASS')
    return errors


def verify_original_proposals(refs, kind, source, data, trained):
    """Frozen event class/sample identities; only scores use the fixed FP32 bound."""
    prior = [json.loads(line) for line in (trained/'model/events.jsonl').read_text().splitlines()]
    source_order = sorted(data['sources'])
    observations = next(x['observations'] for x in source['models'] if x['kind'] == kind)
    seen = []
    hashes = {}
    for mode, rows in refs.items():
        streams = range(63) if mode == 'isolated' else range(1)
        for stream in streams:
            selected = rows[rows['stream'] == stream]
            digest = c.digest(selected['probabilities'].astype('<f4').tobytes())
            if mode == 'continuous':
                expected = observations['continuous_trace_sha256']
            else:
                expected = observations['isolated_clips'][stream]['trace_sha256']
            hashes[mode+':'+str(stream)] = {'source_trace_sha256': expected,
                                         'reference_trace_sha256': digest,
                                         'bit_identical_to_prior_runner': digest == expected}
        for row in rows:
            if row['proposal']:
                cls = int(row['proposal'])
                seen.append({'kind': kind,
                             'stream': 'isolated_original_clip' if mode == 'isolated' else 'artificial_sorted_source_concatenation',
                             'cell_id': source_order[int(row['stream'])] if mode == 'isolated' else None,
                             'proposal_class': cls, 'available_after_sample': int(row['available_sample']),
                             'probability': float(row['probabilities'][cls])})
    old = [x for x in prior if x['kind'] == kind]
    keys = ('kind', 'stream', 'cell_id', 'proposal_class', 'available_after_sample')
    c.require(len(old) == len(seen), 'C_REFERENCE_ORIGINAL_EVENT_COUNT')
    for a, b in zip(old, seen):
        c.require(all(a[k] == b[k] for k in keys)
                  and abs(a['probability']-b['probability']) <= c.TOLERANCES['probabilities'],
                  'C_REFERENCE_ORIGINAL_PROPOSALS_CHANGED')
    return hashes


def native(executable, kind, mode, plan, manifest, trace):
    command = [str(executable), '--supervise', kind, mode, plan, str(manifest), str(trace)]
    run = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
    c.require(run.returncode == 0, 'C_REPLAY_NATIVE_FAILED:'+run.stderr[-4000:])
    result = json.loads(run.stdout)
    c.require(result['kind'] == kind and result['mode'] == mode and result['chunk_plan'] == plan
              and result['pcm_samples'] == 2101760 and result['frames'] == (13010 if mode == 'isolated' else 13134)
              and result['eof_padding_samples'] == result['eof_events'] == 0
              and result['rss_scope'] == 'native-C-process-self-linux-kib-not-python-parent'
              and result['ssc305_measured'] is False and len(result['feed_microseconds']) == result['feed_calls'],
              'C_REPLAY_NATIVE_COUNTERS')
    result['supervisor'] = 'fresh-child-from-small-native-process-before-state-allocation'
    return result


def compile_native(output):
    compiler = shutil.which('gcc')
    c.require(compiler and platform.system() == 'Linux', 'C_REPLAY_NATIVE_LINUX_GCC_REQUIRED')
    binary = output/'en63_replay'
    command = [compiler, *FLAGS, '-I', str(HERE), '-I', str(output/'export'),
               str(HERE/'en63_stream.c'), str(HERE/'replay_cli.c'), '-lm', '-o', str(binary)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    c.require(run.returncode == 0, 'C_REPLAY_COMPILE_FAILED:'+run.stderr[-4000:])
    version = subprocess.check_output([compiler, '--version'], text=True)
    sources = {p.name: c.digest(p.read_bytes()) for p in HERE.iterdir() if p.suffix in ('.c', '.h', '.py', '.md')}
    report = {'schema': 'english63-native-build-v1', 'command': command, 'compiler_version': version,
              'source_sha256': sources, 'binary_sha256': c.digest(binary.read_bytes()),
              'binary_bytes': binary.stat().st_size, 'machine': platform.machine(),
              'system': platform.system(), 'libc': list(platform.libc_ver()),
              'stdout': run.stdout, 'stderr': run.stderr, 'physical_ssc305': False}
    c.write_json(output/'BUILD.json', report)
    return binary


def run(args):
    import numpy as np
    sys.path.insert(0, str(ROOT/'research/english63_online_pilot'))
    import online_pilot as online
    output = args.output_dir
    c.require(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'C_REPLAY_CREATE_ONLY')
    trained = args.trained_result_dir
    source = source_result(trained)
    c.require(c.digest((trained/'model/events.jsonl').read_bytes()) == source['events_sha256'], 'C_REPLAY_SOURCE_EVENTS')
    runtime = online.configure_runtime()
    data = online.load_data(args.tts_zip, args.review_json)
    output.mkdir()
    try:
        exported = export(trained, output/'export')
        binary = compile_native(output)
        models, resources = [], []
        with tempfile.TemporaryDirectory(prefix='en63-c-pcm-') as temporary:
            temporary = Path(temporary)
            paths = []
            for index, cell in enumerate(sorted(data['sources'])):
                samples = data['sources'][cell][2]
                raw = (samples*32768).astype('<i2').tobytes()
                c.require(c.digest(raw) == data['by_id'][cell]['pcm_sha256'], 'C_REPLAY_ORIGINAL_PCM_BYTES')
                path = temporary/(str(index).zfill(4)+'.pcm')
                path.write_bytes(raw); paths.append(path)
            manifest = temporary/'inputs.txt'
            manifest.write_text(''.join(str(p)+'\n' for p in paths))
            for kind in c.KINDS:
                model = online.load_new_weights(trained/'model'/(kind+'-weights.npz'), kind, c.SOURCE_WEIGHTS[kind])
                refs = {mode: reference(model, kind, data, mode == 'continuous', online) for mode in ('isolated', 'continuous')}
                source_trace_identity = verify_original_proposals(refs, kind, source, data, trained)
                np.savez_compressed(output/(kind+'-reference.npz'), **refs,
                                    source_order=np.asarray(sorted(data['sources'])))
                maxima = dict.fromkeys(c.TOLERANCES, 0.0)
                for mode, reference_rows in refs.items():
                    previous = None
                    for plan in c.PLANS:
                        trace = output/(kind+'-'+mode+'-'+plan+'.trace')
                        resource = native(binary, kind, mode, plan, manifest, trace)
                        resources.append(resource)
                        actual = read_trace(trace)
                        errors = compare(reference_rows, actual)
                        c.require(resource['proposal_count'] == len(event_signature(reference_rows)), 'C_REPLAY_RESOURCE_EVENT_COUNT')
                        for field in maxima: maxima[field] = max(maxima[field], errors[field])
                        if previous is not None:
                            c.require(previous == trace.read_bytes(), 'C_REPLAY_CHUNK_NOT_BIT_IDENTICAL')
                        previous = trace.read_bytes()
                profile = native(binary, kind, 'continuous', '160', manifest, '-')
                c.require(profile['trace_io_enabled'] is False and profile['proposal_count'] == len(event_signature(refs['continuous'])),
                          'C_REPLAY_PROFILE_CHANGED_OUTPUT')
                resources.append(profile)
                models.append({'kind': kind, 'source_weights_sha256': c.SOURCE_WEIGHTS[kind],
                               'isolated_clips': 63, 'isolated_frames': len(refs['isolated']),
                               'continuous_frames': len(refs['continuous']), 'chunk_plans': c.PLANS,
                               'reference_forward_sequences': 64,
                               'max_feature_abs_error': maxima['features'], 'max_logit_abs_error': maxima['logits'],
                               'max_probability_abs_error': maxima['probabilities'],
                               'chunk_probabilities_bit_identical': True, 'chunk_events_identical': True,
                               'python_c_events_identical': True, 'eof_padding_samples': 0, 'eof_events': 0,
                               'continuous_state_resets': 0, 'resource_profile_scope': 'native-linux-host-not-ssc305',
                               'isolated_proposals': event_signature(refs['isolated']),
                               'continuous_proposals': event_signature(refs['continuous']),
                               'original_runner_trace_identity_diagnostic': source_trace_identity,
                               'native_profile_without_trace_io': {k: v for k, v in profile.items() if k != 'feed_microseconds'}})
        c.write_json(output/'RESOURCES.json', {'schema': 'english63-native-self-resource-v1',
                     'scope': 'native-linux-host-not-ssc305', 'measurements': resources,
                     'stack_measured': False, 'core_data_plane_heap_allocations': 0,
                     'process_peak_rss_is_not_arena_or_weight_bytes': True, 'ssc305_qualified': False})
        summary = {'schema': c.SCHEMA, 'status': c.STATUS, 'source_audio_clips': 63,
                   'source_pcm_samples': 2101760, 'source_tts_zip_sha256': c.TTS_SHA,
                   'source_review_json_sha256': c.REVIEW_SHA, 'source_training_run_id': 43595338774,
                   'source_training_summary_sha256': c.SOURCE_SUMMARY_SHA,
                   'models': models, 'numeric_tolerances': c.TOLERANCES, 'runtime': runtime,
                   'dataset_audit': data['audit'], 'role_counts': source['role_counts'],
                   'optimizer_updates': 0, 'new_tts_asr_ipa_calls': 0,
                   'formal_training_admitted': False, 'human_gold': False,
                   'model_promotion_allowed': False, 'model_selection_made': False,
                   'threshold_selection': False, 'exported_models': exported['models'],
                   **{name: 'NOT_RUN' for name in c.UNMEASURED}}
        c.validate_summary(summary)
        c.write_json(output/'SUMMARY.json', summary)
        c.write_json(output/'REPLAY.json', c.verification_receipt(output))
        return c.verify_saved(output)
    except Exception as exc:
        c.write_json(output/'FAILED.json', {'schema': c.SCHEMA, 'status': 'FAILED_NOT_PARITY_EVIDENCE',
                     'error': str(exc), 'optimizer_updates': 0, 'model_promotion_allowed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tts-zip', type=Path)
    parser.add_argument('--review-json', type=Path)
    parser.add_argument('--trained-result-dir', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output-dir', type=Path)
    mode.add_argument('--verify-result-dir', type=Path)
    args = parser.parse_args()
    if args.verify_result_dir:
        result = c.verify_saved(args.verify_result_dir)
    else:
        c.require(all(x is not None and x.is_absolute() for x in (args.tts_zip, args.review_json, args.trained_result_dir)),
                  'C_REPLAY_ABSOLUTE_SOURCE_INPUTS')
        result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('EN63_C_REPLAY_FAILED: '+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
