"""Strict local-only Melo collector; legacy schema strings mean compatibility.

No intended text, speaker ID, voice name, model name or generation receipt is
included in the blind archive. No ASR runtime is imported or called here.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import wave
import zipfile

from run_melo6 import (EXPERIMENT, OUTPUT_CAP, MiB, ROOT, canonical, require,
                       regular, sha, validate_plan, write_json, bounded_write)


def check_pcm(raw):
    require(44 < len(raw) <= 320044, 'PCM_FILE_SIZE')
    with wave.open(io.BytesIO(raw), 'rb') as f:
        require((f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype()) ==
                (16000, 1, 2, 'NONE'), 'PCM_FORMAT')
        frames = f.getnframes()
        pcm = f.readframes(frames)
    require(0 < frames <= 160000 and len(raw) == 44 + frames * 2, 'PCM_FRAMES')
    expected = (b'RIFF' + struct.pack('<I', len(raw) - 8) + b'WAVEfmt ' +
                struct.pack('<IHHIIHH', 16, 1, 1, 16000, 32000, 2, 16) +
                b'data' + struct.pack('<I', len(pcm)))
    require(raw[:44] == expected and len(pcm) == frames * 2, 'CANONICAL_PCM_HEADER')
    return pcm


def pack(runtime, out, plan_path, approved_sha):
    _, fixtures, _ = validate_plan(plan_path, approved_sha)
    require(runtime.is_dir() and not runtime.is_symlink(), 'RUNTIME_DIRECTORY')
    # Place the blind pack inside the same capped output tree, avoiding an
    # unaccounted second copy. No intermediate extracted WAV copies are made.
    require(out.resolve().parent == runtime.resolve() and not out.exists(), 'FRESH_BLIND_SUBDIRECTORY')
    generation = runtime / 'generation'
    freeze_path = generation / 'generation-freeze.json'
    require(regular(freeze_path), 'GENERATION_FREEZE_REQUIRED')
    frozen = json.loads(freeze_path.read_bytes())
    require(frozen['schema'] == 'melo-six-generation-freeze-v1' and
            frozen['experiment_id'] == EXPERIMENT and
            frozen['plan_sha256'] == approved_sha, 'GENERATION_FREEZE_IDENTITY')
    files = frozen['files']
    expected_names = {'generation-receipt.json'} | {
        f'melo6-{i:03d}{suffix}' for i in range(1, 7)
        for suffix in ('.started.json', '.native-f32.wav', '.pcm16.wav')}
    require(len(files) == len(expected_names) and {r['path'] for r in files} == expected_names,
            'COMPLETE_GENERATION_INVENTORY')
    require({p.name for p in generation.iterdir()} == expected_names | {'generation-freeze.json'},
            'GENERATION_FILE_SET_CHANGED')
    for item in files:
        path = generation / item['path']
        require(regular(path) and path.stat().st_size == item['bytes'] and
                sha(path) == item['sha256'], 'FROZEN_GENERATION_MISMATCH')
    receipt = json.loads((generation / 'generation-receipt.json').read_bytes())
    require(receipt['schema'] == 'melo-six-generation-receipt-v1' and
            receipt['experiment_id'] == EXPERIMENT and receipt['plan_sha256'] == approved_sha and
            receipt['status'] == 'six_candidates_generated' and
            receipt['attempts_consumed'] == 6 and len(receipt['generation_rows']) == 6,
            'INCOMPLETE_GENERATION_REFUSED')
    contents = {}
    clips = []
    import numpy as np
    from run_melo6 import input_arrays, input_boundary
    for i, row in enumerate(receipt['generation_rows'], 1):
        source = f'melo6-{i:03d}'
        require(row['source_id'] == source and row['status'] == 'generated_candidate'
                and row['attempt_consumed'] is True, 'MISSING_OR_REORDERED_CALL')
        claim = json.loads((generation / (source + '.started.json')).read_bytes())
        require(claim['source_id'] == source and claim['attempt'] == 1 and
                claim['attempt_consumed'] is True and claim['plan_sha256'] == approved_sha,
                'WRITE_AHEAD_CLAIM_REQUIRED')
        require(claim['requested_outputs'] == ['y'] and
                claim['inputs'] == input_boundary(input_arrays(fixtures[i - 1], np)),
                'EXPLICIT_INPUT_BOUNDARY_MISMATCH')
        native = generation / (source + '.native-f32.wav')
        require(sha(native) == row['native_wav_sha256'], 'NATIVE_HASH_MISMATCH')
        raw = (generation / (source + '.pcm16.wav')).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == row['pcm_wav_sha256'], 'PCM_WAV_HASH_MISMATCH')
        pcm = check_pcm(raw)
        require(hashlib.sha256(pcm).hexdigest() == row['pcm_sha256'], 'PCM_CONTENT_HASH_MISMATCH')
        name = f'audio/{digest}.wav'
        require(name not in contents, 'DUPLICATE_WAVEFORM_REFUSED')
        contents[name] = raw
        clips.append({'audio_id': f'clip-{i:06d}', 'audio_path': name, 'wav_sha256': digest})
    contents['job.json'] = canonical({'schema': 'blind-asr-job-v1', 'clips': clips})
    freeze = {'schema': 'qwen6-blind-input-freeze-v1',
              'job_sha256': hashlib.sha256(contents['job.json']).hexdigest(),
              'files': [{'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                        for name, raw in sorted(contents.items())]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, raw in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 5, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            z.writestr(info, raw)
    archive = buffer.getvalue()
    require(len(archive) <= 2 * MiB, 'BLIND_ARCHIVE_CAP')
    from supervision import tree_bytes
    freeze_bytes = canonical(freeze)
    require(tree_bytes(runtime) + len(archive) + len(freeze_bytes) + 4096 <= OUTPUT_CAP,
            'COMBINED_TTS_BLIND_LOG_CAP')
    out.mkdir()
    bounded_write(out / 'blind-inputs.zip', archive)
    bounded_write(out / 'blind-input-freeze.json', freeze_bytes)
    result = {'schema': 'melo-six-blind-export-receipt-v1', 'experiment_id': EXPERIMENT,
              'plan_sha256': approved_sha, 'blind_archive_sha256': sha(out / 'blind-inputs.zip'),
              'blind_freeze_sha256': sha(out / 'blind-input-freeze.json'),
              'blind_job_sha256': freeze['job_sha256'], 'clip_count': 6,
              'legacy_schema_names_for_compatibility_only': True,
              'asr_run_performed': False, 'publication_performed': False}
    write_json(out / 'export-receipt.json', result)
    require(tree_bytes(runtime) <= OUTPUT_CAP, 'FINAL_COMBINED_CAP')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--plan', type=Path, default=ROOT / 'execution-plan.json')
    p.add_argument('--plan-sha256', required=True)
    args = p.parse_args()
    try:
        print(json.dumps(pack(args.runtime, args.out, args.plan, args.plan_sha256)))
    except Exception as error:
        from run_melo6 import safe_diagnostic
        print(json.dumps({'status': 'refused', **safe_diagnostic(error),
                          'raw_error_text_disclosed': False}))
        raise SystemExit(1)
