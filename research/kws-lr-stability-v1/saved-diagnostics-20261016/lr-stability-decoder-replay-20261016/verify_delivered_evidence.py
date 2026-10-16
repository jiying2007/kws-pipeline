#!/usr/bin/env python3
"""Verify delivered native trace and suppression evidence without executing C."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--base', type=Path, help='Optional existing original evidence root, for LP/input comparisons')
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root / 'report.json').read_bytes())
    completion = json.loads((root / 'execution-complete.json').read_bytes())
    claim = json.loads((root / 'execution-claim.json').read_bytes())
    freeze = json.loads((root / 'preexecution-freeze.json').read_bytes())
    plan = json.loads((root / 'analysis-plan.json').read_bytes())
    require(report['plan_sha256'] == sha(root / 'analysis-plan.json') == freeze['plan_sha256'], 'plan binding differs')
    require(report['preexecution_freeze_sha256'] == claim['preexecution_freeze_sha256'] == sha(root / 'preexecution-freeze.json'), 'freeze binding differs')
    require(completion['report_sha256'] == sha(root / 'report.json'), 'completion binding differs')
    require(report['native_output_sha256'] == sha(root / 'native-output.jsonl'), 'trace binding differs')
    sources = json.loads((root / 'source-manifest.json').read_bytes())
    require(sha(root / 'source-manifest.json') == freeze['source_manifest_sha256'], 'source manifest binding differs')
    for path, pin in sources['files'].items():
        raw = (root / 'source' / path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'source digest differs')
        require(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == pin['git_blob_sha1'], 'source blob differs')
    # Package intentionally omits host binaries and the duplicated input transport.
    omitted_build_files = []
    for path, expected in freeze['file_sha256'].items():
        if (root / path).is_file():
            require(sha(root / path) == expected, 'frozen delivered file differs: ' + path)
        else:
            require(path.startswith(freeze['build_directory'] + '/'), 'missing frozen source: ' + path)
            omitted_build_files.append(path)
    lines = [json.loads(line) for line in (root / 'native-output.jsonl').read_text().splitlines()]
    require(len(lines) == 1277 and lines[-1]['kind'] == 'terminal', 'incomplete trace')
    frames, terminal = lines[:-1], lines[-1]
    originals = None
    if args.base is not None:
        pins = dict(plan['original_inputs'], **plan['additional_original_inputs'])
        for pin in pins.values():
            require(sha(args.base / pin['relative_path']) == pin['sha256'], 'original evidence differs')
        originals = json.loads((args.base / pins['observation']['relative_path']).read_bytes())['samples']
    cursor = 0
    totals = {'native_decoder_hits': 0, 'mirror_accepted': 0, 'mirror_suppressed': 0}
    for index, sample in enumerate(report['samples']):
        deadline = 0
        counts = dict.fromkeys(totals, 0)
        events = []
        lp_bytes = bytearray()
        for frame in range(sample['input_frames']):
            row = frames[cursor]
            cursor += 1
            require(row['clip_index'] == index and row['frame'] == frame, 'frame order differs')
            require(row['end_sample'] == 400 + 320 * frame and row['speech_active'] == 1, 'control/time differs')
            packed = b''.join(struct.pack('<I', int(value, 16)) for value in row['lp_float32_bits'])
            values = struct.unpack('<5f', packed)
            require(all(math.isfinite(v) for v in values), 'nonfinite input bits')
            require(abs(math.log(math.fsum(math.exp(v) for v in values))) <= 2e-6, 'unnormalized input bits')
            require(row['raw_argmax'] == max(range(5), key=lambda k: values[k]), 'argmax differs')
            lp_bytes.extend(packed)
            if originals is not None:
                require(originals[index]['record_id'] == sample['record_id'], 'record order differs')
                require(packed == struct.pack('<5f', *originals[index]['log_probs'][frame]), 'saved LP bits differ')
            require(row['suppress_until_before'] == deadline, 'mirror pre-state differs')
            hit = row['decoder_hit']
            require(hit in (0, 1), 'invalid hit')
            if hit:
                accepted = row['end_sample'] >= deadline
                outcome = 1 if accepted else 2
                if accepted:
                    deadline = row['end_sample'] + 19200
                counts['native_decoder_hits'] += 1
                counts['mirror_accepted' if accepted else 'mirror_suppressed'] += 1
                require(row['keyword_id'] in (1, 2) and .55 - 1e-7 <= row['confidence'] <= 1, 'invalid event')
                events.append({'frame': frame, 'end_sample': row['end_sample'],
                               'nominal_clip_coordinate_ms': row['end_sample'] / 16,
                               'keyword_id': row['keyword_id'], 'confidence': row['confidence'],
                               'intended_keyword': row['keyword_id'] == (1 if sample['target_class'] == 'KW1' else 2),
                               'mirror_outcome': 'accepted' if accepted else 'suppressed'})
            else:
                outcome = 0
                require(row['keyword_id'] == 0 and row['confidence'] == 0, 'spurious non-hit fields')
            require(row['mirror_outcome'] == outcome and row['suppress_until_after'] == deadline, 'mirror post-state differs')
        require(hashlib.sha256(lp_bytes).hexdigest() == sample['LP_float32_bytes_sha256'], 'per-clip LP identity differs')
        require(sample['counts'] == counts and sample['events'] == events, 'report differs from native trace')
        for key in totals:
            totals[key] += counts[key]
    require(cursor == 1276 and len(report['samples']) == 12, 'fixed budget differs')
    require(report['totals'] == totals and all(terminal[k] == v for k, v in totals.items()), 'totals differ')
    require(completion['saved_data_passes_completed'] == 1 and completion['frames_completed'] == 1276, 'completion count differs')
    print(json.dumps({'status': 'PASS', 'frames_checked': cursor, 'native_decoder_executed': False,
                      'model_executed': False, 'totals': totals,
                      'original_input_comparison': 'PASS' if originals is not None else 'NOT_REQUESTED',
                      'omitted_host_build_files': omitted_build_files,
                      'scope': 'evidence-integrity and independent suppression verification; not a new decoder run or acoustic qualification'}, indent=2))


if __name__ == '__main__':
    main()
