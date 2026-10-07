"""Reproduce and verify this saved result with Python's standard library only.

No compiler, shared library, model, frontend, decoder, network or audio playback
is used. The frozen scorer validates existing logits and summarizes greedy tokens; it
does not execute the native decoder.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BINDINGS_SHA256 = '9893a8b10bda855856e43271b3a50a8ed1f557b22e54e7065ccf45844f02dd6d'

class VerificationError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise VerificationError(message)

def descriptor(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result

def parse(data):
    return json.loads(data, object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          VerificationError('nonfinite JSON: ' + x)))

def checked(root, name, expected):
    path = root / name
    require(not path.is_symlink() and path.is_file(), 'missing/nonregular file: ' + name)
    require(path.resolve().is_relative_to(root.resolve()), 'file outside root: ' + name)
    data = path.read_bytes()
    require(descriptor(data) == expected, 'byte/hash mismatch: ' + name)
    return data

def source_and_scorer():
    path = ROOT / 'metadata/bindings.json'
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == BINDINGS_SHA256,
            'public bindings changed')
    bindings = parse(data)
    for name, identity in bindings['copied_frozen_files'].items():
        checked(ROOT, name, identity)
    spec = importlib.util.spec_from_file_location('melo5_frozen_scorer', ROOT / 'src/score_endpoint.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return bindings, scorer

def verify(data_dir):
    data_dir = Path(data_dir)
    require(data_dir.is_dir() and not data_dir.is_symlink(), 'data directory required')
    binding, scorer = source_and_scorer()
    expected = binding['data_files']
    require({p.name for p in data_dir.iterdir()} == set(expected),
            'data file set differs from the closed four-file archive')
    saved = {name: checked(data_dir, name, identity) for name, identity in expected.items()}
    raw = saved['original_A20.raw.jsonl']
    records = [parse(line) for line in raw.splitlines()]
    require(len(records) == 50 and 'pid' not in records[0], 'public trace shape/PID')
    require(sum(len(r.get('logits', [])) for r in records) == 128,
            'saved logit row count')
    result = scorer.score(data_dir / 'original_A20.raw.jsonl',
                          ROOT / 'metadata/manifest.json', ROOT / 'metadata/geometry.json',
                          binding['acquisition_bindings'])
    projection = binding['projections']['original_A20.raw.jsonl']
    require(result['source_hashes']['raw_sha256'] == projection['public']['sha256'],
            'public raw hash differs from projection')
    # The unedited historical observation identifies the acquired (pre-PID-removal)
    # raw bytes. This explicit provenance normalization changes no scientific value.
    result['source_hashes']['raw_sha256'] = projection['original']['sha256']
    reproduced = (json.dumps(result, ensure_ascii=False, sort_keys=True,
                             indent=2, allow_nan=False) + '\n').encode()
    require(reproduced == saved['observations.json'],
            'saved observations do not reproduce byte-for-byte')
    summary = parse(saved['SUMMARY.json'])
    streams = result['streams']
    counts = {
        'expected_keyword_hits': sum(s['word_label_observation']['expected_keyword_observed'] is True for s in streams),
        'human_word_positive_clips': sum(s['label'] > 0 for s in streams),
        'nonwake_clips_with_events': sum(s['label'] == 0 and s['event_count'] > 0 for s in streams),
        'human_word_nonwake_clips': sum(s['label'] == 0 for s in streams),
        'activation_events': sum(s['event_count'] for s in streams),
    }
    require(counts == summary['descriptive_counts'], 'summary counts disagree')
    require(result['qualification'] is False, 'unexpected qualification')
    resources = parse(saved['resources.json'])
    require(resources['bindings'] == binding['acquisition_bindings'], 'resource binding drift')
    require(resources['returncode'] == 0 and resources['error'] is None,
            'acquisition did not complete successfully')
    require(all('monotonic_ns' not in sample for sample in resources['proc_observations']),
            'absolute process clock in public projection')
    timing = summary['timing_and_memory']
    require(timing['supervisor_launch_to_reap_wall_seconds'] == resources['supervisor_launch_to_reap_wall_seconds'], 'whole-process wall mismatch')
    require(timing['terminal_user_plus_system_cpu_seconds'] == resources['owned_child_user_cpu_seconds'] + resources['owned_child_system_cpu_seconds'], 'whole-process CPU mismatch')
    require(timing['rusage_lifetime_peak_rss_kib'] == resources['child_lifetime_peak_rss_kib'] == result['native_run_end']['maxrss_kib'], 'lifetime RSS mismatch')
    require(timing['proc_max_observed_rss_kib'] == max(s['VmRSS'] for s in resources['proc_observations']), 'sampled RSS mismatch')
    return {'status': 'PASS_SAVED_ONLY_REPRODUCTION', 'jsonl_records': len(records),
            'saved_six_class_logit_rows': 128, 'descriptive_counts': counts,
            'observations_bytes_reproduced': len(reproduced),
            'new_model_frontend_decoder_audio_calls': 0}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.data_dir), ensure_ascii=False, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit('VERIFICATION FAILED: ' + str(exc))

if __name__ == '__main__':
    main()
