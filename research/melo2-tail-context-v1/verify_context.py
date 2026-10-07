"""Verify the saved one-condition annex against the unchanged original core.

Standard library only; no audio, native libraries, model, frontend, native
 decoder, compiler or network is invoked. Temporary files contain source and
saved evidence only, and are removed on exit.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BINDINGS_SHA256 = '9e584c61b1c8ec40c04ced34f1c19e68035ac82afe9eb8a3ff99c6900e68054b'

class VerificationError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise VerificationError(message)

def descriptor(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def checked(root, name, expected):
    path = Path(root) / name
    require(path.is_file() and not path.is_symlink(), 'missing/nonregular file: ' + name)
    require(path.resolve().is_relative_to(Path(root).resolve()), 'nonlocal file: ' + name)
    data = path.read_bytes()
    require(descriptor(data) == expected, 'byte/hash mismatch: ' + name)
    return data

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def apply_delta(original, delta):
    require(descriptor(original) == delta['core_original'], 'source delta base mismatch')
    text = original.decode('utf-8')
    previous = 0
    for edit in delta['edits']:
        a, b = edit['start'], edit['end']
        require(type(a) is int and type(b) is int and previous <= a <= b <= len(text),
                'invalid/overlapping source delta')
        require(text[a:b] == edit['old'], 'source delta old-text mismatch')
        previous = b
    for edit in reversed(delta['edits']):
        text = text[:edit['start']] + edit['new'] + text[edit['end']:]
    data = text.encode('utf-8')
    require(descriptor(data) == delta['frozen_context_result'], 'source delta result mismatch')
    return data

@contextmanager
def staged(core_source, core_data):
    binding_bytes = (ROOT / 'metadata/bindings.json').read_bytes()
    require(hashlib.sha256(binding_bytes).hexdigest() == BINDINGS_SHA256,
            'annex bindings changed')
    binding = json.loads(binding_bytes)
    for name, identity in binding['copied_frozen_files'].items():
        checked(ROOT, name, identity)
    core = binding['core_dependency']
    for name, identity in core['source_files'].items():
        checked(core_source, name, identity)
    core_verifier = load_module('melo5_public_core_verifier', Path(core_source) / 'verify_saved.py')
    core_report = core_verifier.verify(core_data)
    control = checked(core_data, 'original_A20.raw.jsonl', core['public_control_raw'])
    for key in ('model_sha256', 'library_sha256', 'decoder_config_sha256'):
        core_binding = json.loads((Path(core_source) / 'metadata/bindings.json').read_bytes())
        require(binding['acquisition_bindings'][key] == core_binding['acquisition_bindings'][key],
                'core native identity drift: ' + key)
    with tempfile.TemporaryDirectory(prefix='melo-context-saved-') as directory:
        work = Path(directory)
        for name, delta in binding['source_deltas'].items():
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(apply_delta((Path(core_source) / name).read_bytes(), delta))
        for name in binding['copied_frozen_files']:
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        (work / 'controls').mkdir()
        (work / 'controls/original_A20.raw.jsonl').write_bytes(control)
        receipt = dict(binding['saved_control_receipt'])
        require(receipt['source_raw_sha256'] == core['acquired_control_raw']['sha256'],
                'original control receipt identity mismatch')
        receipt['source_raw_sha256'] = core['public_control_raw']['sha256']
        (work / 'controls/saved-control-validation.json').write_text(json.dumps(receipt))
        comparator = load_module('melo_context_frozen_comparator', work / 'src/compare_prefix.py')
        scorer = load_module('melo_context_frozen_scorer', work / 'src/score_endpoint.py')
        prior = sys.modules.get('compare_prefix')
        sys.modules['compare_prefix'] = comparator
        try:
            yield binding, scorer, comparator, work, core_report
        finally:
            if prior is None:
                del sys.modules['compare_prefix']
            else:
                sys.modules['compare_prefix'] = prior

def verify(data_dir, core_source, core_data):
    data_dir = Path(data_dir)
    require(data_dir.is_dir() and not data_dir.is_symlink(), 'annex data directory required')
    with staged(core_source, core_data) as (binding, scorer, comparator, work, core_report):
        require({p.name for p in data_dir.iterdir()} == set(binding['data_files']),
                'annex data file set differs from closed four-file archive')
        saved = {name: checked(data_dir, name, identity)
                 for name, identity in binding['data_files'].items()}
        result = scorer.score(data_dir / 'original_A20.raw.jsonl',
                              work / 'metadata/manifest.json', work / 'metadata/geometry.json',
                              binding['acquisition_bindings'])
        require(result['source_hashes']['raw_sha256'] == binding['projections']['original_A20.raw.jsonl']['public']['sha256'], 'context public raw hash')
        require(result['prefix_context_comparison']['saved_original_raw_sha256'] == binding['core_dependency']['public_control_raw']['sha256'], 'control public raw hash')
        # Exact historical observations deliberately retain the acquired hashes.
        # Normalize only these two declared provenance references for comparison.
        result['source_hashes']['raw_sha256'] = binding['projections']['original_A20.raw.jsonl']['original']['sha256']
        result['prefix_context_comparison']['saved_original_raw_sha256'] = binding['core_dependency']['acquired_control_raw']['sha256']
        reproduced = (json.dumps(result, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + '\n').encode()
        require(reproduced == saved['observations.json'], 'historical context observations did not reproduce')
        comparisons = result['prefix_context_comparison']['streams']
        require(all(s['all_support_contained_fp32_logits_equal'] for s in comparisons), 'preserved-support logit drift')
        rows = sum(s['support_contained_rows'] for s in comparisons)
        require(rows == 50, 'preserved-support row count')
        end = result['native_run_end']
        require((end['model_rows'], end['decoder_rows_decoded']) == (70, 69), 'model/decoded row accounting')
        summary = json.loads(saved['SUMMARY.json'])
        require(summary['counts']['activation_events'] == end['event_count'] == 1, 'event count')
        require([s['event_count'] for s in result['streams']] == [1, 0], 'per-stream event count')
        event = comparisons[0]['modified_events'][0]
        require(event['availability_minus_original_file_end_samples'] == 2325 and
                event['availability_minus_original_file_end_seconds'] == 0.1453125 and
                event['available_samples'] == 14400 and event['word_end_latency'] is None,
                'EOF-relative availability interpretation')
        require(core_report['descriptive_counts']['expected_keyword_hits'] == 0, 'original controls changed')
        return {'status': 'PASS_SAVED_CONTEXT_AND_CORE', 'prefix_rows_equal': rows,
                'prefix_float32_scalars_equal': rows * 6, 'model_rows': 70,
                'decoded_rows': 69, 'modified_events': {'M1': 1, 'M2': 0},
                'M1_availability_after_original_EOF_ms': 145.3125,
                'original_positive_hits': '0/2 unchanged',
                'observations_bytes_reproduced': len(reproduced),
                'new_model_frontend_native_decoder_audio_calls': 0}

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--core-source', type=Path, default=ROOT.parent / 'melo5-a20-diagnostic-v1')
    parser.add_argument('--core-data', type=Path)
    args = parser.parse_args()
    if args.core_data is None:
        args.core_data = args.data_dir.parent / '2026-10-06-melo5-a20-diagnostic'
    return args

if __name__ == '__main__':
    args = arguments()
    try:
        print(json.dumps(verify(args.data_dir, args.core_source, args.core_data), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit('VERIFICATION FAILED: ' + str(exc))
