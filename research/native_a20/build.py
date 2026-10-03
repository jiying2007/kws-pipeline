"""Build and test isolated research code with invented inputs only, offline."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import ctypes_api
import generate_twiddles

ROOT = Path(__file__).resolve().parent
FLAGS = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fno-fast-math',
         '-ffp-contract=off', '-frounding-math', '-fexcess-precision=standard', '-fPIC']
SOURCES = ['baseline/precision64/model/a20_fsmn.c', 'baseline/precision64/model/load.c',
           'baseline/precision64/model/sha256.c', 'baseline/native/stream/splice.c',
           'baseline/decoder/a20_decoder.c', 'src/donor_fft64.c', 'src/pcm_fft64.c',
           'src/a20_stream_fft64.c', 'src/abi_fft64.c']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New build directory; no overwrites')
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('--output must be outside the source tree')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    def run(command, **kwargs):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120, **kwargs)
        records.append({'command': command, 'returncode': result.returncode,
                        'stdout': result.stdout, 'stderr': result.stderr})
        (out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
        if result.returncode:
            print(result.stdout, end=''); print(result.stderr, end='', file=sys.stderr)
            raise RuntimeError('Build or invented test failed; see commands.json')
        return result.stdout
    run([sys.executable, '-B', str(ROOT/'tests/check_manifest.py')])
    header, certificate = generate_twiddles.generate()
    assert header.encode() == (ROOT/'src/fft64_twiddles.h').read_bytes()
    certificate.pop('certificates')
    assert certificate == json.loads((ROOT/'twiddle-certificate-summary.json').read_text())
    flags = FLAGS[:]
    if args.sanitize:
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    objects = []
    for i, source in enumerate(SOURCES):
        target = out/f'{i:02d}.o'; objects.append(str(target))
        run([args.cc, *flags, '-c', source, '-o', str(target)])
    shared = out/'liba20_fft64.so'
    run([args.cc, *flags, '-shared', *objects, '-lm', '-o', str(shared)])
    run([args.cc, *flags, 'src/a20_pcm_fft64_cli.c', *objects, '-lm', '-o', str(out/'a20_pcm_fft64_cli')])
    # Resource/ABI queries perform no model initialization or forward call.
    run([args.cc, *FLAGS, 'resource_sizes.c', '-o', str(out/'resource_sizes')])
    sizes = json.loads(run([str(out/'resource_sizes')]))
    (out/'resource-sizes.json').write_text(json.dumps(sizes, indent=2, sort_keys=True)+'\n')
    assert sizes['model_cache_bytes'] == 22528 and sizes['weights_bytes'] == 1565280
    if not args.sanitize:
        _, checked = ctypes_api.load(shared)
        arena = ctypes_api.Aligned_Arena(checked['stream_bytes'], checked['stream_alignment'])
        assert arena.pointer.value % arena.alignment == 0
    tests = {
        'frontend': ['tests/test_frontend.c', 'src/pcm_fft64.c', 'baseline/native/stream/splice.c', 'baseline/native/donor_fbank/donor_fbank.c'],
        'model': ['tests/test_model.c', 'src/donor_fft64.c', 'src/pcm_fft64.c', 'baseline/native/stream/splice.c'],
        'decoder': ['tests/test_decoder.c', 'baseline/decoder/a20_decoder.c'],
        'integration': ['tests/test_integration.c', 'baseline/precision64/model/a20_fsmn.c', 'baseline/decoder/a20_decoder.c', 'src/donor_fft64.c', 'src/pcm_fft64.c', 'baseline/native/stream/splice.c'],
        'identity': ['tests/test_identity.c', 'baseline/precision64/model/a20_fsmn.c', 'baseline/precision64/model/load.c', 'baseline/precision64/model/sha256.c'],
    }
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
    for name, sources in tests.items():
        binary = out/('test_'+name)
        run([args.cc, *flags, *sources, '-lm', '-o', str(binary)])
        print(run([str(binary)], env=env), end='')
    receipt = {'schema':'a20-public-offline-build-v1', 'passed':True,
               'compiler':run([args.cc,'--version']).splitlines()[0],
               'flags':flags, 'invented_tests':list(tests), 'asan_ubsan':args.sanitize,
               'leak_detection':False, 'model_asset_loads':0, 'audio_asset_loads':0,
               'pretrained_model_forwards':0, 'network_access':False,
               'source_manifest_sha256':hashlib.sha256((ROOT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
               'library_sha256':hashlib.sha256(shared.read_bytes()).hexdigest()}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
