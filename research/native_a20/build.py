"""Build isolated research code offline; target execution is an explicit mode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
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
TESTS = {
    'frontend': ['tests/test_frontend.c', 'src/pcm_fft64.c', 'baseline/native/stream/splice.c', 'baseline/native/donor_fbank/donor_fbank.c'],
    'model': ['tests/test_model.c', 'src/donor_fft64.c', 'src/pcm_fft64.c', 'baseline/native/stream/splice.c'],
    'decoder': ['tests/test_decoder.c', 'baseline/decoder/a20_decoder.c'],
    'integration': ['tests/test_integration.c', 'baseline/precision64/model/a20_fsmn.c', 'baseline/decoder/a20_decoder.c', 'src/donor_fft64.c', 'src/pcm_fft64.c', 'baseline/native/stream/splice.c'],
    'identity': ['tests/test_identity.c', 'baseline/precision64/model/a20_fsmn.c', 'baseline/precision64/model/load.c', 'baseline/precision64/model/sha256.c'],
}
MACHINES = {'i386': 3, 'arm': 40, 'x86_64': 62, 'aarch64': 183}


def read_elf(path):
    """Inspect ELF identity and load-segment bounds without target execution."""
    with Path(path).open('rb') as handle:
        header = handle.read(64)
    if len(header) < 16 or header[:4] != b'\x7fELF':
        raise ValueError(f'{path}: expected ELF')
    elf_class, data, version = header[4:7]
    if elf_class not in (1, 2) or data not in (1, 2) or version != 1:
        raise ValueError(f'{path}: unsupported ELF identification')
    size = 52 if elf_class == 1 else 64
    if len(header) < size:
        raise ValueError(f'{path}: truncated ELF header')
    endian = '<' if data == 1 else '>'
    kind, machine, elf_version = struct.unpack_from(endian+'HHI', header, 16)
    flags_offset, size_offset = (36, 40) if elf_class == 1 else (48, 52)
    if elf_version != 1 or struct.unpack_from(endian+'H', header, size_offset)[0] != size:
        raise ValueError(f'{path}: invalid ELF version/header size')
    flags = struct.unpack_from(endian+'I', header, flags_offset)[0]
    if machine == MACHINES['arm'] and flags & 0x600 == 0x600:
        raise ValueError(f'{path}: conflicting ARM float ABI flags')
    if kind in (2, 3):
        phoff_offset, phsize_offset, phnum_offset = (28, 42, 44) if elf_class == 1 else (32, 54, 56)
        phoff = struct.unpack_from(endian+('I' if elf_class == 1 else 'Q'), header, phoff_offset)[0]
        phsize = struct.unpack_from(endian+'H', header, phsize_offset)[0]
        phnum = struct.unpack_from(endian+'H', header, phnum_offset)[0]
        file_size = Path(path).stat().st_size
        # Extended program-header numbering is deliberately unsupported.
        if (not 0 < phnum < 65535 or phsize != (32 if elf_class == 1 else 56) or
                phoff < size or phoff+phsize*phnum > file_size):
            raise ValueError(f'{path}: invalid/truncated ELF program headers')
        executable_load = False
        with Path(path).open('rb') as handle:
            handle.seek(phoff)
            for _ in range(phnum):
                program = handle.read(phsize)
                if elf_class == 1:
                    ptype, offset, vaddr, _, filesz, memsz, pflags, align = struct.unpack(endian+'IIIIIIII', program)
                else:
                    ptype, pflags, offset, vaddr, _, filesz, memsz, align = struct.unpack(endian+'IIQQQQQQ', program)
                if offset+filesz > file_size:
                    raise ValueError(f'{path}: ELF segment exceeds file bounds')
                if ptype == 1:
                    if (filesz > memsz or (align > 1 and
                            (align & (align-1) or vaddr % align != offset % align))):
                        raise ValueError(f'{path}: invalid ELF load segment')
                    executable_load |= bool(pflags & 1 and filesz)
        if not executable_load:
            raise ValueError(f'{path}: no executable ELF load segment')
    return {'class': 32 if elf_class == 1 else 64, 'machine': machine,
            'endian': 'little' if data == 1 else 'big', 'type': kind,
            'osabi': header[7], 'abi_version': header[8], 'flags': flags}


def check_elf(path, expected, allowed_types):
    header = read_elf(path)
    for key in ('class', 'machine', 'endian'):
        if header[key] != expected[key]:
            raise ValueError(f'{path}: ELF {key}={header[key]}, expected {expected[key]}; '
                             'cross compilers require --mode compile-only and explicit target identity')
    if header['type'] not in allowed_types:
        raise ValueError(f'{path}: unexpected ELF type {header["type"]}')
    # Native mode supplies the running process ABI; compile-only records these
    # fields without pretending to know a vendor's ABI or loader contract.
    for key in ('osabi', 'abi_version'):
        if key in expected and header[key] != expected[key]:
            raise ValueError(f'{path}: incompatible native ELF {key}')
    if header['machine'] == MACHINES['arm'] and 'flags' in expected:
        mask = 0xff000000 | (0x600 if header['type'] in (2, 3) else 0)
        if header['flags'] & mask != expected['flags'] & mask:
            raise ValueError(f'{path}: incompatible native ARM EABI/float flags')
    return header


def target_flag(value):
    # Deliberately exclude arbitrary compiler/linker options (e.g. -ffast-math,
    # response files or plugins) that could replace the numerical/build contract.
    if (value in ('-marm', '-mthumb', '-mbig-endian', '-mlittle-endian') or
            re.fullmatch(r'-m(?:cpu|arch|tune|fpu|float-abi|abi)=[A-Za-z0-9_.+\-]+', value)):
        return value
    raise argparse.ArgumentTypeError('only architecture/ABI selection flags are supported')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New build directory; no overwrites')
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--mode', choices=('native-test', 'compile-only'), default='native-test')
    parser.add_argument('--target-elf-class', type=int, choices=(32, 64))
    parser.add_argument('--target-machine', choices=tuple(MACHINES))
    parser.add_argument('--target-endian', choices=('little', 'big'))
    parser.add_argument('--target-flag', action='append', type=target_flag, default=[],
                        help='Repeat as --target-flag=-mcpu=...; architecture/ABI flags only')
    parser.add_argument('--sysroot', type=Path)
    args = parser.parse_args(argv)
    selectors = (args.target_elf_class, args.target_machine, args.target_endian)
    if args.mode == 'compile-only':
        if any(value is None for value in selectors):
            parser.error('compile-only requires --target-elf-class, --target-machine and --target-endian')
        expected = {'class': args.target_elf_class, 'machine': MACHINES[args.target_machine],
                    'endian': args.target_endian}
    else:
        if any(value is not None for value in selectors) or args.target_flag or args.sysroot:
            parser.error('target options require --mode compile-only')
        # Match the running Python process, not just uname or the compiler name.
        # This conservatively rejects cross builds even if binfmt emulation exists.
        expected = read_elf(Path(sys.executable).resolve())
    if args.sysroot is not None:
        args.sysroot = args.sysroot.resolve()
        if not args.sysroot.is_dir():
            parser.error('--sysroot must be an existing directory')
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error('--output must be outside the source tree')
    out.mkdir(parents=True, exist_ok=False)
    records = []

    def run(command, **kwargs):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=120, **kwargs)
        except (OSError, subprocess.TimeoutExpired) as error:
            records.append({'command': command, 'error': str(error)})
            (out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
            raise
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
    compiler = run([args.cc, '--version']).splitlines()[0]
    triple = run([args.cc, '-dumpmachine']).strip()
    target_options = args.target_flag[:]
    if args.sysroot is not None:
        target_options.append('--sysroot='+str(args.sysroot))
    flags = [*target_options, *FLAGS]
    if args.sanitize:
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    artifacts = {}

    def inspect(path, allowed_types):
        elf = check_elf(path, expected, allowed_types)
        raw = path.read_bytes()
        artifacts[path.name] = {'elf': elf, 'bytes': len(raw),
                                'sha256': hashlib.sha256(raw).hexdigest()}

    objects = []
    for i, source in enumerate(SOURCES):
        target = out/f'{i:02d}.o'; objects.append(str(target))
        run([args.cc, *flags, '-c', source, '-o', str(target)])
        inspect(target, (1,))
    shared = out/'liba20_fft64.so'
    run([args.cc, *flags, '-shared', *objects, '-lm', '-o', str(shared)])
    inspect(shared, (3,))
    cli = out/'a20_pcm_fft64_cli'
    run([args.cc, *flags, 'src/a20_pcm_fft64_cli.c', *objects, '-lm', '-o', str(cli)])
    inspect(cli, (2, 3))
    sizes_binary = out/'resource_sizes'
    # Keep the existing unsanitized resource query, with the same target options.
    run([args.cc, *target_options, *FLAGS, 'resource_sizes.c', '-o', str(sizes_binary)])
    inspect(sizes_binary, (2, 3))
    for name, sources in TESTS.items():
        binary = out/('test_'+name)
        run([args.cc, *flags, *sources, '-lm', '-o', str(binary)])
        inspect(binary, (2, 3))

    # All compiler products are checked before this only execution/load boundary.
    # compile-only never executes resource_sizes/tests or calls ctypes_api.load,
    # including when the specified target happens to equal the build host.
    executions = 0
    dlopens = 0
    if args.mode == 'native-test':
        sizes = json.loads(run([str(sizes_binary)])); executions += 1
        (out/'resource-sizes.json').write_text(json.dumps(sizes, indent=2, sort_keys=True)+'\n')
        assert sizes['model_cache_bytes'] == 22528 and sizes['weights_bytes'] == 1565280
        if not args.sanitize:
            _, checked = ctypes_api.load(shared); dlopens += 1
            arena = ctypes_api.Aligned_Arena(checked['stream_bytes'], checked['stream_alignment'])
            assert arena.pointer.value % arena.alignment == 0
        env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
        for name in TESTS:
            print(run([str(out/('test_'+name))], env=env), end=''); executions += 1
    native = args.mode == 'native-test'
    receipt = {'schema': 'a20-public-offline-build-v2', 'passed': True, 'mode': args.mode,
               'compile_link': 'PASS', 'elf_identity': 'PASS',
               'resource_sizes': 'PASS' if native else 'NOT_RUN',
               'invented_tests_status': 'PASS' if native else 'NOT_RUN',
               'ctypes_abi': 'PASS' if native and not args.sanitize else 'NOT_RUN',
               'compiler': compiler, 'compiler_target_triple': triple,
               'flags': flags, 'resource_flags': [*target_options, *FLAGS],
               'expected_elf': {key: expected[key] for key in ('class', 'machine', 'endian')},
               'artifacts': artifacts, 'compiled_invented_tests': list(TESTS),
               'invented_tests': list(TESTS) if native else [],
               'target_executions': executions, 'target_dlopen_calls': dlopens,
               'asan_ubsan': bool(native and args.sanitize),
               'sanitizer_instrumentation': args.sanitize,
               'leak_detection': False, 'model_asset_loads': 0, 'audio_asset_loads': 0,
               'pretrained_model_forwards': 0, 'network_access': False,
               'vendor_abi_qualified': False, 'board_qualified': False,
               'source_manifest_sha256': hashlib.sha256((ROOT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
               'library_sha256': hashlib.sha256(shared.read_bytes()).hexdigest()}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
