#!/usr/bin/env python3
"""Opt-in public synthetic ARM/QEMU correctness only; never a benchmark."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time

M = Path(__file__).resolve().parent
ROOT = M.parents[1]
spec = importlib.util.spec_from_file_location('arm_synthetic_guard', M / 'build.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
MODES = (('default', []), ('profile', ['-DKWS_B_PROFILE']),
         ('block2', ['-DKWS_AFFINE_BLOCK2']),
         ('both', ['-DKWS_B_PROFILE', '-DKWS_AFFINE_BLOCK2']))
ARM_FLAGS = ['-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard', '-marm']
TOTAL_SECONDS = 300
PROGRAM_SECONDS = 30
AS_BYTES = 1024**3
OUTPUT_BYTES = 8 * 1024**2


def emit(report):
    encoded = json.dumps(report, indent=2)
    if len(encoded.encode()) + 1 > OUTPUT_BYTES:
        raise RuntimeError('serialized evidence exceeds 8 MiB')
    print(encoded)


@contextmanager
def evidence_directory(started, binding):
    with tempfile.TemporaryDirectory(prefix='b-arm-synthetic-') as directory:
        tmp = Path(directory)
        budget = Budget(tmp)
        budget.start = started
        budget.evidence = {'source_binding_sha256': binding, 'artifacts': {}, 'executions': []}
        try:
            yield tmp, budget
        except Exception as exc:
            emit({'status': 'FAIL', 'error': str(exc)[:4096], 'commands': budget.commands,
                  **budget.evidence, 'qualification_allowed': False, 'performance_claim': False})
            raise RuntimeError('ARM synthetic failure; see structured evidence above') from None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source, reference):
    """Bind both branches to the production body; reject ambiguous markers."""
    marker = '#ifdef KWS_AFFINE_BLOCK2\n'
    if source.count(marker) != 1:
        raise RuntimeError('ambiguous affine source marker')
    region = source.split(marker)[1].split('#endif\n', 1)[0]
    if region.count('#else\n') != 1:
        raise RuntimeError('ambiguous affine branches')
    block, scalar = region.split('#else\n')
    signature = 'static void affine('
    if any(s.count(signature) != 1 for s in (block, scalar)):
        raise RuntimeError('affine signature changed')
    scalar_reference = scalar.replace(signature, 'static __attribute__((noinline)) void affine_reference(')
    if scalar_reference.strip() != reference.strip():
        raise RuntimeError('frozen reference differs from production default')
    return {False: scalar.replace(signature, 'static __attribute__((noinline)) void affine_block2('),
            True: block.replace(signature, 'static __attribute__((noinline)) void affine_block2(')}


def check_elf(path):
    raw = path.read_bytes()
    if (len(raw) < 52 or raw[:7] != b'\x7fELF\x01\x01\x01' or
            struct.unpack_from('<H', raw, 18)[0] != 40):
        raise RuntimeError('expected ELF32 little-endian ARM: ' + path.name)
    flags = struct.unpack_from('<I', raw, 36)[0]
    if flags & 0xff000000 != 0x05000000 or not flags & 0x400 or flags & 0x200:
        raise RuntimeError('expected EABI5 hard-float: ' + path.name)
    return {'sha256': sha(path), 'bytes': len(raw), 'elf_flags': hex(flags)}


class Budget:
    def __init__(self, directory):
        self.directory = directory
        self.start = time.monotonic()
        self.cpu = 0.0
        self.output = 0
        self.commands = []
        self.evidence = {}

    def run(self, argv, kind):
        wall = min(PROGRAM_SECONDS, TOTAL_SECONDS - (time.monotonic() - self.start))
        cpu = min(PROGRAM_SECONDS, TOTAL_SECONDS - self.cpu)
        remaining = OUTPUT_BYTES - self.output
        if min(wall, cpu, remaining) <= 0:
            raise RuntimeError('synthetic aggregate budget exhausted')
        # Record attempts before invocation. Failure propagates; there is no retry.
        record = {'kind': kind, 'argv': list(argv)}
        self.commands.append(record)
        try:
            output, used = guard.run_bounded(argv, self.directory, wall, cpu, output_limit=remaining)
        except Exception as exc:
            record['error'] = str(exc)[:4096]
            raise
        self.cpu += used
        self.output += len(output.encode())
        record.update(returncode=0, output=output, output_sha256=hashlib.sha256(output.encode()).hexdigest())
        return output


def execute(sysroot):
    started = time.monotonic()
    if any(k in os.environ for k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT')):
        raise RuntimeError('loader overrides forbidden')
    binding = guard.validate_sources(ROOT)
    tools = {}
    for name in ('arm-linux-gnueabihf-gcc', 'arm-linux-gnueabihf-nm', 'arm-linux-gnueabihf-readelf', 'qemu-arm-static'):
        tools[name] = shutil.which(name, path='/usr/bin:/bin')
        if not tools[name]:
            raise RuntimeError('required tool missing (no skip): ' + name)
    sysroot = Path(sysroot).resolve()
    loader = sysroot / 'lib/ld-linux-armhf.so.3'
    if not loader.is_file():
        raise RuntimeError('ARM hard-float sysroot loader missing')
    bodies = extract((M / 'fsmn2599.c').read_text(), (M / 'affine_reference.inc').read_text())
    # Isolated imported guard instance: existing host/default guard limit is unchanged.
    guard.AS_LIMIT = AS_BYTES
    with evidence_directory(started, binding) as (tmp, budget):
        versions = {}
        budget.evidence['tool_versions'] = versions
        budget.evidence['source_files'] = json.loads((M / 'source-binding.json').read_text())['files']
        for name, path in tools.items():
            versions[name] = budget.run([path, '--version'], 'tool-version')
        cc = tools['arm-linux-gnueabihf-gcc']
        vendor = tmp / 'include/vendor'
        vendor.mkdir(parents=True)
        (vendor / 'fft_twiddles.h').symlink_to(ROOT / guard.FFT)
        decoder = ROOT / guard.DECODER
        common = [cc, *guard.FLAGS, *ARM_FLAGS, '-I'+str(M), '-I'+str(decoder), '-I'+str(tmp / 'include')]
        sources = [str(M / n) for n in ('b_stream_adapter.c', 'canonical_pcm.c', 'canonical_fbank.c', 'fsmn2599.c', 'sha256.c', 'softmax2599.c')]
        sources.append(str(decoder / 'kws3.c'))
        # Bound QEMU's reserved guest VA as well as host RLIMIT_AS.
        emulator = [tools['qemu-arm-static'], '-R', '256M', '-cpu', 'cortex-a7', '-L', str(sysroot)]
        modes, executions = {}, []
        for name, defines in MODES:
            folder = tmp / name
            folder.mkdir()
            inc = folder / 'affine_block2.inc'
            inc.write_text(bodies['-DKWS_AFFINE_BLOCK2' in defines])
            lib = folder / 'runtime.so'
            extra = [str(M / 'b_profile.c')] if '-DKWS_B_PROFILE' in defines else []
            budget.run(common + defines + ['-shared', '-fPIC', '-Wl,-z,defs', *sources, *extra, '-lm', '-o', str(lib)], 'compile')
            artifacts = {'runtime.so': check_elf(lib)}
            budget.evidence['artifacts'][name] = artifacts
            symbols = budget.run([tools['arm-linux-gnueabihf-nm'], '-D', '--defined-only', str(lib)], 'symbols')
            expected = guard.ADAPTER | guard.DECODER_EXPORTS | guard.CORE_EXPORTS
            if extra:
                expected |= {'b_profile_get', 'b_profile_reset', 'b_profile_size'}
            if {line.split()[-1] for line in symbols.splitlines() if line.strip()} != expected:
                raise RuntimeError('unexpected ARM ABI: ' + name)
            attributes = budget.run([tools['arm-linux-gnueabihf-readelf'], '-A', str(lib)], 'elf-attributes')
            if 'Tag_CPU_arch: v7' not in attributes or 'Tag_ABI_VFP_args: VFP registers' not in attributes:
                raise RuntimeError('ARMv7 VFP calling convention missing')
            fixtures = (
                ('contract', [str(M / 'test_profile_contract.c'), str(lib)]),
                ('mock', [str(M / 'test_b_stream_adapter.c'), str(M / 'canonical_pcm.c')]),
                ('unit', [str(M / 'test_profile_unit.c')]),
                ('affine', [str(M / 'test_affine_block2.c')]),
            )
            outputs = {}
            for test, inputs in fixtures:
                binary = folder / test
                budget.run(common + defines + ['-DKWS_FIXTURE_CORRECTNESS_ONLY', '-I'+str(folder), *inputs, '-lm', '-o', str(binary)], 'compile')
                artifacts[test] = check_elf(binary)
                attempt = {'mode': name, 'fixture': test, 'binary_sha256': artifacts[test]['sha256'], 'command_index': len(budget.commands)}
                budget.evidence['executions'].append(attempt)
                output = budget.run([*emulator, str(binary)], 'synthetic-execution')
                outputs[test] = output
                executions.append({'mode': name, 'fixture': test, 'binary_sha256': artifacts[test]['sha256'], 'command_index': len(budget.commands)-1})
                if test == 'affine' and (not output.startswith('PASS 648 bytewise output and FP sticky-flag comparisons;') or 'reference_ns' in output):
                    raise RuntimeError('affine correctness-only completion missing')
            modes[name] = {'defines': defines, 'artifacts': artifacts, 'exports': sorted(expected),
                           'extracted_source_sha256': sha(inc), 'outputs': outputs}
        for test in ('contract', 'mock', 'unit', 'affine'):
            if len({m['outputs'][test] for m in modes.values()}) != 1:
                raise RuntimeError('four-mode observable fixture outputs differ: ' + test)
        if guard.validate_sources(ROOT) != binding:
            raise RuntimeError('source identity changed during execution')
        tool_hashes = {n: sha(Path(p)) for n,p in tools.items()}
        elapsed = time.monotonic() - started
        if elapsed > TOTAL_SECONDS or budget.cpu > TOTAL_SECONDS:
            raise RuntimeError('final aggregate budget exhausted')
        return {'schema': 1, 'status': 'PUBLIC_ARM_QEMU_SYNTHETIC_PASS',
                'source_binding_sha256': binding, 'source_files': json.loads((M / 'source-binding.json').read_text())['files'],
                'tool_versions': versions, 'tool_sha256': tool_hashes,
                'sysroot': str(sysroot), 'loader_sha256': sha(loader),
                'fp_flags': guard.FLAGS, 'arm_flags': ARM_FLAGS, 'modes': modes,
                'commands': budget.commands, 'executions': executions, 'binary_executions': len(executions),
                'compiler_invocations': 20, 'affine_cases_per_mode': 648,
                'usage': {'wall_seconds': elapsed, 'child_cpu_seconds': budget.cpu, 'child_output_bytes': budget.output},
                'limits': {'total_seconds': TOTAL_SECONDS, 'program_seconds': PROGRAM_SECONDS, 'as_bytes': AS_BYTES, 'output_bytes': OUTPUT_BYTES},
                'model_loads': 0, 'model_forwards': 0, 'audio_reads': 0,
                'correctness_only': True, 'performance_claim': False, 'qualification_allowed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--emulator-sysroot', default='/usr/arm-linux-gnueabihf')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return
    try:
        emit(execute(args.emulator_sysroot))
    except Exception as exc:
        print('ARM synthetic check failed: ' + str(exc)[:4096], file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
