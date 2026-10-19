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
        # Command bytes can expand under JSON escaping and appear in multiple
        # report fields. Never feed the oversized commands back into a FAIL.
        summary = {'schema': 1, 'status': 'FAIL',
                   'error': 'serialized evidence limit; incomplete receipt',
                   'full_report_bytes': len(encoded.encode()) + 1,
                   'full_report_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                   'command_attempts': len(report.get('commands', [])),
                   'execution_attempts': len(report.get('executions', [])),
                   'qualification_allowed': False, 'performance_claim': False}
        print(json.dumps(summary, separators=(',', ':')))
        return False
    print(encoded)
    return True


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


def check_emulator(path, pin):
    raw = path.read_bytes()
    if len(raw) != pin['binary_bytes'] or sha(path) != pin['binary_sha256']:
        raise RuntimeError('emulator differs from pinned official static i386 binary')
    if raw[:7] != b'\x7fELF\x01\x01\x01' or struct.unpack_from('<H', raw, 18)[0] != 3:
        raise RuntimeError('emulator must be ELF32 i386, not an ELF64 personality switch')
    offset = struct.unpack_from('<I', raw, 28)[0]
    size, count = struct.unpack_from('<HH', raw, 42)
    if size != 32 or not count or offset+size*count > len(raw):
        raise RuntimeError('invalid emulator program headers')
    if any(struct.unpack_from('<I', raw, offset+i*size)[0] == 3 for i in range(count)):
        raise RuntimeError('emulator must be static (no PT_INTERP)')


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


def execute(sysroot, emulator_path=None):
    started = time.monotonic()
    if any(k in os.environ for k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT')):
        raise RuntimeError('loader overrides forbidden')
    binding = guard.validate_sources(ROOT)
    tools = {}
    for name in ('arm-linux-gnueabihf-gcc', 'arm-linux-gnueabihf-nm', 'arm-linux-gnueabihf-readelf'):
        tools[name] = shutil.which(name, path='/usr/bin:/bin')
        if not tools[name]:
            raise RuntimeError('required tool missing (no skip): ' + name)
    if emulator_path is None:
        raise RuntimeError('explicit pinned i386 emulator path is required')
    pinned = json.loads((M / 'qemu-i386-tool.json').read_text())
    emulator_path = Path(emulator_path).resolve()
    check_emulator(emulator_path, pinned)
    tools['qemu-arm-i386'] = str(emulator_path)
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
        # True ELF32 host QEMU uses sparse guest mappings. Small -R reservations
        # are invalid for ARM's high commpage; ELF64 default reserves 4 GiB.
        # Keep the host RLIMIT_AS hard limit at 1 GiB; never rely on guest rlimits.
        emulator = [tools['qemu-arm-i386'], '-cpu', 'max', '-L', str(sysroot)]
        modes, executions = {}, []
        probe_source = tmp / 'startup.c'
        probe_source.write_text('#include <stdio.h>\nint main(void){puts("ARM loader startup PASS");return 0;}\n')
        probe = tmp / 'startup'
        budget.run(common + [str(probe_source), '-o', str(probe)], 'compile')
        budget.evidence['artifacts']['startup'] = check_elf(probe)
        header = budget.run([tools['arm-linux-gnueabihf-readelf'], '-l', str(probe)], 'startup-headers')
        if 'Requesting program interpreter:' not in header:
            raise RuntimeError('startup probe must exercise dynamic ARM loader')
        startup = {'mode': None, 'fixture': 'loader-startup', 'binary_sha256': sha(probe),
                   'source_sha256': sha(probe_source), 'command_index': len(budget.commands)}
        budget.evidence['executions'].append(startup)
        if budget.run([*emulator, str(probe)], 'startup-execution') != 'ARM loader startup PASS\n':
            raise RuntimeError('ARM dynamic loader startup probe failed')
        executions.append(startup)
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
            if 'Tag_CPU_arch: v8' not in attributes or 'Tag_ABI_VFP_args: VFP registers' not in attributes:
                raise RuntimeError('ARMv8 AArch32 VFP calling convention missing')
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
                'compiler_invocations': 21, 'synthetic_executions': 16,
                'startup_probe': startup, 'emulator_pin': pinned, 'affine_cases_per_mode': 648,
                'usage': {'wall_seconds': elapsed, 'child_cpu_seconds': budget.cpu, 'child_output_bytes': budget.output},
                'limits': {'total_seconds': TOTAL_SECONDS, 'program_seconds': PROGRAM_SECONDS, 'as_bytes': AS_BYTES, 'output_bytes': OUTPUT_BYTES},
                'model_loads': 0, 'model_forwards': 0, 'audio_reads': 0,
                'correctness_only': True, 'performance_claim': False, 'qualification_allowed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--emulator-sysroot', default='/usr/arm-linux-gnueabihf')
    parser.add_argument('--emulator', help='independently extracted pinned i386 static qemu-arm')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return
    try:
        if not emit(execute(args.emulator_sysroot, args.emulator)):
            return 1
    except Exception as exc:
        print('ARM synthetic check failed: ' + str(exc)[:4096], file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
