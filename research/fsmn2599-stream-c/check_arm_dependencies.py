#!/usr/bin/env python3
"""Opt-in new ARM snapshot; does not execute B or historical numerical fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import check_arm_synthetic as arm

MAX_OBJECTS = 64
ARCHIVE_BYTES = 16 * 1024**2
SEARCH = ('lib', 'usr/lib', 'lib/arm-linux-gnueabihf', 'usr/lib/arm-linux-gnueabihf')


def identity(path):
    if not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise RuntimeError('missing or oversized identity input')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


class Closure:
    def __init__(self, sysroot, artifacts, inspect):
        self.sysroot = sysroot.resolve(strict=True)
        self.artifacts = artifacts.resolve(strict=True)
        self.inspect = inspect
        self.nodes = {}
        self.edges = []
        self.seen = {}
        self.active = set()
        self.resolutions = {}

    def label(self, path):
        # Lexical and real containment both matter: never publish a private path.
        for root, prefix in ((self.artifacts, '@artifact/'), (self.sysroot, '@sysroot/')):
            if path.is_relative_to(root) and path.resolve(strict=True).is_relative_to(root):
                return prefix + path.relative_to(root).as_posix()
        raise RuntimeError('dependency path escape')

    def track(self, path):
        label = self.label(path)
        real = path.resolve(strict=True)
        record = {'resolved': self.label(real), **identity(real)}
        if label in self.seen and self.seen[label] != record:
            raise RuntimeError('dependency identity changed')
        self.seen[label] = record
        return real

    def resolve(self, name):
        if not re.fullmatch(r'[A-Za-z0-9_+.-]+', name) or name in ('.', '..'):
            raise RuntimeError('unsafe dependency name')
        candidates = [self.artifacts / name] + [self.sysroot / d / name for d in SEARCH]
        found = []
        for path in candidates:
            if path.exists() or path.is_symlink():
                found.append(self.track(path))
        unique = set(found)
        if not unique:
            raise RuntimeError('missing dependency: ' + name)
        if len(unique) != 1:
            raise RuntimeError('ambiguous dependency: ' + name)
        result = unique.pop()
        if name in self.resolutions and self.resolutions[name] != result:
            raise RuntimeError('dependency resolution changed')
        self.resolutions[name] = result
        return result

    def visit(self, path):
        path = self.track(path)
        key = self.label(path)
        if key in self.active:
            raise RuntimeError('dependency cycle')
        if key in self.nodes:
            return key
        if len(self.nodes) >= MAX_OBJECTS:
            raise RuntimeError('dependency object limit')
        self.active.add(key)
        info = self.inspect(path)
        self.nodes[key] = {**identity(path), **info}
        for name in info['needed']:
            child = self.visit(self.resolve(name))
            required = set(info['version_needs'].get(name, []))
            if not required <= set(self.nodes[child]['version_definitions']):
                raise RuntimeError('unresolved symbol version: ' + name)
            self.edges.append({'from': key, 'needed': name, 'to': child})
        if info['interpreter']:
            interp = info['interpreter']
            if interp != '/lib/ld-linux-armhf.so.3':
                raise RuntimeError('unexpected interpreter')
            child = self.visit(self.sysroot / interp.lstrip('/'))
            self.edges.append({'from': key, 'interpreter': interp, 'to': child})
        self.active.remove(key)
        return key

    def verify(self):
        original = dict(self.seen)
        for name in list(self.resolutions):
            self.resolve(name)
        for label in original:
            root = self.artifacts if label.startswith('@artifact/') else self.sysroot
            self.track(root / label.split('/', 1)[1])
        if self.seen != original:
            raise RuntimeError('dependency identity changed')


def parse_elf(dynamic, program, versions, notes):
    if re.search(r'\((?:RPATH|RUNPATH|AUDIT|DEPAUDIT|FILTER|AUXILIARY)\)', dynamic):
        raise RuntimeError('unsupported dynamic loader directive')
    needed = re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', dynamic)
    if len(needed) != len(set(needed)):
        raise RuntimeError('duplicate DT_NEEDED')
    interpreter = re.findall(r'Requesting program interpreter: ([^\]]+)\]', program)
    if len(interpreter) > 1:
        raise RuntimeError('ambiguous interpreter')
    definitions, requirements = [], {}
    mode, library = None, None
    for line in versions.splitlines():
        if line.startswith('Version definition section'): mode = 'definitions'
        elif line.startswith('Version needs section'): mode = 'needs'
        elif line.startswith('Version symbols section'): mode = None
        if mode == 'needs':
            match = re.search(r'File: (\S+)\s+Cnt:', line)
            if match:
                library = match[1]
                requirements.setdefault(library, [])
            match = re.search(r'Name: (\S+)', line)
            if match:
                if library is None: raise RuntimeError('unbound version requirement')
                requirements[library].append(match[1])
        elif mode == 'definitions':
            match = re.search(r'Name: (\S+)', line)
            if match: definitions.append(match[1])
    if not set(requirements) <= set(needed):
        raise RuntimeError('version requirement has no dependency')
    return {'needed': needed, 'interpreter': interpreter[0] if interpreter else None,
            'version_needs': requirements, 'version_definitions': definitions,
            'build_ids': re.findall(r'Build ID: ([0-9a-f]+)', notes)}


def save(folder, report):
    encoded = json.dumps(report, indent=2) + '\n'
    if len(encoded.encode()) > arm.OUTPUT_BYTES:
        raise RuntimeError('report output limit')
    previous = folder / 'report.json'
    # Charge both old and pending reports at the atomic replacement peak.
    size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if size + len(encoded.encode()) > ARCHIVE_BYTES:
        raise RuntimeError('retained directory size limit')
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                         prefix='.report-', delete=False) as stream:
            pending = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, previous)
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def io_failure():
    # Storage may be unwritable. Do not promise another disk write or expose
    # exception paths; this stderr receipt is explicitly not durable evidence.
    print(json.dumps({'schema': 1, 'status': 'FAIL',
                      'retention_error': 'report I/O failure; disk receipt may be stale or durability unknown',
                      'qualification_allowed': False}), file=sys.stderr)


def _finalize(folder, report):
    """Always reserve a bounded failure receipt; never relabel truncation PASS."""
    try:
        save(folder, report)
    except RuntimeError:
        report['status'] = 'FAIL'
        report['retention_error'] = 'archive or report limit; partial evidence only'
        report['omitted_artifacts'] = []
        # This fresh directory is owned exclusively by this invocation. Retain
        # names and hashes before omitting generated bytes, never external inputs.
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.name != 'report.json':
                report['omitted_artifacts'].append({'path': '@artifact/' + path.relative_to(folder).as_posix(), **identity(path)})
                path.unlink()
        if len(json.dumps(report).encode()) > arm.OUTPUT_BYTES // 2:
            summary = {'schema': 1, 'status': 'FAIL', 'retention_error': report['retention_error'],
                       'historical_616': 'MISSING_ORIGINAL_ELF_SYSROOT',
                       'retrospective_certification': False, 'qualification_allowed': False,
                       'probe_attempts': report.get('probe_attempts', 0),
                       'full_report_sha256': hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
                       'omitted_artifact_count': len(report['omitted_artifacts'])}
            report.clear(); report.update(summary)
        save(folder, report)


def finalize(folder, report):
    try:
        _finalize(folder, report)
    except OSError:
        report['status'] = 'FAIL'
        raise


def execute(destination, sysroot, emulator):
    destination.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'status': 'INCOMPLETE', 'historical_616': 'MISSING_ORIGINAL_ELF_SYSROOT',
              'retrospective_certification': False, 'model_loads': 0, 'audio_reads': 0,
              'B_executions': 0, 'probe_attempts': 0, 'qualification_allowed': False,
              'limits': {'total_seconds': 300, 'command_seconds': 30, 'child_as_bytes': arm.AS_BYTES,
                         'output_bytes': arm.OUTPUT_BYTES, 'archive_bytes': ARCHIVE_BYTES,
                         'dependency_objects': MAX_OBJECTS, 'automatic_retries': 0}}
    save(destination, report)
    class SnapshotBudget(arm.Budget):
        def run(self, argv, kind):
            before = arm.guard.child_cpu()
            try:
                output = super().run(argv, kind)
            except Exception:
                self.cpu += arm.guard.child_cpu() - before
                report['failed_command_output_accounting'] = 'unknown; output limit still enforced by runner'
                raise
            # Raw readelf output is parsed below. Keep its hash, not a second
            # escaped copy consuming the bounded public JSON report.
            self.commands[-1].pop('output', None)
            self.commands[-1]['output_bytes'] = len(output.encode())
            return output
    budget = SnapshotBudget(destination)
    arm.guard.AS_LIMIT = arm.AS_BYTES
    closure = None
    aliases = {str(destination): '@artifact', str(sysroot): '@sysroot', str(arm.ROOT): '@source', str(emulator): '@tool/qemu-arm'}
    def clean(value):
        if isinstance(value, dict): return {clean(k): clean(v) for k, v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, str):
            for source, target in sorted(aliases.items(), key=lambda item: -len(item[0])):
                value = value.replace(source, target)
        return value
    tools, tool_ids = {}, {}
    try:
        if any(k.startswith(('LD_', 'QEMU_')) for k in os.environ):
            raise RuntimeError('loader/emulator overrides forbidden')
        report['source_binding_sha256'] = arm.guard.validate_sources(arm.ROOT)
        sysroot = sysroot.resolve(strict=True)
        aliases[str(sysroot)] = '@sysroot'
        if (sysroot / 'etc/ld.so.cache').exists() or (sysroot / 'etc/ld.so.preload').exists():
            raise RuntimeError('sysroot loader cache/preload unsupported')
        for directory in SEARCH:
            for name in ('glibc-hwcaps', 'tls', 'v7l', 'vfp', 'neon'):
                if (sysroot / directory / name).exists():
                    raise RuntimeError('sysroot hwcap search unsupported')
        for name in ('gcc', 'g++', 'readelf'):
            found = shutil.which('arm-linux-gnueabihf-' + name, path='/usr/bin:/bin')
            if not found: raise RuntimeError('required ARM tool missing: ' + name)
            tools[name] = Path(found).resolve(strict=True)
        tools['qemu-arm'] = emulator.resolve(strict=True)
        pin = json.loads((arm.M / 'qemu-i386-tool.json').read_text())
        arm.check_emulator(tools['qemu-arm'], pin)
        report['emulator_pin'] = pin
        # GCC delegates compilation: retain identities of those actual components too.
        for name, driver in (('cc1', 'gcc'), ('cc1plus', 'g++'), ('as', 'gcc'), ('ld', 'gcc'), ('collect2', 'gcc')):
            found = budget.run([str(tools[driver]), '-print-prog-name=' + name], 'tool-location').strip()
            path = Path(found) if '/' in found else Path(shutil.which(found, path='/usr/bin:/bin') or '/missing')
            tools[name] = path.resolve(strict=True)
        report['tools'] = {}
        for name, path in tools.items():
            aliases[str(path)] = '@tool/' + name
            tool_ids[name] = identity(path)
            report['tools'][name] = dict(tool_ids[name])
            report['tools'][name]['version'] = budget.run([str(path), '--version'], 'tool-version')
        def inspect(path):
            arm.check_elf(path)
            outputs = [budget.run([str(tools['readelf']), '-W', flag, str(path)], 'elf-inspect') for flag in ('-d', '-l', '-V', '-n')]
            return parse_elf(*outputs)
        closure = Closure(sysroot, destination, inspect)
        report['nodes'], report['edges'] = closure.nodes, closure.edges
        report['compiler_library_resolution'] = {}
        for name in ('libc.so.6', 'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'ld-linux-armhf.so.3'):
            driver = 'g++' if name == 'libstdc++.so.6' else 'gcc'
            found = budget.run([str(tools[driver]), '-print-file-name=' + name], 'compiler-library-location').strip()
            if not Path(found).is_absolute(): raise RuntimeError('compiler library unresolved: ' + name)
            resolved = closure.track(Path(found).resolve(strict=True))
            if resolved != closure.resolve(name):
                raise RuntimeError('compiler and snapshot sysroot differ: ' + name)
            report['compiler_library_resolution'][name] = closure.label(resolved)
        decoder = arm.ROOT / arm.guard.DECODER
        vendor = destination / 'include/vendor'
        vendor.mkdir(parents=True)
        # Public source copy, no external symlink in the retained directory.
        shutil.copyfile(arm.ROOT / arm.guard.FFT, vendor / 'fft_twiddles.h')
        include = ['-I'+str(arm.M), '-I'+str(decoder), '-I'+str(destination / 'include')]
        flags = [*arm.guard.FLAGS, *arm.ARM_FLAGS]
        report['c_compile_flags'] = flags
        cpp_flags = ['-std=c++11', '-Wall', '-Wextra', '-Werror', '-pedantic', *arm.ARM_FLAGS]
        report['cpp_compile_flags'] = cpp_flags
        report['emulator_args'] = ['-cpu', 'max', '-L', '@sysroot']
        sources = [str(arm.M / n) for n in ('b_stream_adapter.c', 'canonical_pcm.c', 'canonical_fbank.c', 'fsmn2599.c', 'sha256.c', 'softmax2599.c')] + [str(decoder / 'kws3.c')]
        report['roots'] = []
        for mode, defines in arm.MODES:
            library = destination / ('runtime-' + mode + '.so')
            extra = [str(arm.M / 'b_profile.c')] if '-DKWS_B_PROFILE' in defines else []
            budget.run([str(tools['gcc']), *flags, *defines, *include, '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-soname,'+library.name, *sources, *extra, '-lm', '-o', str(library)], 'compile-only')
            report['roots'].append(closure.visit(library))
            client = destination / ('cpp-' + mode)
            budget.run([str(tools['g++']), *cpp_flags, *defines, *include, str(arm.M / 'test_cpp_link.cpp'), '-L'+str(destination), '-l:'+library.name, '-lm', '-o', str(client)], 'compile-only')
            report['roots'].append(closure.visit(client))
        probe = destination / 'environment-probe'
        budget.run([str(tools['gcc']), *flags, str(arm.M / 'arm_environment_probe.c'), '-lm', '-ldl', '-o', str(probe)], 'compile-probe')
        probe_key = closure.visit(probe)
        report['roots'].append(probe_key)
        closure.verify()
        for name, path in tools.items():
            if identity(path) != tool_ids[name]: raise RuntimeError('tool identity changed')
        report['probe_attempts'] = 1
        save(destination, clean(report))
        output = budget.run([str(tools['qemu-arm']), '-cpu', 'max', '-L', str(sysroot), str(probe)], 'environment-probe-once')
        report['probe_output'] = output
        fields = {}
        loaded = set()
        for line in output.splitlines():
            key, value = line.split(' ', 1)
            if key == 'OBJECT':
                if value == 'linux-vdso.so.1': continue
                if not value.startswith('/') or '..' in Path(value).parts:
                    raise RuntimeError('unexpected probe object path')
                path = sysroot / value.lstrip('/')
                loaded.add(closure.label(closure.track(path)))
            elif key in fields: raise RuntimeError('duplicate probe field')
            else: fields[key] = value
        if set(fields) != {'FPSCR', 'ROUND', 'EXCEPT', 'HWCAP', 'HWCAP2', 'GLIBC'}:
            raise RuntimeError('incomplete probe fields')
        expected, pending = set(), [probe_key]
        while pending:
            node = pending.pop()
            for edge in closure.edges:
                if edge['from'] == node and edge['to'] not in expected:
                    expected.add(edge['to']); pending.append(edge['to'])
        if loaded != expected: raise RuntimeError('probe loaded-set differs from static closure')
        report['probe_environment'] = {**fields, 'fpscr_rounding_bits': (int(fields['FPSCR'], 16) >> 22) & 3,
                                       'flush_to_zero': bool(int(fields['FPSCR'], 16) & (1 << 24)),
                                       'default_nan': bool(int(fields['FPSCR'], 16) & (1 << 25)),
                                       'loaded_objects': sorted(loaded)}
        closure.verify()
        for name, path in tools.items():
            if identity(path) != tool_ids[name]: raise RuntimeError('tool identity changed')
        if arm.guard.validate_sources(arm.ROOT) != report['source_binding_sha256']:
            raise RuntimeError('source identity changed')
        if time.monotonic() - budget.start > arm.TOTAL_SECONDS: raise RuntimeError('total budget exhausted')
        report['status'] = 'NEW_SNAPSHOT_PASS'
    except Exception as exc:
        report['status'] = 'FAIL'
        # Do not publish exception paths; structured command evidence is sanitized below.
        report['error_type'] = type(exc).__name__
        report['error'] = clean(str(exc)) if isinstance(exc, RuntimeError) else 'input or tool failure'
    finally:
        report['commands'] = budget.commands
        report['usage'] = {'wall_seconds': time.monotonic() - budget.start, 'child_cpu_seconds': budget.cpu, 'child_output_bytes': budget.output}
        if closure: report['resolution_chain'] = closure.seen
        final_report = clean(report)
        finalize(destination, final_report)
        report['status'] = final_report['status']
    return report['status'] == 'NEW_SNAPSHOT_PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sysroot', type=Path, default=Path('/usr/arm-linux-gnueabihf'))
    parser.add_argument('--emulator', type=Path)
    args = parser.parse_args()
    if not args.execute:
        parser.print_help(); return 0
    if args.output is None or args.emulator is None:
        parser.error('--output and --emulator are required')
    try:
        return 0 if execute(args.output.absolute(), args.sysroot.absolute(), args.emulator.absolute()) else 1
    except OSError:
        io_failure()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
