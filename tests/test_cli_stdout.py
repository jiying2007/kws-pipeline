#!/usr/bin/env python3
"""Hosted CLI stdout error contracts, with test-only synthetic compute providers.

No model execution, real audio, downloads, installs, or historical receipt edits.
Input containers are invented here; this does not validate acoustic output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ('kws_wav', 'kws_board_bench', 'kws_feature_dump', 'kws_posterior_dump',
         'kws_decoder_replay', 'kws_decoder_path_replay', 'kws_trace_slice')
SHA = b'ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402'
FINGERPRINT = 0x316aea1a94b303bf

from prepare_frozen_decoder_source import FIXTURES, FROZEN_FILES, OVERLAY, stage_source


def require(value, message):
    if not value:
        raise RuntimeError(message)


def run(command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def fixtures(root):
    # Zero-filled tensor storage is parsed for ABI only and never evaluated.
    model = bytearray(6812)
    struct.pack_into('<4sHHHHHHIIIfffQIIIIII', model, 0,
                     b'KWSP', 2, 72, 32, 64, 5, 0, 16000, 400, 320,
                     1., 1., 1., FINGERPRINT, 72, 2120, 6216, 6472, 6792, 6812)
    (root / 'model.kwm').write_bytes(model)
    for name, seqs in (('one.kwk', [[1]]), ('dual.kwk', [[1, 2, 3, 4], [3, 4, 3, 4]])):
        data = struct.pack('<4sHHHHIQ', b'KWKP', 3, 24, len(seqs), 5,
                           24 + 48 * len(seqs), FINGERPRINT)
        for index, tokens in enumerate(seqs):
            data += struct.pack('<IfHBBBBH16H', index + 1, .55, len(tokens),
                                0, 0, 0, 0, 0, *(tokens + [0] * (16 - len(tokens))))
        (root / name).write_bytes(data)
    for name, tokens in (('one.kwtr', [1]), ('dual.kwtr', [1, 2, 3, 4])):
        data = struct.pack('<8sIHHIIIIQQ64s8x', b'KWTRACE1', 1, 5, 0,
                           16000, 400, 320, 0, FINGERPRINT, len(tokens), SHA)
        for index, token in enumerate(tokens):
            logits = [-8.] * 5
            logits[token] = 8.
            data += struct.pack('<QB7x5f', 400 + index * 320, 1, *logits)
        (root / name).write_bytes(data)
    with wave.open(str(root / 'audio.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, 400, 'NONE', 'not compressed'))
        wav.writeframes(b'\0' * 800)


def check_stats_input_alias(executable, root):
    """Actual CLI I/O with synthetic compute; never overwrite any input alias."""
    root.mkdir()
    inputs = ('model.kwm', 'one.kwk', 'audio.wav', 'timeline.tsv')
    cases = [(name, form, False) for name in inputs
             for form in ('same', 'normalized', 'symlink', 'hardlink')]
    cases += [('separate', form, False)
              for form in ('new', 'existing', 'symlink', 'hardlink')]
    cases += [('audio.wav', 'same', True), ('error', 'loop', False),
              ('error', 'not-directory', False)]
    failures = []
    expected_stdout = None
    for index, (target, form, trace) in enumerate(cases):
        case = root / f'{index:02d}-{target}-{form}'
        case.mkdir()
        fixtures(case)
        (case / 'timeline.tsv').write_text(
            'kws-afe-timeline-v1\n0\t400\t0\t0\t16\t0\t1\t0\t'
            + 'a' * 64 + '\n', encoding='ascii')
        before = {name: (case / name).read_bytes() for name in inputs}
        allowed = target == 'separate'
        if allowed:
            stats = case / 'stats.json'
            if form == 'existing':
                stats.write_bytes(b'old non-input stats\n')
            elif form in ('symlink', 'hardlink'):
                (case / 'unrelated.json').write_bytes(b'old unrelated stats\n')
                if form == 'symlink':
                    stats.symlink_to('unrelated.json')
                else:
                    stats.hardlink_to(case / 'unrelated.json')
        elif target == 'error':
            if form == 'loop':
                stats = case / 'loop'
                stats.symlink_to('loop')
            else:
                stats = case / 'audio.wav' / 'not-a-directory'
        elif form == 'same':
            stats = case / target
        elif form == 'normalized':
            (case / 'child').mkdir()
            stats = case / 'child' / '..' / target
        elif form == 'symlink':
            stats = case / 'stats-link'
            stats.symlink_to(target)
        else:
            stats = case / 'stats-link'
            stats.hardlink_to(case / target)
        command = [executable, case / 'model.kwm', case / 'one.kwk',
                   case / 'audio.wav', 'stats-alias-fixture', '--metadata-tsv',
                   case / 'timeline.tsv', '--stats-json', stats]
        if trace:
            command += ['--trace-out', case / 'trace.kwtr']
        result = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10)
        changed = [name for name, data in before.items()
                   if (case / name).read_bytes() != data]
        try:
            require(not changed, f'input bytes changed: {changed}')
            require((result.returncode == 0) == allowed,
                    f'unexpected exit {result.returncode}: {result.stderr!r}')
            if allowed:
                require(json.loads(stats.read_bytes())['schema_version'] == 1,
                        'stats output missing or invalid')
                if expected_stdout is None:
                    expected_stdout = result.stdout
                require(result.stdout == expected_stdout, 'non-alias event bytes changed')
            else:
                require(result.stdout == b'', 'alias/error was rejected after PCM output')
                require(not (case / 'trace.kwtr').exists(), 'rejection left a trace')
                if trace:
                    require(result.returncode == 1, 'trace exclusive-open behavior changed')
        except RuntimeError as exc:
            failures.append(f'{index}:{target}:{form}: {exc}')
    print(f'stats-input alias cases: {len(cases)} checked; failures={failures}')
    require(not failures, 'stats/input alias contract failed')


def check_historical_source_contracts(temp):
    """Test the helper on isolated copies; never mutate current or frozen input."""
    source = temp / 'historical-contract-input'
    pins = json.loads((ROOT / OVERLAY).read_bytes())['source_sha256']
    names = [*pins, OVERLAY, *(str(Path(FIXTURES) / name) for name in FROZEN_FILES)]
    for name in names:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    before = {name: (source / name).read_bytes() for name in names}
    output = temp / 'historical-contract-output'
    require(stage_source(source, output) == pins, 'historical pin set changed')
    require(all(hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
                for name, expected in pins.items()), 'staged historical source differs from pins')
    require(all((source / name).read_bytes() == data for name, data in before.items()),
            'historical staging modified source input')

    def reject(destination):
        try:
            stage_source(source, destination)
        except (OSError, ValueError):
            return
        raise RuntimeError('historical staging accepted an invalid input')

    reject(output)
    # All retained files are authoritative exact bytes, not a fallback that
    # silently rebases scientific pins to whichever current source is present.
    for index, name in enumerate(sorted(FROZEN_FILES)):
        path = source / FIXTURES / name
        original = path.read_bytes()
        path.write_bytes(original + b'\n/* fixture drift */\n')
        refused = temp / f'refused-fixture-{index}'
        reject(refused)
        require(not refused.exists(), 'fixture drift left partial output')
        path.write_bytes(original)
    path = source / 'src/frontend.c'
    original = path.read_bytes()
    path.write_bytes(original + b'\n/* unretained source drift */\n')
    refused = temp / 'refused-live-source'
    reject(refused)
    require(not refused.exists(), 'current source drift left partial output')
    path.write_bytes(original)
    outside = temp / 'outside-frontend.c'
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    refused = temp / 'refused-symlink-escape'
    reject(refused)
    require(not refused.exists(), 'source symlink escape left partial output')
    path.unlink()
    path.write_bytes(original)
    overlay_path = source / OVERLAY
    original = overlay_path.read_bytes()
    bad = json.loads(original)
    bad['source_sha256']['../outside.c'] = '0' * 64
    overlay_path.write_text(json.dumps(bad))
    refused = temp / 'refused-traversal'
    reject(refused)
    require(not refused.exists(), 'unsafe pin path left partial output')
    overlay_path.write_bytes(original)
    require(all((source / name).read_bytes() == data for name, data in before.items()),
            'historical negative tests did not restore their isolated inputs')


def check_trace_slice_cleanup(compiler, root):
    """Artificial traces only; failed opens must not acquire cleanup ownership.

    An open that truncates then fails writing its header may leave partial bytes:
    the writer API does not report ownership on that failure. This is not an
    atomic overwrite or general input/output-alias guarantee.
    """
    root.mkdir()
    wrapper = root / 'trace_io_faults.c'
    wrapper.write_text('''#include <errno.h>
#include <stdio.h>
#include <string.h>
static FILE *output_stream = NULL;
static FILE *test_open(const char *path, const char *mode) {
  FILE *stream;
  if (DENY_OPEN && strcmp(mode, "wb+") == 0) { errno = EACCES; return NULL; }
  stream = fopen(path, mode);
  if (strcmp(mode, "wb+") == 0) output_stream = stream;
  return stream;
}
static size_t test_write(const void *data, size_t size, size_t count, FILE *stream) {
  if (FAIL_OFFSET >= 0L && ftell(stream) == FAIL_OFFSET) { errno = EIO; return 0u; }
  return fwrite(data, size, count, stream);
}
static int test_close(FILE *stream) {
  if (stream == output_stream) {
    fputs("TRACE_SLICE_OUTPUT_CLOSED\\n", stderr);
    output_stream = NULL;
  }
  return fclose(stream);
}
#define fopen test_open
#define fwrite test_write
#define fclose test_close
''' + '#include ' + json.dumps(str(ROOT / 'tools/kws_trace_io.c')) + '\n')
    binaries = {}
    for name, deny, fail in (('normal', 0, -1), ('open', 1, -1),
                             ('header', 0, 8), ('append', 0, 120)):
        executable = root / name
        run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
             f'-DDENY_OPEN={deny}', f'-DFAIL_OFFSET={fail}L',
             ROOT / 'tools/kws_trace_slice.c', wrapper, '-lm', '-o', executable])
        binaries[name] = executable

    def trace(vocab):
        return (struct.pack('<8sIHHIIIIQQ64s8x', b'KWTRACE1', 1, vocab, 0,
                            16000, 400, 320, 0, FINGERPRINT, 1, SHA)
                + struct.pack('<QB7x', 400, 1) + struct.pack('<' + 'f' * vocab, *([0.] * vocab)))

    cases = (
        ('wide-existing', 257, 'normal', False, True, 400, False, 'preserve'),
        ('wide-same-path', 257, 'normal', True, False, 400, False, 'preserve'),
        ('wide-missing', 257, 'normal', False, False, 400, False, 'absent'),
        ('open-failure', 4, 'open', False, True, 400, False, 'preserve'),
        ('normal-new', 4, 'normal', False, False, 400, False, 'success'),
        ('normal-overwrite', 4, 'normal', False, True, 400, False, 'success'),
        ('read-failure', 4, 'normal', False, True, 400, True, 'absent'),
        ('append-failure', 4, 'append', False, True, 400, False, 'absent'),
        ('empty-selection', 4, 'normal', False, True, 1, False, 'absent'),
        ('header-new', 4, 'header', False, False, 400, False, 'partial'),
        ('header-overwrite', 4, 'header', False, True, 400, False, 'partial'),
    )
    failures = []
    for name, vocab, binary, same, existing, end, truncated, outcome in cases:
        case = root / name
        case.mkdir()
        source, output = case / 'input.kwtr', case / 'output.kwtr'
        original = trace(vocab)
        if truncated: original = original[:-1]
        source.write_bytes(original)
        if same: output = source
        elif existing: output.write_bytes(b'existing-output-sentinel\n')
        prior = output.read_bytes() if output.exists() else None
        result = subprocess.run([str(binaries[binary]), str(source), str(output), '0', str(end)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            require(result.stderr.count(b'TRACE_SLICE_OUTPUT_CLOSED\n') ==
                    int(vocab <= 256 and binary != 'open'),
                    name + ': output stream was not closed exactly once')
            require(source.exists() and source.read_bytes() == original,
                    name + ': input was removed or changed')
            require(result.returncode == (0 if outcome == 'success' else 1),
                    name + ': incorrect exit status')
            if outcome == 'success':
                require(output.read_bytes() == original, name + ': valid trace bytes changed')
                require(json.loads(result.stdout) == {'frames': 1, 'start_sample_exclusive': 0,
                                                       'end_sample_inclusive': 400}, name + ': receipt changed')
            else:
                require(not result.stdout and b'cannot slice acoustic trace' in result.stderr,
                        name + ': missing failure status')
                if outcome == 'preserve':
                    require(output.exists() and output.read_bytes() == prior,
                            name + ': unowned output was removed or changed')
                elif outcome == 'partial':
                    require(output.exists() and output.read_bytes() == b'KWTRACE1',
                            name + ': failed header-open must retain its partial bytes')
                else:
                    require(not output.exists(), name + ': failed output was retained')
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append(str(exc))
    print(f'trace-slice cleanup cases: {len(cases)} checked; failures={failures}')
    require(not failures, 'trace-slice cleanup contract failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    require(compiler is not None, 'requested compiler is not installed')
    require(Path('/dev/full').is_char_device(), 'Linux /dev/full is required for this hosted contract')
    with tempfile.TemporaryDirectory(prefix='kws-stdout-contract-') as temp:
        root = Path(temp)
        check_trace_slice_cleanup(compiler, root / 'trace-slice-cleanup')
        generated = root / 'generated'
        run([sys.executable, ROOT / 'tools/gen_parameter_limits.py',
             ROOT / 'configs/parameter-contract.json', generated])
        fixtures(root)
        check_historical_source_contracts(root)
        historical = root / 'historical-source'
        stage_source(ROOT, historical)
        support_names = ('src/kws.c', 'src/decoder.c', 'src/frontend.c',
                   'src/model.c', 'src/keyword_pack.c', 'tools/tool_io.c', 'tools/sha256.c',
                   'tools/kws_trace_io.c')
        support = [ROOT / name for name in support_names]
        historical_support = [historical / name for name in support_names]
        sys.path.insert(0, str(ROOT / 'research/near_pronunciation_v1'))
        from prepare_decoder import materialize
        derived = root / 'derived'
        materialize(historical, derived)
        def compile_tool(name, source, *, research=False):
            executable = root / name
            selected = ([derived / 'src/kws.c', derived / 'src/decoder.c', *historical_support[2:]]
                        if research else support)
            selected_root = historical if research else ROOT
            flags = ['-DKWS_RESEARCH_CHILD_ADMISSION=1'] if research else []
            if name.startswith('kws_bench'): flags.append('-DKWS_STDOUT_TEST_BENCH=1')
            run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                 '-Wconversion', '-Wshadow', '-Wcast-qual', *flags,
                 f'-DKWS_STDOUT_TEST_TOOL="{source}"', f'-I{derived / "src"}' if research else f'-I{ROOT / "src"}',
                 f'-I{selected_root / "include"}', f'-I{selected_root / "src"}',
                 f'-I{selected_root / "tools"}', f'-I{generated}',
                 ROOT / 'tests/cli_stdout_wrapper.c', *selected, '-lm', '-o', executable])
            return executable
        def command(name, executable):
            model, pack, audio = root / 'model.kwm', root / 'one.kwk', root / 'audio.wav'
            if name == 'kws_bench': return [executable]
            if name == 'kws_feature_dump': return [executable, audio, '1']
            if name == 'kws_posterior_dump': return [executable, model, audio, root / 'posterior.kwtr']
            if name == 'kws_trace_slice': return [executable, root / 'one.kwtr', root / 'slice.kwtr', '0', '400']
            if name in ('kws_decoder_replay', 'kws_decoder_path_replay'):
                return [executable, model, pack, root / 'one.kwtr', 'synthetic']
            if name.startswith('research-'):
                return [executable, model, root / 'dual.kwk', root / 'dual.kwtr', 'synthetic',
                        '--child-admission', name.removeprefix('research-')]
            if name == 'kws_board_bench': return [executable, model, pack, audio]
            return [executable, model, pack, audio, 'synthetic']
        cases = [(name, ROOT / 'tools' / (name + '.c'), False) for name in TOOLS]
        cases.append(('kws_bench', ROOT / 'bench/bench_kws.c', False))
        cases.extend((f'research-{mode}', derived / 'tools/kws_decoder_replay.c', True)
                     for mode in ('strict_children', 'single_fuzzy_wo'))
        for name, source, research in cases:
            executable = compile_tool(name, source, research=research)
            if name == 'kws_wav':
                check_stats_input_alias(executable, root / 'stats-input-alias')
            argv = command(name, executable)
            normal = run(argv, cwd=root)
            require(normal.stdout and not normal.stderr, f'{name}: missing normal output or unexpected stderr')
            if name == 'kws_bench':
                require(len(normal.stdout.splitlines()) == 6 and
                        all(line.startswith(b'case=') for line in normal.stdout.splitlines()),
                        'hosted benchmark did not emit all six synthetic cases')
            else:
                for line in normal.stdout.splitlines(): json.loads(line)
            with Path('/dev/full').open('wb') as full:
                failed = subprocess.run(list(map(str, argv)), stdout=full, stderr=subprocess.PIPE, cwd=root)
            require(failed.returncode == 1 and b'cannot write stdout' in failed.stderr,
                    f'{name}: did not report failed buffered stdout: {failed.returncode}, {failed.stderr!r}')
            if name == 'kws_posterior_dump':
                require(not (root / 'posterior.kwtr').exists(),
                        'posterior dump retained output after reported write failure')
            # A generated pre-fix negative control proves this fixture exercises
            # the original buffered-write hole, and checks unchanged good bytes.
            raw = source.read_text()
            if name in ('kws_trace_slice', 'kws_bench'):
                fixed = '  if (fflush(stdout) != 0 || ferror(stdout) != 0) {\n    fprintf(stderr, "cannot write stdout\\n");\n    return 1;\n  }\n  return 0;'
                original = ('  return 0;' if name == 'kws_bench' else
                            '  return ferror(stdout) != 0 ? 1 : 0;')
            else:
                fixed = '  if (fflush(stdout) != 0 || ferror(stdout) != 0) {\n    fprintf(stderr, "cannot write stdout\\n");\n    goto cleanup;\n  }\n  exit_code = 0;'
                original = '  exit_code = ferror(stdout) != 0 ? 1 : 0;'
            require(raw.count(fixed) == 1, f'{name}: stdout finalization contract changed')
            mutant = root / (name + '-pre-fix.c')
            mutant.write_text(raw.replace(fixed, original))
            previous = compile_tool(name + '-pre-fix', mutant, research=research)
            previous_argv = command(name, previous)
            old_normal = run(previous_argv, cwd=root)
            require(old_normal.stdout == normal.stdout, f'{name}: normal stdout bytes changed')
            with Path('/dev/full').open('wb') as full:
                old_failed = subprocess.run(list(map(str, previous_argv)), stdout=full,
                                            stderr=subprocess.PIPE, cwd=root)
            require(old_failed.returncode == 0 and not old_failed.stderr,
                    f'{name}: negative control did not reproduce original false success')
            if name == 'kws_trace_slice':
                with Path('/dev/full').open('wb') as full:
                    failed = subprocess.run([str(executable), '--self-test'], stdout=full,
                                            stderr=subprocess.PIPE, cwd=root)
                require(failed.returncode == 1 and b'cannot write stdout' in failed.stderr,
                        'trace-slice self-test ignored stdout failure')
            print(name + ': synthetic/stub stdout bytes unchanged; buffered write failure rejected')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
