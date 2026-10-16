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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    require(compiler is not None, 'requested compiler is not installed')
    require(Path('/dev/full').is_char_device(), 'Linux /dev/full is required for this hosted contract')
    with tempfile.TemporaryDirectory(prefix='kws-stdout-contract-') as temp:
        root = Path(temp)
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
