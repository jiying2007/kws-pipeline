#!/usr/bin/env python3
"""Opt-in local research transport smoke. Never train or qualify a model.

Private archive/checkpoint bytes stay local. Public CI uses --self-test only.
An archive digest binds user-selected bytes, not authorship or execution trust.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import wave
import zipfile

LIMIT = 16 * 1024 * 1024
FLAGS = {'research_only': True, 'production_qualified': False,
         'promotion_allowed': False, 'shipping_approved': False}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def admit(archive, expected_archive, pack, expected_pack, tokens):
    """Verify all members without extracting or deserializing a checkpoint."""
    require(len(archive) <= LIMIT, 'archive size limit')
    require(digest(archive) == expected_archive, 'archive digest mismatch')
    require(digest(pack) == expected_pack, 'keyword pack digest mismatch')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        entries = z.infolist()
        require(len(entries) <= 64 and sum(x.file_size for x in entries) <= LIMIT,
                'archive member budget')
        require(len({x.filename for x in entries}) == len(entries), 'duplicate archive member')
        manifest = json.loads(z.read('PACKAGE-MANIFEST.json'))
        require(isinstance(manifest, dict), 'invalid manifest')
        require(set(manifest) == set(z.namelist()) - {'PACKAGE-MANIFEST.json'},
                'manifest member set mismatch')
        for name, item in manifest.items():
            data = z.read(name)
            require(len(data) == item['bytes'] and digest(data) == item['sha256'],
                    'manifest member digest mismatch')
        side = json.loads(z.read('candidate.research-sidecar.json'))
        require(all(side.get(k) is v for k, v in FLAGS.items()), 'research purpose required')
        model = z.read('candidate.kwm')
        checkpoint = z.read('candidate.research.pt')
        provenance = z.read('candidate.kwm.provenance.json')
        require(digest(model) == side['model']['sha256'], 'model binding mismatch')
        require(digest(checkpoint) == side['checkpoint']['sha256'], 'checkpoint binding mismatch')
        require(digest(provenance) == side['exporter_provenance']['sha256'],
                'provenance binding mismatch')
        prov = json.loads(provenance)
        require(digest(tokens) == prov['tokens']['sha256'] == prov['tokens']['checkpoint_sha256'],
                'token binding mismatch')
        require(len(model) >= 72 and model[:4] == b'KWSP' and
                struct.unpack_from('<H', model, 4)[0] == 2, 'expected KWSP ABI2 model')
        require(len(pack) >= 24 and pack[:4] == b'KWKP' and
                struct.unpack_from('<H', pack, 4)[0] == 3, 'expected KWKP ABI3 pack')
        require(struct.unpack_from('<Q', model, 40)[0] == struct.unpack_from('<Q', pack, 16)[0],
                'vocabulary fingerprint mismatch')
    return model, {'archive_sha256': expected_archive, 'model_sha256': digest(model),
                   'checkpoint_sha256': digest(checkpoint), 'keyword_pack_sha256': expected_pack,
                   'tokens_sha256': digest(tokens)}


def execute(args):
    out = args.output_root
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    receipt = {'schema': 'private-candidate-host-smoke-v1', 'status': 'NOT_RUN',
               'production_qualified': False, 'training_runs': 0,
               'live_runs': 0, 'replay_runs': 0, 'cases': [], 'invocations': []}

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def launch(command, label):
        remaining = 1800 - (time.monotonic() - started)
        require(remaining > 0, 'wall budget exceeded')
        argv = [str(x) for x in command]
        invocation = {'label': label, 'command': argv, 'status': 'reserved'}
        receipt['invocations'].append(invocation)
        save()
        call_started = time.monotonic()
        try:
            result = subprocess.run(argv, capture_output=True, timeout=min(30, remaining))
        except subprocess.TimeoutExpired as exc:
            (out / (label + '.stdout')).write_bytes(exc.stdout or b'')
            (out / (label + '.stderr')).write_bytes(exc.stderr or b'')
            invocation.update(status='timeout', elapsed_seconds=time.monotonic()-call_started)
            save()
            raise
        invocation.update(status='completed', returncode=result.returncode,
                          elapsed_seconds=time.monotonic()-call_started)
        (out / (label + '.stdout')).write_bytes(result.stdout)
        (out / (label + '.stderr')).write_bytes(result.stderr)
        save()
        require(result.returncode == 0, label + ': nonzero exit')
        return result.stdout

    save()
    try:
        require(args.archive.stat().st_size <= LIMIT and args.pack.stat().st_size <= LIMIT,
                'input size limit')
        archive = args.archive.read_bytes()
        pack_bytes = args.pack.read_bytes()
        model, identity = admit(archive, args.archive_sha256, pack_bytes,
                                args.pack_sha256, args.tokens.read_bytes())
        receipt['identity'] = identity
        receipt['harness_sha256'] = digest(Path(__file__).read_bytes())
        receipt['source_revision_label'] = args.source_revision
        # Snapshot admitted bytes: a path changed later cannot change the invocation.
        model_path, pack_path = out / 'candidate.kwm', out / 'keywords.kwk'
        model_path.write_bytes(model)
        pack_path.write_bytes(pack_bytes)
        wav_runner = args.wav_runner.resolve(strict=True)
        replay_runner = args.replay_runner.resolve(strict=True)
        receipt['runner_sha256'] = {n: digest(p.read_bytes()) for n, p in
                                   [('wav', wav_runner), ('replay', replay_runner)]}
        save()
        references = {}
        cases = [(n, b, 0) for n in (399, 400, 1201, 32000) for b in (160, 320)]
        cases += [(1201, b, 1) for b in (160, 320)]
        for n, block, repeat in cases:
            name = f'n{n}-b{block}-r{repeat}'
            audio, trace, stats, completion = [out / (name + suffix) for suffix in
                                               ('.wav', '.kwtr', '.stats.json', '.completion.json')]
            with wave.open(str(audio), 'wb') as stream:
                stream.setparams((1, 2, 16000, n, 'NONE', 'not compressed'))
                pcm = [int(6000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n)]
                stream.writeframes(struct.pack('<' + 'h' * n, *pcm))
            receipt['live_runs'] += 1
            save()
            live = launch([wav_runner, model_path, pack_path, audio, f'synthetic-{n}',
                           '--block-samples', block, '--trace-out', trace, '--stats-json', stats],
                          name + '-live')
            receipt['replay_runs'] += 1
            save()
            replay = launch([replay_runner, model_path, pack_path, trace, f'synthetic-{n}',
                             '--completion', completion], name + '-replay')
            s, c = json.loads(stats.read_bytes()), json.loads(completion.read_bytes())
            frames = max(0, 1 + (n - 400) // 320)
            require(s['processed_samples'] == n and s['processed_frames'] == frames,
                    'live frame/sample count mismatch')
            require(c['completion']['frames_consumed'] == frames, 'replay frame count mismatch')
            for field in ('decoder_hits', 'detections', 'refractory_suppressed'):
                require(s[field] == c['completion'][field], 'replay counter mismatch')
            require(live == replay, 'live/replay events mismatch')
            value = live, stats.read_bytes(), trace.read_bytes()
            if n in references:
                require(value == references[n], 'block/fresh-init mismatch')
            else:
                references[n] = value
            receipt['cases'].append({'id': name, 'samples': n, 'frames': frames,
                                     'detections': s['detections'], 'trace_sha256': digest(value[2]),
                                     'events_sha256': digest(live), 'stats_sha256': digest(value[1])})
            save()
        require(receipt['live_runs'] == receipt['replay_runs'] == 10, 'process budget mismatch')
        receipt['status'] = 'TRANSPORT_PASS_EFFECT_NOT_QUALIFIED'
    except BaseException as exc:
        receipt['status'] = 'FAILED' if receipt['live_runs'] else 'NOT_RUN'
        receipt['error_type'] = type(exc).__name__
        save()
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        save()
    print(json.dumps({'status': receipt['status'], 'live_runs': 10, 'replay_runs': 10}))


def self_test():
    """Artificial admission tests only; no executable or real model is loaded."""
    tokens = b'<blank>\na\n'
    model = bytearray(72); model[:4] = b'KWSP'; struct.pack_into('<H', model, 4, 2)
    pack = bytearray(24); pack[:4] = b'KWKP'; struct.pack_into('<H', pack, 4, 3)
    checkpoint = b'artificial checkpoint, never deserialized'
    prov = json.dumps({'tokens': {'sha256': digest(tokens), 'checkpoint_sha256': digest(tokens)}}).encode()
    side = dict(FLAGS, model={'sha256': digest(model)}, checkpoint={'sha256': digest(checkpoint)},
                exporter_provenance={'sha256': digest(prov)})
    def bundle(changes=None):
        members = {'candidate.kwm': bytes(model), 'candidate.research.pt': checkpoint,
                   'candidate.kwm.provenance.json': prov,
                   'candidate.research-sidecar.json': json.dumps(side).encode()}
        members.update(changes or {})
        manifest = {n: {'bytes': len(d), 'sha256': digest(d)} for n, d in members.items()}
        buff = io.BytesIO()
        with zipfile.ZipFile(buff, 'w') as z:
            for n, d in members.items(): z.writestr(n, d)
            z.writestr('PACKAGE-MANIFEST.json', json.dumps(manifest))
        return buff.getvalue()
    data = bundle()
    admit(data, digest(data), pack, digest(pack), tokens)
    bad_side = dict(side, promotion_allowed=True)
    bad_model = bytearray(model); bad_model[40] = 1
    negatives = [(data, '0'*64, pack, digest(pack), tokens),
                 (data, digest(data), pack, '0'*64, tokens),
                 (data, digest(data), pack, digest(pack), b'wrong tokens')]
    wrong_pack = bytearray(pack); wrong_pack[16] = 1
    negatives.append((data, digest(data), wrong_pack, digest(wrong_pack), tokens))
    for changes in ({'candidate.research-sidecar.json': json.dumps(bad_side).encode()},
                    {'candidate.research.pt': b'changed'}, {'candidate.kwm': bytes(bad_model)},
                    {'candidate.kwm.provenance.json': b'{}'}):
        bad = bundle(changes); negatives.append((bad, digest(bad), pack, digest(pack), tokens))
    for args in negatives:
        try: admit(*args)
        except (ValueError, KeyError): pass
        else: raise ValueError('invalid artificial input admitted')
    with tempfile.TemporaryDirectory(prefix='private-smoke-admission-') as temp:
        root = Path(temp)
        args = argparse.Namespace(output_root=root / 'output', archive=root / 'missing',
                                  pack=root / 'missing-pack')
        try:
            execute(args)
        except FileNotFoundError:
            pass
        else:
            raise ValueError('missing input admitted')
        before = (args.output_root / 'receipt.json').read_bytes()
        record = json.loads(before)
        require(record['status'] == 'NOT_RUN' and record['live_runs'] == 0,
                'missing input reported as execution')
        try:
            execute(args)
        except FileExistsError:
            pass
        else:
            raise ValueError('existing output accepted')
        require((args.output_root / 'receipt.json').read_bytes() == before, 'receipt overwritten')
    print('private candidate admission: 11 artificial cases passed; executions=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--execute', action='store_true')
    for name in ('archive', 'pack', 'tokens', 'wav-runner', 'replay-runner', 'output-root'):
        parser.add_argument('--' + name, type=Path)
    for name in ('archive-sha256', 'pack-sha256', 'source-revision'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.self_test:
        require(not args.execute, 'self-test cannot execute private input')
        self_test()
    elif not args.execute:
        print(json.dumps({'status': 'NOT_RUN', 'reason': 'explicit --execute required'}))
    else:
        for name in ('archive', 'pack', 'tokens', 'wav_runner', 'replay_runner', 'output_root',
                     'archive_sha256', 'pack_sha256', 'source_revision'):
            if getattr(args, name) is None:
                print(json.dumps({'status': 'NOT_RUN', 'reason': 'missing required input'}))
                parser.error('missing --' + name.replace('_', '-'))
        execute(args)
