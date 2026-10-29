#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import argparse
import json
import os
import shlex
import pathlib
import stat
import subprocess
import sys
import tempfile
import wave
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import run_corpus
from run_corpus import cached_trace_valid, ensure_cached_trace, trace_cache_paths  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: pathlib.Path, seconds: int = 2) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000 * seconds)



class OutputPathTests(unittest.TestCase):
    def exercise_alias(self, output_name, target_name, alias_kind='direct', cache=False):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            files = {name: root / name for name in
                     ('runner', 'model', 'keywords', 'references', 'audio', 'metadata',
                      'detections', 'provenance', 'corpus-identity', 'posterior-dump', 'decoder-replay')}
            for name, path in files.items():
                path.write_bytes((name + '-original').encode())
            write_wav(files['audio'])
            files['references'].write_text(json.dumps({
                'recording': 'fixture', 'path': 'audio', 'duration_s': 2.0,
                'speaker_id': 's', 'session_id': 't', 'source_id': 'u', 'expected': [],
                **({'metadata_path': 'metadata'} if target_name == 'metadata' else {})
            }) + '\n')
            outputs = {name: files[name] for name in ('detections', 'provenance', 'corpus-identity')}
            target = files[target_name]
            if alias_kind == 'direct':
                outputs[output_name] = target
            else:
                alias = root / 'alias'
                if alias_kind == 'symlink':
                    alias.symlink_to(target)
                elif alias_kind == 'hardlink':
                    os.link(target, alias)
                elif alias_kind == 'normalized':
                    (root / 'sub').mkdir()
                    alias = root / 'sub' / '..' / target.name
                outputs[output_name] = alias
            argv = ['run_corpus', '--runner', str(files['runner']), '--model', str(files['model']),
                    '--keywords', str(files['keywords']), '--references', str(files['references']),
                    '--audio-root', str(root)]
            for name, path in outputs.items():
                argv += ['--' + name, str(path)]
            if cache:
                argv += ['--posterior-dump', str(files['posterior-dump']),
                         '--decoder-replay', str(files['decoder-replay']),
                         '--posterior-cache', str(root / 'cache')]
            before = {name: path.read_bytes() for name, path in files.items()}
            with patch.object(sys, 'argv', argv), patch.object(run_corpus.subprocess, 'run') as run:
                run.return_value.stdout = ''
                with self.assertRaisesRegex(ValueError, 'output path'):
                    run_corpus.main()
                run.assert_not_called()
            for name, path in files.items():
                self.assertEqual(path.read_bytes(), before[name], name)

    def test_outputs_cannot_alias_inputs(self):
        for output in ('detections', 'provenance', 'corpus-identity'):
            for target in ('references', 'model', 'keywords', 'runner', 'audio', 'metadata'):
                with self.subTest(output=output, target=target):
                    self.exercise_alias(output, target)

    def test_outputs_cannot_alias_each_other(self):
        for a, b in (('detections', 'provenance'), ('detections', 'corpus-identity'),
                     ('provenance', 'corpus-identity')):
            with self.subTest(a=a, b=b):
                self.exercise_alias(a, b)

    def test_filesystem_aliases(self):
        for kind in ('normalized', 'symlink', 'hardlink'):
            for target in ('references', 'audio', 'provenance'):
                with self.subTest(kind=kind, target=target):
                    self.exercise_alias('detections', target, kind)

    def test_cache_tree_outputs_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            cache = root / 'cache'
            cache.mkdir()
            link = root / 'cache-link'
            link.symlink_to(cache, target_is_directory=True)
            args = argparse.Namespace(references=root/'refs', model=root/'model',
                keywords=root/'kw', runner=root/'runner', posterior_dump=root/'dump',
                decoder_replay=root/'replay', posterior_cache=cache,
                provenance=None, corpus_identity=None)
            for path in (cache, cache/'new.bin', link/'new.bin'):
                args.detections = path
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'output path'):
                    run_corpus.validate_output_paths(args, [], True)
            self.assertEqual(list(cache.iterdir()), [])

    def test_cache_executables_are_protected(self):
        for target in ('posterior-dump', 'decoder-replay'):
            with self.subTest(target=target):
                self.exercise_alias('detections', target, cache=True)


class KeywordPackConsistencyTests(unittest.TestCase):
    """Artificial stdlib fixtures only; no model execution or qualification."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.files = {name: self.root / name for name in
                      ('runner', 'model', 'keywords', 'references', 'posterior-dump',
                       'decoder-replay', 'detections', 'provenance', 'corpus-identity', 'trace')}
        for name, path in self.files.items():
            path.write_bytes(('fixture-' + name).encode())
        self.files['keywords'].write_bytes(b'fixture-pack-v1')
        self.original_pack_sha = sha256_file(self.files['keywords'])
        for index in range(2):
            write_wav(self.root / f'{index}.wav')
        self.files['references'].write_text(''.join(json.dumps({
            'recording': f'r{index}', 'path': f'{index}.wav', 'duration_s': 2.0,
            'expected': []}) + '\n' for index in range(2)))
        self.before = {name: self.files[name].read_bytes() for name in
                       ('detections', 'provenance', 'corpus-identity')}
        self.calls = []
        self.executions = 0

    def invoke(self, *, cache=False, cache_hit=True, mutate=None, cache_mutate=None, provenance=True):
        self.executions = 0
        argv = ['run_corpus', '--audio-root', str(self.root)]
        names = ['runner', 'model', 'keywords', 'references', 'detections', 'corpus-identity']
        if provenance:
            names.append('provenance')
        if cache:
            names += ['posterior-dump', 'decoder-replay']
            argv += ['--posterior-cache', str(self.root / 'cache')]
        for name in names:
            argv += ['--' + name, str(self.files[name])]

        def execute(command, **kwargs):
            self.executions += 1
            pack = self.files['keywords'].read_bytes()
            self.calls.append(pack)
            if mutate:
                mutate(len(self.calls))
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
                'recording': command[4], 'keyword_id': 1 if pack.endswith(b'v1') else 2,
                'time_s': 1.0, 'confidence': 0.8}) + '\n')

        def cached(**kwargs):
            if cache_mutate:
                cache_mutate()
            return self.files['trace'], {'posterior_dump_sha256': kwargs['posterior_dump_sha256'],
                'trace_sha256': sha256_file(self.files['trace']), 'frames': 3}, cache_hit

        with patch.object(sys, 'argv', argv), \
                patch.object(run_corpus.subprocess, 'run', side_effect=execute), \
                patch.object(run_corpus, 'ensure_cached_trace', side_effect=cached):
            return run_corpus.main()

    def assert_outputs_preserved(self, present=True):
        for name, original in self.before.items():
            if present:
                self.assertEqual(self.files[name].read_bytes(), original, name)
            else:
                self.assertFalse(self.files[name].exists(), name)

    def test_stable_direct_and_cache_bind_original_pack(self):
        for cache, hit in ((False, False), (True, False), (True, True)):
            with self.subTest(cache=cache, hit=hit):
                self.calls.clear()
                self.assertEqual(self.invoke(cache=cache, cache_hit=hit), 0)
                self.assertEqual(self.calls, [b'fixture-pack-v1'] * 2)
                receipt = json.loads(self.files['provenance'].read_text())
                self.assertEqual(receipt['keyword_pack_sha256'], self.original_pack_sha)
                if cache:
                    self.assertEqual(receipt['posterior_cache_hits'], 2 if hit else 0)
                    self.assertEqual(receipt['posterior_cache_misses'], 0 if hit else 2)
                rows = [json.loads(line) for line in self.files['detections'].read_text().splitlines()]
                self.assertEqual([row['keyword_id'] for row in rows], [1, 1])

    def test_pack_may_change_between_independent_cached_runs(self):
        self.assertEqual(self.invoke(cache=True), 0)
        trace_sha = sha256_file(self.files['trace'])
        self.files['keywords'].write_bytes(b'fixture-pack-v2')
        self.calls.clear()
        self.assertEqual(self.invoke(cache=True), 0)
        self.assertEqual(self.calls, [b'fixture-pack-v2'] * 2)
        receipt = json.loads(self.files['provenance'].read_text())
        self.assertEqual(receipt['keyword_pack_sha256'], sha256_file(self.files['keywords']))
        self.assertNotEqual(receipt['keyword_pack_sha256'], self.original_pack_sha)
        self.assertEqual(receipt['posterior_cache_hits'], 2)
        self.assertEqual([trace['trace_sha256'] for trace in receipt['posterior_traces']], [trace_sha] * 2)

    def test_first_or_later_replay_pack_replacement_preserves_outputs(self):
        for cache in (False, True):
            for replace_at in (1, 2):
                for present in (False, True):
                    with self.subTest(cache=cache, replace_at=replace_at, present=present):
                        self.calls.clear()
                        self.files['keywords'].write_bytes(b'fixture-pack-v1')
                        for name, original in self.before.items():
                            if present:
                                self.files[name].write_bytes(original)
                            else:
                                self.files[name].unlink(missing_ok=True)
                        def mutate(call):
                            if call == replace_at:
                                self.files['keywords'].write_bytes(b'fixture-pack-v2')
                        with self.assertRaisesRegex(ValueError, 'keyword pack changed'):
                            self.invoke(cache=cache, mutate=mutate)
                        self.assertEqual(len(self.calls), replace_at)
                        self.assert_outputs_preserved(present)

    def test_same_bytes_replacement_is_allowed(self):
        def replace(call):
            replacement = self.root / 'same-pack'
            replacement.write_bytes(b'fixture-pack-v1')
            replacement.replace(self.files['keywords'])
        self.assertEqual(self.invoke(mutate=replace), 0)
        self.assertEqual(self.calls, [b'fixture-pack-v1'] * 2)
        receipt = json.loads(self.files['provenance'].read_text())
        self.assertEqual(receipt['keyword_pack_sha256'], self.original_pack_sha)

    def test_pack_change_before_replay_is_rejected(self):
        original = run_corpus.audio_identity
        def measure(*args):
            result = original(*args)
            self.files['keywords'].write_bytes(b'fixture-pack-v2')
            return result
        with patch.object(run_corpus, 'audio_identity', side_effect=measure):
            with self.assertRaisesRegex(ValueError, 'keyword pack changed'):
                self.invoke()
        self.assertEqual(self.calls, [])
        self.assert_outputs_preserved()

    def test_pack_change_during_cache_lookup_is_rejected_before_replay(self):
        with self.assertRaisesRegex(ValueError, 'keyword pack changed'):
            self.invoke(cache=True, cache_mutate=lambda:
                        self.files['keywords'].write_bytes(b'fixture-pack-v2'))
        self.assertEqual(self.calls, [])
        self.assert_outputs_preserved()

    def test_pack_change_before_publication_preserves_outputs(self):
        original = run_corpus.corpus_digest
        def digest(rows):
            result = original(rows)
            self.files['keywords'].write_bytes(b'fixture-pack-v2')
            return result
        with patch.object(run_corpus, 'corpus_digest', side_effect=digest):
            with self.assertRaisesRegex(ValueError, 'keyword pack changed'):
                self.invoke()
        self.assertEqual(len(self.calls), 2)
        self.assert_outputs_preserved()

    def test_missing_pack_fails_before_replay(self):
        self.files['keywords'].unlink()
        with self.assertRaises(OSError):
            self.invoke()
        self.assertEqual(self.executions, 0)
        self.assertEqual(self.calls, [])
        self.assert_outputs_preserved()

    def test_removed_pack_during_replay_preserves_outputs(self):
        with self.assertRaises(OSError):
            self.invoke(mutate=lambda call: self.files['keywords'].unlink())
        self.assertEqual(self.executions, 1)
        self.assertEqual(len(self.calls), 1)
        self.assert_outputs_preserved()

    def test_pack_read_failure_preserves_outputs(self):
        original = run_corpus.sha256_file
        fail = [False]
        def unreadable(path):
            if path == self.files['keywords'] and fail[0]:
                raise PermissionError('fixture pack I/O failure')
            return original(path)
        for initial in (True, False):
            with self.subTest(initial=initial):
                self.calls.clear()
                fail[0] = initial
                with patch.object(run_corpus, 'sha256_file', side_effect=unreadable):
                    with self.assertRaisesRegex(OSError, 'fixture pack I/O failure'):
                        self.invoke(mutate=lambda call: fail.__setitem__(0, True))
                self.assertEqual(self.executions, 0 if initial else 1)
                self.assert_outputs_preserved()

    def test_pack_drift_is_rejected_without_provenance_output(self):
        with self.assertRaisesRegex(ValueError, 'keyword pack changed'):
            self.invoke(provenance=False, mutate=lambda call:
                        self.files['keywords'].write_bytes(b'fixture-pack-v2'))
        self.assertEqual(len(self.calls), 1)
        self.assert_outputs_preserved()


class RecordingJsonlTests(unittest.TestCase):
    """Real C string helper and Python consumers; no model or inference."""

    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build.cleanup)
        root = pathlib.Path(cls.build.name)
        source = root / 'runner.c'
        cls.runner = root / 'runner'
        source.write_text(r'''#include "tool_io.h"
#include <stdio.h>
int main(int argc, char **argv) {
  if (argc != 5) return 2;
  fputs("{\"recording\":", stdout);
  kws_tool_print_json_string(stdout, argv[4]);
  fputs(",\"keyword_id\":1,\"time_s\":0.5,\"confidence\":0.8}\n", stdout);
  return fflush(stdout) != 0;
}
''', encoding='utf-8')
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        subprocess.run(compiler + ['-std=c11', '-Wall', '-Wextra', '-Werror',
            '-I', str(ROOT / 'tools'), '-I', str(ROOT / 'include'), str(source),
            str(ROOT / 'tools/tool_io.c'), '-o', str(cls.runner)],
            check=True, capture_output=True, timeout=30)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.files = {name: self.root / name for name in
            ('model', 'keywords', 'references', 'posterior-dump', 'trace',
             'detections', 'provenance', 'corpus-identity')}
        for name, path in self.files.items():
            path.write_bytes(('synthetic-not-a-model-' + name).encode())
        write_wav(self.root / 'audio.wav', seconds=1)

    def prepare(self, recording, *, cache=False):
        reference = {'recording': recording, 'path': 'audio.wav',
                     'duration_s': 1.0, 'expected': []}
        self.reference_bytes = (json.dumps(reference) + '\n').encode()
        self.files['references'].write_bytes(self.reference_bytes)
        self.sentinels = {}
        for name in ('detections', 'provenance', 'corpus-identity'):
            self.sentinels[name] = ('preserve-' + name).encode()
            self.files[name].write_bytes(self.sentinels[name])
        argv = ['run_corpus', '--runner', str(self.runner), '--audio-root', str(self.root)]
        for name in ('model', 'keywords', 'references', 'detections', 'provenance', 'corpus-identity'):
            argv += ['--' + name, str(self.files[name])]
        if cache:
            argv += ['--posterior-dump', str(self.files['posterior-dump']),
                     '--decoder-replay', str(self.runner),
                     '--posterior-cache', str(self.root / 'cache')]
        return argv

    def invoke(self, recording, *, cache=False, hit=False, transform=lambda value: value):
        argv = self.prepare(recording, cache=cache)
        real_run = subprocess.run

        def execute(command, **kwargs):
            self.assertEqual(command[4], recording)
            completed = real_run(command, **kwargs, timeout=30)
            completed.stdout = transform(completed.stdout)
            return completed

        def cached(**kwargs):
            return self.files['trace'], {
                'posterior_dump_sha256': kwargs['posterior_dump_sha256'],
                'trace_sha256': sha256_file(self.files['trace']), 'frames': 1}, hit

        with patch.object(sys, 'argv', argv), \
                patch.object(run_corpus.subprocess, 'run', side_effect=execute), \
                patch.object(run_corpus, 'ensure_cached_trace', side_effect=cached) as cache_call:
            self.assertEqual(run_corpus.main(), 0)
            self.assertEqual(cache_call.call_count, int(cache))
        self.assertEqual(self.files['references'].read_bytes(), self.reference_bytes)

    def verify_outputs(self, recording, count, *, cache=False, hit=False):
        import score_events
        data = self.files['detections'].read_bytes()
        # This actual downstream reader still uses splitlines; persisted JSONL
        # must remain safe for it without changing the decoded recording ID.
        rows = score_events.load_jsonl(self.files['detections'])
        self.assertEqual(len(rows), count)
        self.assertEqual([row['recording'] for row in rows], [recording] * count)
        expected = json.dumps({'recording': recording, 'keyword_id': 1,
                               'time_s': 0.5, 'confidence': 0.8}, ensure_ascii=False)
        for separator in ('\u0085', '\u2028', '\u2029'):
            expected = expected.replace(separator, f'\\u{ord(separator):04x}')
        self.assertEqual(data, ((expected + '\n') * count).encode())
        receipt = json.loads(self.files['provenance'].read_text())
        self.assertEqual(receipt['detections_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt['references_sha256'], hashlib.sha256(self.reference_bytes).hexdigest())
        self.assertEqual(receipt['detections'], count)
        self.assertEqual(receipt['audio_files'][0]['recording'], recording)
        if cache:
            self.assertEqual(receipt['posterior_cache_hits'], int(hit))
            self.assertEqual(receipt['posterior_cache_misses'], int(not hit))
        summary = self.root / 'score.json'
        scored = subprocess.run([sys.executable, '-S', '-B', str(ROOT / 'eval/score_events.py'),
            '--references', str(self.files['references']), '--detections', str(self.files['detections']),
            '--summary', str(summary)], capture_output=True, text=True, timeout=30)
        self.assertEqual(scored.returncode, 0, scored.stderr)
        self.assertEqual(json.loads(summary.read_text())['false_accepts'], count)
        self.assertEqual(self.files['references'].read_bytes(), self.reference_bytes)

    def test_helper_recording_roundtrip_direct_and_cache(self):
        values = ['plain', '普通中文', 'a"\\\b\f\n\r\t\x1cb', r'a\u2028b',
                  'a\u0085b', 'a\u2028b', 'a\u2029b', '中\u0085\u2028\u2029文']
        for recording in values:
            for cache, hit in ((False, False), (True, False), (True, True)):
                with self.subTest(recording=recording, cache=cache, hit=hit):
                    self.invoke(recording, cache=cache, hit=hit)
                    self.verify_outputs(recording, 1, cache=cache, hit=hit)

    def test_lf_crlf_blank_empty_and_multiple_records(self):
        transforms = [('lf', lambda value: value, 1),
                      ('no_final_lf', lambda value: value.rstrip('\n'), 1),
                      ('crlf', lambda value: value.replace('\n', '\r\n'), 1),
                      ('blank_lines', lambda value: '\r\n \t\n' + value + '\n', 1),
                      ('multiple', lambda value: value + value, 2),
                      ('empty', lambda value: '', 0)]
        recording = '中\u2028文'
        for name, transform, count in transforms:
            for cache in (False, True):
                with self.subTest(framing=name, cache=cache):
                    self.invoke(recording, cache=cache, hit=True, transform=transform)
                    self.verify_outputs(recording, count, cache=cache, hit=True)

    def test_invalid_output_preserves_existing_artifacts(self):
        bad = ['not-json\n', '[]\n', '{}\n',
               json.dumps({'recording': 'wrong'}) + '\n',
               '{"recording":"a\nb"}\n',
               json.dumps({'recording': 'expected'}) + '\nnot-json\n']
        for output in bad:
            for cache in (False, True):
                with self.subTest(output=output, cache=cache):
                    with self.assertRaisesRegex(ValueError, 'runner'):
                        self.invoke('expected', cache=cache, transform=lambda value: output)
                    for name, original in self.sentinels.items():
                        self.assertEqual(self.files[name].read_bytes(), original)
                    self.assertEqual(self.files['references'].read_bytes(), self.reference_bytes)

    def test_reference_source_boundary_is_not_broadened(self):
        for separator in ('\u0085', '\u2028', '\u2029'):
            with self.subTest(separator=separator):
                self.prepare('a' + separator + 'b')
                self.assertEqual(run_corpus.load_references(self.files['references'])[0]['recording'],
                                 'a' + separator + 'b')
                raw = json.dumps({'recording': 'a' + separator + 'b', 'path': 'audio.wav'},
                                 ensure_ascii=False) + '\n'
                self.files['references'].write_text(raw, encoding='utf-8')
                with self.assertRaises(ValueError):
                    run_corpus.load_references(self.files['references'])

    def test_full_cli_helper_to_scorer(self):
        recording = '中\u0085\u2028\u2029文'
        argv = self.prepare(recording)
        completed = subprocess.run([sys.executable, '-S', '-B', str(ROOT / 'eval/run_corpus.py'),
                                    *argv[1:]], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.verify_outputs(recording, 1)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        runner = root / "runner.py"
        model = root / "model.kwm"
        keywords = root / "keywords.kwk"
        references = root / "references.jsonl"
        audio = root / "audio.wav"
        detections = root / "detections.jsonl"
        provenance = root / "provenance.json"
        corpus = root / "corpus.json"

        runner.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "print(json.dumps({'recording': sys.argv[4], 'keyword_id': 1, "
            "'time_s': 1.0, 'confidence': 0.8}))\n",
            encoding="utf-8",
        )
        runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
        model.write_bytes(b"model-fixture")
        keywords.write_bytes(b"pack-fixture")
        write_wav(audio)
        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": "audio.wav",
                    "duration_s": 2.0,
                    "speaker_id": "spk-1",
                    "session_id": "session-1",
                    "source_id": "source-1",
                    "expected": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--provenance", str(provenance),
                "--corpus-identity", str(corpus),
            ]
        )

        detection_rows = [json.loads(line) for line in detections.read_text(encoding="utf-8").splitlines()]
        assert detection_rows == [
            {"recording": "room-1", "keyword_id": 1, "time_s": 1.0, "confidence": 0.8}
        ]
        result = json.loads(provenance.read_text(encoding="utf-8"))
        identity = json.loads(corpus.read_text(encoding="utf-8"))
        assert result["schema_version"] == 2
        assert result["runner_sha256"] == sha256_file(runner)
        assert result["model_sha256"] == sha256_file(model)
        assert result["keyword_pack_sha256"] == sha256_file(keywords)
        assert result["references_sha256"] == sha256_file(references)
        assert result["detections_sha256"] == sha256_file(detections)
        assert result["audio_corpus_sha256"] == identity["corpus_sha256"]
        assert result["audio_files"] == identity["recordings"]
        assert result["recordings"] == 1
        assert result["detections"] == 1

        posterior_dump = root / "posterior_dump.py"
        decoder_replay = root / "decoder_replay.py"
        posterior_cache = root / "posterior-cache"
        dump_count = root / "dump-count.txt"

        posterior_dump.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, json, os, pathlib, sys\n"
            "model=pathlib.Path(sys.argv[1]); audio=pathlib.Path(sys.argv[2]); trace=pathlib.Path(sys.argv[3])\n"
            "model_sha=hashlib.sha256(model.read_bytes()).hexdigest()\n"
            "audio_sha=hashlib.sha256(audio.read_bytes()).hexdigest()\n"
            "trace.write_bytes(('trace-v1:'+model_sha+':'+audio_sha).encode())\n"
            "trace_sha=hashlib.sha256(trace.read_bytes()).hexdigest()\n"
            "count=pathlib.Path(os.environ['DUMP_COUNT_FILE'])\n"
            "count.write_text(count.read_text()+'1\\n' if count.exists() else '1\\n')\n"
            "print(json.dumps({'schema_version':1,'evidence_class':'kws-posterior-trace-v1',"
            "'model_sha256':model_sha,'trace_sha256':trace_sha,'frames':3,'vocab_size':4}))\n",
            encoding="utf-8",
        )
        decoder_replay.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "pack=pathlib.Path(sys.argv[2]).read_bytes()\n"
            "keyword=2 if pack.endswith(b'v2') else 1\n"
            "if pathlib.Path(sys.argv[3]).read_bytes().startswith(b'trace-v2:'): keyword=3\n"
            "print(json.dumps({'recording':sys.argv[4],'keyword_id':keyword,"
            "'time_s':1.0,'confidence':0.8}))\n",
            encoding="utf-8",
        )
        posterior_dump.chmod(posterior_dump.stat().st_mode | stat.S_IXUSR)
        decoder_replay.chmod(decoder_replay.stat().st_mode | stat.S_IXUSR)
        env = dict(os.environ)
        env["DUMP_COUNT_FILE"] = str(dump_count)

        cached_provenance = root / "cached-provenance.json"
        cached_command = [
            sys.executable, str(ROOT / "eval" / "run_corpus.py"),
            "--runner", str(runner), "--model", str(model),
            "--keywords", str(keywords), "--references", str(references),
            "--audio-root", str(root), "--detections", str(detections),
            "--provenance", str(cached_provenance),
            "--posterior-dump", str(posterior_dump),
            "--decoder-replay", str(decoder_replay),
            "--posterior-cache", str(posterior_cache),
        ]
        first_dump_sha = sha256_file(posterior_dump)
        identity_args = {
            "model_sha256": sha256_file(model),
            "audio_sha256": sha256_file(audio),
            "posterior_dump_sha256": first_dump_sha,
        }
        # A valid pre-producer-identity cache must not be reused or relabelled.
        legacy_trace = (posterior_cache / identity_args["model_sha256"]
                        / identity_args["audio_sha256"][:2]
                        / (identity_args["audio_sha256"] + ".kwtr"))
        legacy_trace.parent.mkdir(parents=True)
        legacy_trace.write_bytes(b"trace-v2:unidentified-producer")
        legacy_summary = {
            "schema_version": 1,
            "evidence_class": "kws-posterior-trace-cache-v1",
            "model_sha256": identity_args["model_sha256"],
            "audio_sha256": identity_args["audio_sha256"],
            "trace_sha256": sha256_file(legacy_trace), "frames": 3, "vocab_size": 4,
        }
        legacy_sidecar = legacy_trace.with_suffix(".json")
        legacy_sidecar.write_text(json.dumps(legacy_summary), encoding="utf-8")
        assert cached_trace_valid(legacy_trace, legacy_sidecar, **identity_args) is None
        subprocess.check_call(cached_command, env=env)
        cached_first = json.loads(cached_provenance.read_text(encoding="utf-8"))
        cached_rows = [
            json.loads(line)
            for line in detections.read_text(encoding="utf-8").splitlines()
        ]
        assert cached_rows[0]["keyword_id"] == 1
        assert cached_first["evaluation_mode"] == "posterior-replay-cache-v1"
        assert cached_first["posterior_cache_hits"] == 0
        assert cached_first["posterior_cache_misses"] == 1
        assert len(cached_first["posterior_traces"]) == 1
        assert cached_first["posterior_dump_sha256"] == first_dump_sha
        assert cached_first["posterior_traces"][0]["posterior_dump_sha256"] == first_dump_sha
        first_trace, first_sidecar, _ = trace_cache_paths(posterior_cache, **identity_args)
        assert first_trace.parent.parent.name == first_dump_sha
        assert json.loads(first_sidecar.read_text())["posterior_dump_sha256"] == first_dump_sha
        assert legacy_trace.read_bytes() == b"trace-v2:unidentified-producer"
        first_trace_sha = cached_first["posterior_traces"][0]["trace_sha256"]
        assert dump_count.read_text(encoding="utf-8").splitlines() == ["1"]

        keywords.write_bytes(b"pack-fixture-v2")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--provenance", str(cached_provenance),
                "--posterior-dump", str(posterior_dump),
                "--decoder-replay", str(decoder_replay),
                "--posterior-cache", str(posterior_cache),
                "--decoder-state-retention", "0.91",
                "--decoder-refractory-ms", "250",
                "--decoder-blank-retention", "0.85",
                "--decoder-fuzzy-child-cost-log", "-4.0",
            ],
            env=env,
        )
        cached_second = json.loads(cached_provenance.read_text(encoding="utf-8"))
        cached_rows = [
            json.loads(line)
            for line in detections.read_text(encoding="utf-8").splitlines()
        ]
        assert cached_rows[0]["keyword_id"] == 2
        assert cached_second["posterior_cache_hits"] == 1
        assert cached_second["posterior_cache_misses"] == 0
        assert cached_second["decoder_replay_overrides"] == {
            "state_retention": 0.91,
            "refractory_ms": 250,
            "blank_retention": 0.85,
            "fuzzy_child_cost_log": -4.0,
        }
        assert cached_second["posterior_traces"][0]["trace_sha256"] == first_trace_sha
        assert dump_count.read_text(encoding="utf-8").splitlines() == ["1"]

        def check_cached_run(*, hits: int, misses: int, keyword_id: int) -> dict:
            subprocess.check_call(cached_command, env=env)
            result = json.loads(cached_provenance.read_text(encoding="utf-8"))
            producer_sha = sha256_file(posterior_dump)
            assert result["posterior_cache_hits"] == hits
            assert result["posterior_cache_misses"] == misses
            assert result["posterior_dump_sha256"] == producer_sha
            assert result["posterior_traces"][0]["posterior_dump_sha256"] == producer_sha
            assert json.loads(detections.read_text())["keyword_id"] == keyword_id
            return result

        # Decoder-only changes still reuse acoustic traces.
        decoder_replay.write_text(decoder_replay.read_text() + "# decoder-only change\n")
        decoder_changed = check_cached_run(hits=1, misses=0, keyword_id=2)
        assert decoder_changed["decoder_replay_sha256"] != cached_second["decoder_replay_sha256"]
        assert dump_count.read_text().splitlines() == ["1"]

        # Same executable path, different bytes: regenerate and identify the new producer.
        posterior_dump.write_text(posterior_dump.read_text().replace("trace-v1:", "trace-v2:"))
        second_dump_sha = sha256_file(posterior_dump)
        producer_changed = check_cached_run(hits=0, misses=1, keyword_id=3)
        assert second_dump_sha != first_dump_sha
        assert producer_changed["posterior_traces"][0]["trace_sha256"] != first_trace_sha
        assert first_trace.is_file() and first_sidecar.is_file()
        check_cached_run(hits=1, misses=0, keyword_id=3)
        assert dump_count.read_text().splitlines() == ["1", "1"]

        identity_args["posterior_dump_sha256"] = second_dump_sha
        _, sidecar, _ = trace_cache_paths(posterior_cache, **identity_args)
        for bad_producer in (None, first_dump_sha):
            summary = json.loads(sidecar.read_text())
            if bad_producer is None:
                summary.pop("posterior_dump_sha256")
            else:
                summary["posterior_dump_sha256"] = bad_producer
            sidecar.write_text(json.dumps(summary), encoding="utf-8")
            check_cached_run(hits=0, misses=1, keyword_id=3)
        # Even a legacy entry copied into the new namespace cannot be admitted.
        summary = json.loads(sidecar.read_text())
        summary.update(schema_version=1, evidence_class="kws-posterior-trace-cache-v1")
        sidecar.write_text(json.dumps(summary), encoding="utf-8")
        check_cached_run(hits=0, misses=1, keyword_id=3)
        assert len(dump_count.read_text().splitlines()) == 5

        # A stale caller snapshot must fail before a hit or generator invocation.
        try:
            ensure_cached_trace(
                posterior_dump=posterior_dump, cache_root=posterior_cache,
                model=model, audio=audio,
                **{**identity_args, "posterior_dump_sha256": first_dump_sha},
            )
        except ValueError as exc:
            assert "executable changed during this run" in str(exc)
        else:
            raise AssertionError("stale producer identity accepted")
        assert len(dump_count.read_text().splitlines()) == 5

        # Mutation while dumping must not publish a cache entry under the old identity.
        stable_dump_source = posterior_dump.read_text()
        posterior_dump.write_text(
            stable_dump_source.replace("os.environ['DUMP_COUNT_FILE']", repr(str(dump_count)))
            + "self=pathlib.Path(__file__); self.write_text(self.read_text()+'# changed\\n')\n"
        )
        drift_args = {**identity_args, "posterior_dump_sha256": sha256_file(posterior_dump)}
        try:
            ensure_cached_trace(
                posterior_dump=posterior_dump, cache_root=posterior_cache,
                model=model, audio=audio, **drift_args,
            )
        except ValueError as exc:
            assert "executable changed during trace generation" in str(exc)
        else:
            raise AssertionError("mid-generation producer mutation accepted")
        drift_trace, drift_sidecar, _ = trace_cache_paths(posterior_cache, **drift_args)
        assert not drift_trace.exists() and not drift_sidecar.exists()
        assert not list(drift_trace.parent.glob("*.tmp"))

        # A later executable replacement must not relabel a trace already consumed.
        posterior_dump.write_text(stable_dump_source)
        replay_source = decoder_replay.read_text()
        decoder_replay.write_text(
            replay_source + f"producer=pathlib.Path({str(posterior_dump)!r}); "
            "producer.write_text(producer.read_text()+'# replaced after replay\\n')\n"
        )
        subprocess.check_call(cached_command, env=env)
        consumed = json.loads(cached_provenance.read_text())
        assert consumed["posterior_cache_hits"] == 1
        assert sha256_file(posterior_dump) != second_dump_sha
        assert consumed["posterior_dump_sha256"] == second_dump_sha
        assert consumed["posterior_traces"][0]["posterior_dump_sha256"] == second_dump_sha
        posterior_dump.write_text(stable_dump_source)
        decoder_replay.write_text(replay_source)

        # Model and WAV identity remain separate invalidation dimensions.
        model.write_bytes(model.read_bytes() + b"-changed")
        check_cached_run(hits=0, misses=1, keyword_id=3)
        audio_bytes = audio.read_bytes()
        audio.write_bytes(audio_bytes[:-2] + b"\x02\x00")
        check_cached_run(hits=0, misses=1, keyword_id=3)
        check_cached_run(hits=1, misses=0, keyword_id=3)
        audio.write_bytes(audio_bytes)

        override_without_replay = subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--decoder-state-retention", "0.91",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert override_without_replay.returncode != 0
        assert "decoder replay overrides require posterior replay cache mode" in (
            override_without_replay.stderr + override_without_replay.stdout
        )

        original = audio.read_bytes()
        audio.write_bytes(original[:-2] + b"\x01\x00")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--provenance", str(provenance),
            ]
        )
        changed = json.loads(provenance.read_text(encoding="utf-8"))
        assert changed["audio_corpus_sha256"] != result["audio_corpus_sha256"]

        references.write_text(
            json.dumps({"recording": "room-1", "path": "audio.wav",
                        "duration_s": 3.0, "expected": []}) + "\n",
            encoding="utf-8",
        )
        invalid_duration = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "run_corpus.py"),
             "--runner", str(runner), "--model", str(model),
             "--keywords", str(keywords), "--references", str(references),
             "--audio-root", str(root), "--detections", str(detections)],
            check=False, text=True, capture_output=True,
        )
        assert invalid_duration.returncode == 2
        assert "reference duration_s does not match WAV duration" in invalid_duration.stderr

        # Identity admission validates numeric type, missing values and one-sample tolerance.
        sys.path.insert(0, str(ROOT / "eval"))
        from run_corpus import audio_identity
        row = {"recording": "room-1", "_execution_path": "audio.wav"}
        for duration in (2.0, 2.0 + 1.0 / 32000.0):
            assert audio_identity({**row, "duration_s": duration}, audio)["duration_s"] == 2.0
        for duration in (None, True, "2", float("nan"), float("inf"), -2,
                         2.0 + 2.0 / 16000.0):
            try:
                audio_identity({**row, "duration_s": duration}, audio)
            except ValueError as exc:
                assert "reference duration_s" in str(exc)
            else:
                raise AssertionError(duration)

        # Direct runner receives exactly the hash-bound metadata path. The
        # metadata-blind posterior cache must never silently strip this input.
        metadata = root / "timeline.tsv"
        metadata.write_text("synthetic-sidecar-v1\n", encoding="ascii")
        references.write_text(json.dumps({"recording": "room-1", "path": "audio.wav",
            "duration_s": 2.0, "expected": [], "metadata_path": metadata.name,
            "metadata_sha256": sha256_file(metadata)}) + "\n", encoding="utf-8")
        runner.write_text("#!/usr/bin/env python3\nimport sys\n"
                          "assert sys.argv[5:] == ['--metadata-tsv', " + repr(str(metadata)) + "]\n",
                          encoding="utf-8")
        command = [sys.executable, str(ROOT / "eval/run_corpus.py"),
                   "--runner", str(runner), "--model", str(model), "--keywords", str(keywords),
                   "--references", str(references), "--audio-root", str(root),
                   "--detections", str(detections), "--provenance", str(provenance)]
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        result = json.loads(provenance.read_text())
        assert result["afe_metadata_files"] == [{"recording": "room-1", "path": metadata.name,
                                                 "sha256": sha256_file(metadata)}]
        assert result["recalibration_required"] is True
        result = subprocess.run(command + ["--posterior-dump", str(posterior_dump),
            "--decoder-replay", str(decoder_replay), "--posterior-cache", str(posterior_cache)],
            check=False, text=True, capture_output=True)
        assert result.returncode == 2 and "not supported by the posterior cache" in result.stderr
        runner.write_text("#!/usr/bin/env python3\nimport pathlib, sys\n"
                          "pathlib.Path(sys.argv[6]).write_text('changed')\n", encoding="utf-8")
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        assert result.returncode == 2 and "metadata changed during replay" in result.stderr
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        assert result.returncode == 2 and "metadata SHA-256 mismatch" in result.stderr

    print("test_run_corpus: ok")
    return 0


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(main())
