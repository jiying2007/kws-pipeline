import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "training"):
    sys.path.insert(0, str(directory))
import wave

from speech_data_queue import CONTRACT_SHA256, TOKENIZER_SHA256
from speech_data_queue import capture, canonical_hash, evaluate, inspect_pcm16_wav
from weak_speech_dataset import load_examples


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.put('rights.txt', b'SYNTHETIC FIXTURE ONLY, no actual resource authorization')
        self.put('tokens.txt', b'<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\n')
        with wave.open(str(self.root / 'audio.wav'), 'wb') as w:
            w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\x01\x00' * 3200)
        self.audio = inspect_pcm16_wav(self.root / 'audio.wav')
        self.plan = {'schema': 'speech-data-queue-v1', 'call_budget': 4,
                     'keywords': ['你好小窝', '小窝小窝'], 'tokens_path': 'tokens.txt',
                     'tokens_sha256': self.digest('tokens.txt'), 'recognizers': [], 'items': []}
        for i, (model, family) in enumerate([('Qwen/Qwen3-ASR-0.6B', 'qwen3-asr'), ('FunAudioLLM/SenseVoiceSmall', 'sensevoice')]):
            dec = {'scope': 'single_decode_each_anonymous_audio_no_context', 'model_id': model, 'asset_lock_sha256': str(i + 1)*64}
            self.put(f'dec{i}.json', dec)
            self.plan['recognizers'].append({'id': f'asr{i}', 'family': family, 'model_id': model,
                'revision': 'fixture', 'model_sha256': str(i + 1)*64, 'config_sha256': canonical_hash(dec), 'blind': True})
        self.plan['items'] = [self.item('one')]
        self.events = [self.event(0, '你好，小窝。'), self.event(1, '你好小窝')]

    def put(self, path, data):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        (self.root / path).write_bytes(data)

    def digest(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def item(self, ident):
        return {'id': ident, 'source_kind': 'real_speech', 'intent_kind': 'positive', 'intended_text': '你好小窝',
                'lineage_roots': ['speaker:fixture'], 'split': 'train', 'recipe_sha256': '0'*64,
                'resources': [], 'rights': {'path': 'rights.txt', 'sha256': self.digest('rights.txt'), 'research_use': 'allowed'},
                'audio_path': 'audio.wav', **self.audio}

    def event(self, i, transcript, item='one'):
        r = self.plan['recognizers'][i]
        raw = {'schema': 'anonymous-two-asr-raw-v1', 'status': 'success', 'model_id': r['model_id'], 'revision': r['revision'],
               'outcome': {'raw_text': transcript, 'completeness': 'complete', 'quality_flags': []},
               'evidence': {'input_binding': {'raw_pcm16': {'sha256': self.audio['pcm_sha256']}, 'wav': {'sha256': self.audio['file_sha256']}},
                            'prompt_context': '', 'transcript_repair_applied': False,
                            'token_binding': {'raw_redecode_matches_capture': True, 'metadata_boundary_verified': True, 'lexical_unknown_present': False}}}
        raw['evidence'].update(decoded_skip_special_tokens='language Chinese<asr_text>' + transcript, token_ids=[1, 151645])
        decoded = '<|zh|><|EMO_UNKNOWN|><|Speech|><|woitn|>' + transcript
        raw['evidence']['token_evidence'] = {'decoded': decoded, 'token_ids': [1,2,3,4,5]}
        raw['evidence']['token_binding'].update(parser_contract_sha256=CONTRACT_SHA256,
            tokenizer={'sha256': TOKENIZER_SHA256}, raw_decoded_sha256=hashlib.sha256(decoded.encode()).hexdigest(),
            raw_token_ids=[1,2,3,4,5], metadata_token_ids=[1,2,3,4], lexical_token_ids=[5])
        if not transcript and i == 1:
            raw['outcome']['quality_flags'] = ['empty_transcript']
        name = f'{item}-{i}.json'; self.put(name, raw)
        return {'plan_sha256': canonical_hash(self.plan), 'item_id': item, 'operation': r['id'], 'attempt': 1,
                'call_state': 'called', 'outcome': 'success', 'binding': 'verified', 'blind': True,
                'pcm_sha256': self.audio['pcm_sha256'], 'raw_result_path': name, 'declaration_path': f'dec{i}.json',
                'raw_artifacts': [{'path': p, 'sha256': self.digest(p)} for p in [name, f'dec{i}.json']]}

    def run_case(self):
        for e in self.events:
            e['plan_sha256'] = canonical_hash(self.plan)
        return evaluate(self.plan, self.events, self.root)

    def test_agreement(self):
        row = self.run_case()['items'][0]
        self.assertEqual(row['machine_text'], '你好小窝')
        self.assertEqual(row['target_ids'], [1,2,3,4])
        self.assertFalse(row['human_gold'])

    def test_disagreement(self):
        self.events[1] = self.event(1, '你好小屋')
        self.assertEqual(self.run_case()['items'][0]['reason'], 'asr_disagreement')

    def test_intent_not_actual(self):
        self.events = [self.event(i, '你好') for i in range(2)]
        self.assertEqual(self.run_case()['items'][0]['actual_kind'], 'speech')

    def test_negative_keyword(self):
        self.plan['items'][0]['intent_kind'] = 'negative'
        self.assertEqual(self.run_case()['items'][0]['reason'], 'negative_contains_keyword')

    def test_oov(self):
        self.events = [self.event(i, '你好小屋') for i in range(2)]
        self.assertEqual(self.run_case()['items'][0]['reason'], 'transcript_outside_training_vocabulary')

    def test_blank_not_background(self):
        self.events = [self.event(i, '') for i in range(2)]
        self.assertNotEqual(self.run_case()['items'][0]['status'], 'machine_candidate')

    def test_completed_no_repeat(self):
        self.assertEqual(self.run_case()['proposed_tasks'], [])

    def test_budget(self):
        self.events = []; self.plan['call_budget'] = 1
        self.assertEqual(len(self.run_case()['proposed_tasks']), 1)

    def test_no_intent_in_asr_task(self):
        self.events = []
        for task in self.run_case()['proposed_tasks']:
            self.assertNotIn('intended_text', task); self.assertNotIn('recipe_sha256', task)

    def test_unknown_not_retry(self):
        self.events[1].update(call_state='unknown', outcome='infra_failure')
        report = self.run_case()
        self.assertEqual(report['proposed_tasks'], [])
        self.assertEqual(report['statistics']['unknown_calls_reserved'], 1)

    def test_infra_pre_call_retry(self):
        self.events[1].update(call_state='not_called', outcome='infra_failure')
        task = self.run_case()['proposed_tasks'][0]
        self.assertEqual(task['attempt'], 2)

    def test_cannot_retry_content(self):
        self.events[1]['outcome'] = 'content_failure'
        bad = copy.deepcopy(self.events[1]); bad['attempt'] = 2; self.events.append(bad)
        with self.assertRaises(ValueError): self.run_case()

    def test_cannot_relabel_model(self):
        self.plan['recognizers'][1]['family'] = 'fake-family'
        self.assertEqual(self.run_case()['items'][0]['reason'], 'primary_asr_incomplete')

    def test_raw_failure_not_vote(self):
        raw = json.loads((self.root/'one-1.json').read_bytes()); raw['status'] = 'failed_unbound'
        self.put('one-1.json', raw)
        self.events[1]['raw_artifacts'][0]['sha256'] = self.digest('one-1.json')
        self.events[1]['transcript'] = '你好小窝'
        self.assertEqual(self.run_case()['items'][0]['reason'], 'primary_asr_incomplete')

    def test_supplied_transcript_ignored(self):
        self.events[1]['transcript'] = '你好小屋'
        self.assertEqual(self.run_case()['items'][0]['machine_text'], '你好小窝')

    def test_resource_failure_quarantines(self):
        self.plan['items'][0]['resources'] = [{'path': 'absent', 'sha256': 'a'*64}]
        self.assertEqual(self.run_case()['items'][0]['reason'], 'preflight_unavailable')

    def test_rights_unknown(self):
        self.plan['items'][0]['rights']['research_use'] = 'unknown'
        self.assertEqual(self.run_case()['items'][0]['reason'], 'preflight_unavailable')

    def test_lineage_transitive(self):
        a, b = self.item('two'), self.item('three')
        a['lineage_roots'] = ['speaker:fixture', 'reference:two']; b['lineage_roots'] = ['reference:two']; b['split'] = 'test'
        self.plan['items'] += [a, b]
        report = self.run_case()
        self.assertTrue(all(r['reason'] == 'lineage_split_conflict' for r in report['items']))
        self.assertEqual(len({r['lineage_group'] for r in report['items']}), 1)

    def test_audio_mismatch(self):
        self.plan['items'][0]['pcm_sha256'] = 'f'*64
        self.assertEqual(self.run_case()['items'][0]['reason'], 'audio_integrity')

    def test_ci_and_public_source_record(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        for mode in ('', '-O ', '-OO '):
            self.assertIn('python3 -S -B ' + mode + 'tests/test_speech_data_queue.py', workflow)
        retention = json.loads((ROOT / 'research/consolidation/source-retention-2026-10-07.json').read_bytes())
        receipts = [r for r in retention['active_workflow_maintenance']
                    if r.get('id') == 'maintained-speech-data-queue-20261017']
        self.assertEqual(len(receipts), 1)
        # This is a historical publication receipt, not a new immutable runtime pin.
        # Future maintained source changes must not rewrite its old identities.
        recorded = {row['path'] for row in receipts[0]['files']}
        self.assertTrue({'tools/speech_data_queue.py', 'training/weak_speech_dataset.py',
                         'tests/test_speech_data_queue.py', '.github/workflows/ci.yml'} <= recorded)

    def test_missing_raw_sidecar_isolates_item(self):
        self.events[0]['raw_artifacts'].append({'path': 'missing.log', 'sha256': 'a'*64})
        self.assertEqual(self.run_case()['items'][0]['reason'], 'evidence_capture_incomplete')

    def test_tts_requires_receipt(self):
        self.plan['items'][0]['source_kind'] = 'tts'
        self.assertEqual(self.run_case()['items'][0]['reason'], 'generation_evidence_incomplete')

    def test_bad_item_does_not_block_good(self):
        bad = self.item('bad'); bad['audio_path'] = 'missing.wav'
        self.plan['items'].append(bad)
        report = self.run_case()
        self.assertEqual(report['items'][0]['status'], 'machine_candidate')
        self.assertEqual(report['items'][1]['reason'], 'audio_integrity')
        self.assertEqual(report['statistics']['effective_yield'], 0.5)

    def test_config_drift(self):
        self.plan['recognizers'][1]['config_sha256'] = 'e'*64
        self.assertEqual(self.run_case()['items'][0]['reason'], 'primary_asr_incomplete')

    def test_third_is_configuration_only(self):
        third = copy.deepcopy(self.plan['recognizers'][1])
        third.update(id='third', family='reserved-third', model_id='reserved-model', model_sha256='3'*64)
        self.plan['recognizers'].append(third)
        self.events[1] = self.event(1, '你好小屋')
        report = self.run_case()
        self.assertEqual(report['items'][0]['third_route_task']['operation'], 'third')
        self.assertFalse(report['items'][0]['third_route_task']['execution_authorized'])
        self.assertEqual(report['proposed_tasks'], [])

    def test_capture_and_loader(self):
        self.put('plan.json', self.plan); self.put('events.json', self.events)
        package = self.root/'package'
        capture(self.root/'plan.json', self.root/'events.json', self.root, package)
        examples = list(load_examples(package))
        self.assertEqual(len(examples), 1); self.assertEqual(examples[0][1], [1,2,3,4])
        self.assertEqual(len(examples[0][0][0]), 32)
        with self.assertRaises(FileExistsError): capture(self.root/'plan.json', self.root/'events.json', self.root, package)
        (package/'weak-train.jsonl').write_text('')
        with self.assertRaises(ValueError): list(load_examples(package))

    def test_raw_saved_before_bad_plan(self):
        self.plan['schema'] = 'bad'; self.put('plan.json', self.plan); self.put('events.json', self.events)
        package = self.root/'package'
        with self.assertRaises(ValueError): capture(self.root/'plan.json', self.root/'events.json', self.root, package)
        self.assertTrue((package/'raw-events.json').is_file())
        self.assertTrue((package/'artifact-000000.wav').is_file())


if __name__ == '__main__': unittest.main()
