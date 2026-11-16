"""Source/stdlib fixtures only. This file never compiles or executes C or a model."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import unittest

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1] if (HERE.parents[1] / 'src/kws.c').is_file() else HERE.parent / 'kws-pipeline'
COUNTS = {1: 3104, 2: 2432, 3: 96, 4: 8}
CLOCKS = tuple(range(16000, 48001, 1280))
ORIGINS = (0, 0, 48000)
NAMES = ('zero', 'sine400', 'lcg14')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def bits(value):
    need(type(value) in (int, float) and math.isfinite(value), 'FINITE_FLOAT')
    return struct.unpack('<I', struct.pack('<f', value))[0]


def scores_from_saved_trace(raw, origin):
    """Independent fixture reader; input authenticity belongs to the later artifact gate."""
    need(type(origin) is int and origin in ORIGINS, 'DECLARED_ORIGIN')
    need(len(raw) == 588232 and raw[:8] == b'EFFTR01\0', 'COMPLETE_FULLCHAIN_TRACE')
    rows, position = [], 8
    for index, sample in enumerate(CLOCKS):
        for stage, count in COUNTS.items():
            need(struct.unpack_from('<IIQ', raw, position) == (stage, count, sample), 'TRACE_HEADERS_AND_CLOCK')
            position += 16
            value = struct.unpack_from('<%df' % count, raw, position)
            need(all(math.isfinite(x) for x in value), 'FINITE_TRACE')
            if stage == 4:
                need(value[0] == float(index >= 15) and value[1] in (0., 1., 2.), 'READY_AND_EVENT_FIELDS')
                rows.append({'available_after_sample': sample + origin, 'ready': int(value[0]),
                    'keyword_id': int(value[1]),
                    'logit_bits': list(struct.unpack_from('<3I', raw, position + 8)),
                    'probability_bits': list(struct.unpack_from('<3I', raw, position + 20))})
            position += 4 * count
    need(position == len(raw), 'TRACE_END')
    return rows


def validate_fixture_report(report, references):
    """Check complete output without executing producer code; never a provenance/admission decision."""
    need(report.get('schema') == 'english-stream-integration-v1' and report.get('passed') is True and
         report.get('new_ort_inference_calls') == 0, 'REPORTED_COMPLETE_INTEGRATION')
    need(len(report['cases']) == len(references) == 3, 'THREE_CASES')
    for index, (case, raw) in enumerate(zip(report['cases'], references)):
        name, origin = NAMES[index], ORIGINS[index]
        need(case['recording'] == name and case['input_origin_sample'] == origin and case['passed'] is True and
             case['accepted_samples'] == 48000 and case['compared_scores'] == 26, 'COMPLETE_CASE')
        expected = scores_from_saved_trace(raw, origin)
        need(case['scores'] == expected, 'EVERY_HEAD_SCORE_AND_CLOCK')
        need(type(case['accept_wall_seconds']) in (int, float) and
             math.isfinite(case['accept_wall_seconds']) and case['accept_wall_seconds'] >= 0, 'MEASURED_FINITE_TIME')
        expected_events = [row for row in expected if row['keyword_id']]
        need(len(case['events']) == len(expected_events), 'EVENT_COUNT')
        for event, score in zip(case['events'], expected_events):
            need(event['recording'] == name and event['keyword_id'] == score['keyword_id'] and
                 event['available_after_sample'] == score['available_after_sample'] and
                 event['input_origin_sample'] == origin and
                 event['time_s'] == (score['available_after_sample'] - origin) / 16000 and
                 bits(event['confidence']) == score['probability_bits'][score['keyword_id']], 'EVENT_CALLBACK_CONTENT')
        need(case['cumulative_embeddings'] == 26 * (index + 1) and
             case['cumulative_ready_outputs'] == 11 * (index + 1) and
             case['next_input_sample'] == origin + 48000 and case['poisoned'] == 0, 'ACTUAL_COUNTERS')
    return True


def invented_trace(with_event=True):
    out = bytearray(b'EFFTR01\0')
    for index, sample in enumerate(CLOCKS):
        for stage, count in COUNTS.items():
            value = [0.] * count
            if stage == 4 and index >= 15:
                value = [1., float(with_event and index == 17), -0., .2, -.1, 1., 0., 0.]
                if with_event and 15 <= index <= 17:
                    value[5:] = [.05, .9, .05]
            out.extend(struct.pack('<IIQ', stage, count, sample))
            out.extend(struct.pack('<%df' % count, *value))
    return bytes(out)


def invented_report(raw):
    cases = []
    for index, (name, origin) in enumerate(zip(NAMES, ORIGINS)):
        scores = scores_from_saved_trace(raw, origin)
        events = [{'recording': name, 'keyword_id': r['keyword_id'],
            'available_after_sample': r['available_after_sample'], 'input_origin_sample': origin,
            'time_s': (r['available_after_sample'] - origin) / 16000,
            'confidence': struct.unpack('<f', struct.pack('<I', r['probability_bits'][r['keyword_id']]))[0]}
            for r in scores if r['keyword_id']]
        cases.append({'recording': name, 'passed': True, 'input_origin_sample': origin,
            'accepted_samples': 48000, 'accept_wall_seconds': .25, 'compared_scores': 26,
            'scores': scores, 'events': events, 'cumulative_embeddings': 26 * (index + 1),
            'cumulative_ready_outputs': 11 * (index + 1), 'next_input_sample': origin + 48000, 'poisoned': 0})
    return {'schema': 'english-stream-integration-v1', 'passed': True, 'cases': cases, 'new_ort_inference_calls': 0}


class StreamSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = invented_trace()
        cls.core = (HERE / 'ekws_research_stream.c').read_text()
        cls.header = (HERE / 'ekws_research_stream.h').read_text()
        cls.runner = (HERE / 'ekws_pcm_stream.c').read_text()
        cls.cmake = (HERE / 'CMakeLists.txt').read_text()
        cls.plan = json.loads((HERE / 'integration_plan.json').read_bytes())

    def test_disabled_actual_evidence_and_separate_model_abi(self):
        pins = json.loads((HERE / 'integration_pins.json').read_bytes())
        self.assertFalse(pins['enabled'])
        for key in ('source_commit', 'next_main_before', 'profile_source_sha256'):
            self.assertIsNone(pins[key])
        self.assertTrue(all(v is None for k, v in pins['fullchain'].items() if k != 'run_attempt'))
        self.assertEqual(pins['plan_sha256'], hashlib.sha256(canonical(self.plan)).hexdigest())
        self.assertFalse(self.plan['legacy_kwsp_kwkp_compatibility'])
        self.assertEqual(self.plan['model_format'], 'EHKWS01')
        self.assertEqual(self.plan['execution_repository'], 'jiying2007/kws-data')
        self.assertEqual(self.plan['component_repository'], 'jiying2007/kws-pipeline')

    def test_budget_and_third_discontinuity_are_predeclared(self):
        p = self.plan
        self.assertEqual(p['inference_processes_per_backend'], 1)
        self.assertEqual(p['build_sessions_per_backend'], 1)
        self.assertEqual(p['embeddings_per_backend'], 78)
        self.assertEqual(p['ready_outputs_per_backend'], 33)
        self.assertEqual(p['accepted_samples_per_backend'], 144000)
        self.assertEqual([c['input_origin_sample'] for c in p['cases']], list(ORIGINS))
        self.assertEqual(p['cases'][2]['boundary'], 'discontinuity-route-change')
        for key in ('new_ort_inference_calls', 'optimizer_updates', 'tts_asr_phone_calls', 'eof_padding_samples',
                    'separate_benchmark_processes', 'separate_inference_consumer_processes', 'real_audio_cases'):
            self.assertEqual(p[key], 0, key)

    def test_frozen_kernel_and_head_source_bytes_match_cmake(self):
        variables = {'EKWS_MEL': PIPELINE / 'research/english_mel_specialization/fixed_frontend_mel.c',
            'EKWS_FRONTEND': PIPELINE / 'research/english_final_embedding_frontend',
            'EKWS_HEAD': PIPELINE / 'research/english_final_embedding', 'EKWS_PIPELINE_SOURCE': PIPELINE}
        matched = 0
        for name, digest in re.findall(r'ekws_exact_file\("([^\"]+)" "([0-9a-f]{64})"\)', self.cmake):
            for variable, prefix in variables.items():
                token = '${' + variable + '}'
                if name.startswith(token):
                    path = Path(str(prefix) + name[len(token):])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, str(path)); matched += 1
        self.assertEqual(matched, 11)  # mel/header, all 8 head files, repository license
        self.assertIn('9bf740a692370dfed02f034f87f6f0dfcaac42f3f83152c79fd27088ccc71941', self.cmake)
        self.assertIn('c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4', self.cmake)

    def test_public_api_implemented_and_math_is_delegated_once(self):
        names = set(re.findall(r'(?:int|size_t)\s+(ekws_\w+)\s*\(', self.header))
        definitions = set(re.findall(r'(?:int|size_t)\s+(ekws_\w+)\s*\([^;]+?\)\s*\{', self.core, re.S))
        self.assertEqual(names, definitions)
        self.assertEqual(len(re.findall(r'\behead_step\s*\(', self.core)), 1)
        self.assertEqual(len(re.findall(r'\bef_frontend_feed\s*\(', self.core)), 1)
        self.assertNotRegex(self.core, r'\b(malloc|calloc|realloc|free|printf|fopen|fwrite|pthread_create|expf)\s*\(')
        self.assertNotIn('0.80', self.core)
        self.assertIn('one call may emit several events', self.header)
        self.assertIn('Reusing workspace as an input buffer is invalid', self.header)

    def test_reset_discontinuity_poison_and_reentry_are_explicit(self):
        self.assertIn('ef_frontend_reset(&s->frontend); ehead_reset(&s->head);', self.core)
        self.assertIn('ef_frontend_discontinuity(&s->frontend, next)', self.core)
        self.assertIn('ehead_discontinuity(&s->head)', self.core)
        self.assertGreaterEqual(self.core.count('if (s->busy) return EKWS_REENTRANT;'), 4)
        self.assertIn('if (s->poisoned) return EKWS_POISONED;', self.core)
        self.assertIn('s->stats.counters_saturated = 1;', self.core)
        self.assertLess(self.core.index('add(s, &s->stats.accepted_samples'), self.core.index('if (result != EF_OK)'))

    def test_default_off_install_namespace_and_main_patch(self):
        self.assertIn('option(KWS_ENABLE_ENGLISH_RESEARCH_STREAM "Build the separate unqualified English research stream" OFF)', self.cmake)
        self.assertIn('NAMESPACE KwsEnglishResearch::', self.cmake)
        self.assertIn('EXPORT_NAME stream', self.cmake)
        self.assertIn('PUBLIC m', self.cmake)
        self.assertIn('install(TARGETS ekws_pcm_stream', self.cmake)
        self.assertIn('KwsEnglishResearch/licenses', self.cmake)
        self.assertNotIn('add_test(', self.cmake)
        self.assertNotIn('try_run(', self.cmake)
        self.assertNotIn('execute_process(', self.cmake)
        patch = (HERE / 'main-cmake-opt-in.patch').read_text().splitlines()
        body = patch[next(i for i, line in enumerate(patch) if line.startswith('@@')) + 1:]
        before = '\n'.join(line[1:] for line in body if line.startswith((' ', '-'))) + '\n'
        after = '\n'.join(line[1:] for line in body if line.startswith((' ', '+'))) + '\n'
        main = (PIPELINE / 'CMakeLists.txt').read_text()
        self.assertTrue(before in main or after in main, 'main patch must match current original or already integrated source')
        self.assertIn('add_subdirectory(research/english_stream_integration)', after)
        self.assertNotIn('KwsPipeline::core', after)

    def test_all_score_fields_match_with_declared_absolute_origin(self):
        report = invented_report(self.raw)
        self.assertTrue(validate_fixture_report(report, [self.raw] * 3))
        self.assertEqual(report['cases'][2]['scores'][0]['available_after_sample'], 64000)
        self.assertEqual(report['cases'][2]['scores'][15]['available_after_sample'], 83200)
        self.assertEqual(report['cases'][2]['scores'][-1]['available_after_sample'], 96000)

    def test_empty_events_cannot_hide_missing_scores(self):
        raw = invented_trace(with_event=False); report = invented_report(raw)
        self.assertTrue(validate_fixture_report(report, [raw] * 3))
        report['cases'][0]['scores'] = []
        with self.assertRaisesRegex(ValueError, 'EVERY_HEAD_SCORE'):
            validate_fixture_report(report, [raw] * 3)

    def test_signed_zero_and_a_single_probability_bit_are_required(self):
        for section, value in (('logit_bits', 0), ('probability_bits', None)):
            report = invented_report(self.raw)
            row = report['cases'][1]['scores'][15][section]
            row[0] = value if value is not None else row[0] ^ 1
            with self.assertRaisesRegex(ValueError, 'EVERY_HEAD_SCORE'):
                validate_fixture_report(report, [self.raw] * 3)

    def test_wrong_clock_ready_or_event_is_not_a_pass(self):
        for key, value in (('available_after_sample', 35200), ('ready', 0), ('keyword_id', 2)):
            report = invented_report(self.raw); report['cases'][2]['scores'][15][key] = value
            with self.assertRaisesRegex(ValueError, 'EVERY_HEAD_SCORE'):
                validate_fixture_report(report, [self.raw] * 3)

    def test_event_callback_and_relative_time_are_independent_checks(self):
        for key, value in (('time_s', 9.), ('input_origin_sample', 0), ('confidence', .81), ('keyword_id', 2)):
            report = invented_report(self.raw); report['cases'][2]['events'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'EVENT_CALLBACK_CONTENT'):
                validate_fixture_report(report, [self.raw] * 3)

    def test_missing_partial_or_nonfinite_timing_cannot_pass(self):
        report = invented_report(self.raw); report['cases'] = report['cases'][:2]
        with self.assertRaisesRegex(ValueError, 'THREE_CASES'):
            validate_fixture_report(report, [self.raw] * 3)
        for timing in (float('nan'), float('inf'), -1.):
            report = invented_report(self.raw); report['cases'][0]['accept_wall_seconds'] = timing
            with self.assertRaisesRegex(ValueError, 'MEASURED_FINITE_TIME'):
                validate_fixture_report(report, [self.raw] * 3)
        report = invented_report(self.raw); report['cases'][1]['passed'] = False
        with self.assertRaisesRegex(ValueError, 'COMPLETE_CASE'):
            validate_fixture_report(report, [self.raw] * 3)

    def test_reference_rejects_header_truncation_or_nonfinite(self):
        for raw in (self.raw[:-1], b'WRONG001' + self.raw[8:]):
            with self.assertRaisesRegex(ValueError, 'COMPLETE_FULLCHAIN_TRACE'):
                scores_from_saved_trace(raw, 0)
        changed = bytearray(self.raw); struct.pack_into('<Q', changed, 16, 16001)
        with self.assertRaisesRegex(ValueError, 'TRACE_HEADERS_AND_CLOCK'):
            scores_from_saved_trace(changed, 0)
        changed = bytearray(self.raw); struct.pack_into('<I', changed, 24, 0x7f800000)
        with self.assertRaisesRegex(ValueError, 'FINITE_TRACE'):
            scores_from_saved_trace(changed, 0)

    def test_runner_observes_all_scores_and_separates_io_from_timing(self):
        self.assertIn('callbacks.score = verify ? score_observer : NULL;', self.runner)
        self.assertIn('c->compared_scores != VERIFY_WINDOWS', self.runner)
        self.assertIn('same_float(score->logits[k], expected->logits[k])', self.runner)
        function = self.runner[self.runner.index('static int verify_case'):self.runner.index('static void print_verification')]
        self.assertLess(function.index('read_exact_file('), function.index('clock_gettime('))
        self.assertLess(function.index('load_reference('), function.index('clock_gettime('))
        self.assertNotIn('printf(', function)
        self.assertNotIn('fwrite(', function)
        self.assertIn('live.origin = i == 2 ? 48000 : 0;', self.runner)
        self.assertIn('if (verify && !attempted) print_verification(reports, 0, 0);', self.runner)


if __name__ == '__main__':
    unittest.main()
