"""Reconstructed SAME canonical PCM-derived reference. Inert import, no launcher.

New runtime/source identity requires recovery review and durable Library source /
protocol persistence before any reference validation or candidate evaluation.
This module itself grants no execution authorization and has no hidden forward.
"""
from decimal import localcontext
import json
import sys
import numpy as np
from assets import HISTORICAL_CONTRACT_SHA, check_pins, contract_binding, digest, inert_module, load_state, load_tables
from authority import metadata as authority_metadata
from decoder import CanonicalDecoder
from diagnostics import retain as retain_diagnostics
from ieee754 import bits, from_bits, mul, round_ratio_bits, certify_interval, sub
from schedule import stream_schedule, operation_budget
from softmax import softmax_interval
from runtime_guard import validate_runtime


def preprocess(pcm400, hamming):
    assert isinstance(pcm400, np.ndarray) and pcm400.dtype == np.int16 and pcm400.shape == (400,)
    assert hamming.dtype == np.float32 and hamming.shape == (400,)
    total = sum(map(int, pcm400))
    mean = from_bits(round_ratio_bits(total, 400))
    dc = [sub(float(int(v)), mean) for v in pcm400]
    coefficient = from_bits(0x3f7851ec)
    pre = [sub(value, mul(coefficient, dc[index - 1 if index else 0])) for index, value in enumerate(dc)]
    window = np.zeros(512, dtype=np.float32)
    window[:400] = [mul(value, float(weight)) for value, weight in zip(pre, hamming)]
    assert np.isfinite(window).all() and np.all(np.abs(window.astype(np.float64)) <= 65536)
    trace = dict(pcm400_sha256=digest(pcm400.tobytes()), integer_dc_sum=total,
                 mean_f32_bits_hex=f'{bits(mean):08x}',
                 dc_f32le_sha256=digest(np.asarray(dc, dtype='<f4').tobytes()),
                 preemphasis_f32le_sha256=digest(np.asarray(pre, dtype='<f4').tobytes()),
                 window512_f32le_sha256=digest(window.astype('<f4', copy=False).tobytes()))
    return window, trace


def cmvn(spliced, mean, istd):
    assert spliced.dtype == mean.dtype == istd.dtype == np.float32
    assert spliced.ndim == 2 and spliced.shape[1] == len(mean) == len(istd) == 400
    assert np.isfinite(spliced).all() and np.isfinite(mean).all() and np.isfinite(istd).all() and np.all(istd > 0)
    centered = np.array([[sub(float(value), float(m)) for value, m in zip(row, mean)] for row in spliced],
                        dtype=np.float32).reshape((-1, 400))
    normalized = np.array([[mul(float(value), float(scale)) for value, scale in zip(row, istd)] for row in centered],
                          dtype=np.float32).reshape((-1, 400))
    assert np.isfinite(normalized).all()
    return normalized, digest(centered.astype('<f4', copy=False).tobytes())


class CanonicalPipeline:
    """One fixed reference sequence per PCM stream; caller owns execution gate.

    Constructor prepares constants only. Every evaluate_pcm call requires a
    separately reviewed, durable recovery protocol and once-only bounded harness.
    No new seed, candidate, actual PCM, or reference output is authorized here.
    """
    def __init__(self):
        self.source_identity = check_pins()
        self.binding = contract_binding()
        self.runtime_guards = validate_runtime()
        self.tables = load_tables()
        self.state, self.weight_manifest = load_state()
        self.dft = inert_module('recovered_bound_decimal_dft', 'frontend-diagnosis/decimal_dft.py')
        self.fsmn = inert_module('recovered_bound_decimal_fsmn', 'oracle-corrected/oracle/decimal_fsmn.py')
        self.decimal_context = self.fsmn._contexts()[0]
        with localcontext(self.decimal_context):
            self.roots = self.dft.roots()
        self.model = self.fsmn.Decimal80FSMN(self.state)

    def evaluate_pcm(self, pcm, *, expected_pcm_sha256, ingress_chunks=None, with_events=True):
        before = check_pins()
        runtime_before = validate_runtime()
        assert before == self.source_identity
        assert isinstance(pcm, np.ndarray) and pcm.dtype == np.int16 and pcm.ndim == 1
        assert len(expected_pcm_sha256) == 64
        assert digest(pcm.astype('<i2', copy=False).tobytes()) == expected_pcm_sha256
        if ingress_chunks is None:
            ingress_chunks = [len(pcm)] if len(pcm) else []
        assert all(isinstance(n, int) and n >= 0 for n in ingress_chunks) and sum(ingress_chunks) == len(pcm)
        schedule = stream_schedule(len(pcm))
        assert schedule['model_rows'] <= 64, 'unreviewed model sequence length: STOP'
        fbank_rows, frame_records = [], []
        for call in schedule['calls']:
            for start in call['window_sample_starts']:
                window, trace = preprocess(pcm[start:start + 400], self.tables['donor_hamming'])
                with localcontext(self.decimal_context):
                    ideal = self.dft.evaluate(window, self.tables['donor_mel_weights'],
                                              self.tables['donor_mel_offsets'], self.tables['donor_mel_bins'], self.roots)
                    enclosed = self.dft.output_intervals(ideal, self.tables['donor_mel_weights'],
                                                        self.tables['donor_mel_offsets'], self.tables['donor_mel_bins'])
                row, certificates = [], []
                for lower, upper in enclosed['log_intervals']:
                    value, certificate = certify_interval(lower, upper)
                    row.append(value)
                    certificates.append([certificate['lower'], certificate['upper'], certificate['rne_bits_hex']])
                fbank_rows.append(row)
                frame_records.append(dict(sample_start=start, preprocessing=trace,
                                          logfbank_round_certificates=certificates))
                del ideal, enclosed, window
        fbank = np.array(fbank_rows, dtype=np.float32).reshape((-1, 80))
        assert len(fbank) == schedule['fbank_rows']
        splice_indices = [indices for call in schedule['calls'] for indices in call['splice_fbank_indices']]
        spliced = np.array([fbank[indices].reshape(400) for indices in splice_indices], dtype=np.float32).reshape((-1, 400))
        normalized, centered_sha = cmvn(spliced, self.state['global_cmvn.mean'], self.state['global_cmvn.istd'])
        assert len(normalized) == schedule['model_rows']
        if len(normalized):
            full = self.model.evaluate(normalized)
            final = self.fsmn.slice_interval_result(full, 0, len(normalized))
            final['counters'] = full['counters']
            diagnostic, cache_errors = retain_diagnostics(full, self.fsmn.DIMS)
            final['final_cache'] = full['final_cache']
            final['final_cache_error_bounds'] = cache_errors
            final['final_cache_sha256'] = digest(json.dumps(full['final_cache'], separators=(',', ':')).encode())
            del full
        else:
            final = dict(schema='a20-decimal80-ideal-result.v1', precision_digits=80, uncertainty_cap='1e-20',
                         input_rows=0, logits=[], lower=[], upper=[], error_bounds=[], interval_radii=[],
                         final_cache_sha256=None, final_cache=[], final_cache_error_bounds=[],
                         counters=dict(reference_rows=0, affine_calls=0, memory_calls=0, weighted_products=0))
            diagnostic, _ = retain_diagnostics(dict(input_rows=0, final_cache=[],
                stages={f'stage{i}': [] for i in range(21)},
                stage_error_bounds={f'stage{i}': [] for i in range(21)}), self.fsmn.DIMS)
        probability_records, rounded = [], []
        for values, lower, upper in zip(final['logits'], final['lower'], final['upper']):
            record = softmax_interval(values, lower, upper)
            row, certificates = [], []
            for lo, hi in record['probability_intervals']:
                value, certificate = certify_interval(lo, hi)
                row.append(value)
                certificates.append([certificate['lower'], certificate['upper'], certificate['rne_bits_hex']])
            record['round_certificates'] = certificates
            probability_records.append(record)
            rounded.append(row)
        probabilities = np.array(rounded, dtype=np.float32).reshape((-1, 6))
        decoder = CanonicalDecoder() if with_events else None
        calls, events = [], []
        for geometry in schedule['calls']:
            first_feature, first_row = geometry['fbank_begin'], geometry['model_begin']
            feature_count, row_count = geometry['fbank_rows'], geometry['selected_rows']
            raw_call = self.fsmn.slice_interval_result(final, first_row, row_count) if row_count else {
                **{key: final[key] for key in ['schema', 'precision_digits', 'uncertainty_cap']},
                'input_rows': 0,
                **{key: [] for key in ['logits', 'lower', 'upper', 'error_bounds', 'interval_radii']}}
            item = dict(geometry=geometry, fbank=fbank[first_feature:first_feature + feature_count],
                        spliced=spliced[first_row:first_row + row_count], cmvn=normalized[first_row:first_row + row_count],
                        probabilities=probabilities[first_row:first_row + row_count], ideal_logits=raw_call,
                        ideal_probabilities=probability_records[first_row:first_row + row_count])
            if decoder is not None:
                item['decoder'] = decoder.process(probabilities[first_row:first_row + row_count])
                returned = item['decoder']['return_value']
                if returned.get('state') == 1:
                    events.append(dict(returned, available_audio_samples=geometry['available_samples']))
            calls.append(item)
        after = check_pins()
        assert runtime_before == validate_runtime() == self.runtime_guards
        assert before == after and digest(pcm.astype('<i2', copy=False).tobytes()) == expected_pcm_sha256
        return dict(schema='a20-canonical-pcm-pipeline.recovery-v1', field_semantics='a20-canonical-pcm-pipeline.v3',
                    contract_sha256=self.binding['recovery_contract_sha256'],
                    historical_v3_sha256=HISTORICAL_CONTRACT_SHA, historical_v3_bytes_recovered=False,
                    source_reconstructed=True, reference_authorities=authority_metadata(),
                    pcm_f32_authority=False, pcm_i16_sha256=expected_pcm_sha256,
                    weight_state_sha256=self.weight_manifest['state_sha256'], sources_before=before, sources_after=after,
                    runtime=dict(python=sys.version, numpy=np.__version__, byteorder=sys.byteorder, guards=runtime_before),
                    ingress_chunks=ingress_chunks, schedule=schedule, workload=operation_budget([schedule]),
                    cmvn_centered_f32le_sha256=centered_sha,
                    canonical_cmvn_f32le_sha256=digest(normalized.astype('<f4', copy=False).tobytes()),
                    frame_records=frame_records, final_logits=final, probability_records=probability_records,
                    stage_diagnostics=diagnostic, calls=calls, events=events, events_included=bool(with_events),
                    authority='frozen PCM16 -> exact software FP32 preprocessing -> bounded mathematical DFT/power/mel/log -> certified FP32 fbank -> exact splice -> two software FP32 CMVN operations -> propagated Decimal80 FSMN -> bounded mathematical softmax -> certified FP32 decoder probabilities',
                    ambiguous_rounding='STOP; no escalation or midpoint fallback', hidden_native_calls=0, hidden_torch_calls=0)


def compare_raw_logits(actual, ideal):
    assert ideal['schema'] == 'a20-decimal80-ideal-result.v1'
    assert ideal['precision_digits'] == 80 and ideal['uncertainty_cap'] in ('1E-20', '1e-20')
    if ideal['input_rows'] == 0:
        assert actual.dtype == np.float32 and actual.shape == (0, 6)
        assert all(ideal[key] == [] for key in ['logits', 'lower', 'upper', 'error_bounds', 'interval_radii'])
        return dict(passed=True, elements=0, failed_elements=0, failures=[], maximum_absolute_error_upper='0',
                    maximum_tolerance_ratio_upper='0', minimum_margin_lower=None, atol='1e-4', rtol='1e-5',
                    criterion='empty sequence: shape/type checked; no fabricated mathematical row')
    bound = inert_module('recovered_bound_raw_compare', 'oracle-corrected/oracle/decimal_fsmn.py')
    return bound.compare_float32_logits(actual, ideal)
