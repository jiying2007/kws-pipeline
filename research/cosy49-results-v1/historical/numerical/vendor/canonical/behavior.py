"""Saved-record behavior comparison and reference-only feasibility, no forward."""
from fractions import Fraction
import math

PHRASES = {'你好小窝': 1, '小窝小窝': 2}
EXACT_FIELDS = ['valid', 'state', 'keyword', 'start_frame', 'end_frame', 'rows_decoded',
                'total_frames', 'last_active_pos', 'chunk_rows_available', 'eof_extra_rows',
                'call_index', 'available_samples', 'call_samples', 'is_final_short']
SCORE_FIELDS = ['score', 'hit_score']


def normalize_call(call):
    geometry, decoder = call['geometry'], call['decoder']
    returned = decoder['return_value']
    active = bool(returned and returned['state'])
    return dict(valid=int(bool(returned)), state=int(returned['state']) if returned else 0,
                keyword=PHRASES.get(returned.get('keyword'), 0) if returned else 0,
                start_frame=round(returned['start'] * 100) if active else -1,
                end_frame=round(returned['end'] * 100) if active else -1,
                score=float(returned['score']) if active else 0., rows_decoded=decoder['rows_decoded'],
                total_frames=decoder['decoder_total_frames'], last_active_pos=decoder['last_active_pos'],
                hit_score=decoder['hit_score'], chunk_rows_available=geometry['selected_rows'], eof_extra_rows=0,
                **{key: geometry[key] for key in ['call_index', 'available_samples', 'call_samples', 'is_final_short']})


def _compare_fields(expected, actual, score_tolerance, exact_fields, score_fields):
    assert len(expected) == len(actual)
    tolerance, issues = Fraction(score_tolerance), []
    for index, (reference, observed) in enumerate(zip(expected, actual)):
        for field in exact_fields:
            if reference[field] != observed[field]:
                issues.append(dict(call=index, field=field, expected=reference[field], actual=observed[field]))
        for field in score_fields:
            a, b = reference[field], observed[field]
            assert math.isfinite(a) and math.isfinite(b)
            distance = abs(Fraction.from_float(float(a)) - Fraction.from_float(float(b)))
            if distance > tolerance:
                issues.append(dict(call=index, field=field, expected=a, actual=b,
                                   exact_distance=str(distance), tolerance=str(tolerance)))
    return dict(passed=not issues, calls=len(actual), issues=issues)



def compare_records(expected, actual, score_tolerance='0.00001'):
    """Complete canonical/native behavior comparison: no fields are optional."""
    return _compare_fields(expected, actual, score_tolerance, EXACT_FIELDS, SCORE_FIELDS)


def reference_feasibility(canonical_records, original_official_records, *,
                          observed_exact_fields, observed_score_fields):
    """Explicit protocol masks for actually retained original official evidence.

    The caller must normalize both sides using a reviewed adapter. This function
    never intersects keys automatically or fabricates absent official internals.
    Canonical/native comparisons use compare_records and remain complete.
    """
    allowed = set(EXACT_FIELDS) | {'begin_byte', 'end_byte', 'available_audio_samples', 'eof_flush', 'geometry'}
    exact_fields, score_fields = list(observed_exact_fields), list(observed_score_fields)
    assert exact_fields and score_fields and len(set(exact_fields)) == len(exact_fields)
    assert len(set(score_fields)) == len(score_fields)
    assert set(exact_fields) <= allowed and set(score_fields) <= set(SCORE_FIELDS)
    assert 'last_active_pos' not in exact_fields and 'hit_score' not in score_fields, 'original internal fields were not retained'
    assert 'score' in score_fields
    report = _compare_fields(original_official_records, canonical_records, '0.00002', exact_fields, score_fields)
    report.update(schema='a20-dual-reference-feasibility.recovery-v1',
                  observed_exact_fields=exact_fields, observed_score_fields=score_fields,
                  unrecorded_original_fields_not_fabricated=['last_active_pos', 'hit_score'],
                  action='eligible_for_separately_authorized_candidate_replay' if report['passed'] else 'STOP_FOR_SPECIFICATION_DECISION',
                  candidate_gates_waived=False, candidate_calls=0, reference_calls=0)
    return report
