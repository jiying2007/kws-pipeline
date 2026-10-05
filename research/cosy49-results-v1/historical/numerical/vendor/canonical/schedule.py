"""Reconstructed integer-only canonical schedule; never reads PCM."""
from dataclasses import dataclass


@dataclass
class State:
    wave_samples: int = 0
    feature_count: int = 0
    offset: int = 0
    total_samples: int = 0
    total_fbank: int = 0
    total_selected: int = 0


def plan_call(state, count, index, final_short):
    assert isinstance(count, int) and 0 < count <= 4800
    assert 0 <= state.wave_samples < 800 and state.feature_count in (0, 3, 4) and state.offset in (0, 1, 2)
    assert state.total_samples == 160 * state.total_fbank + state.wave_samples
    wave = state.wave_samples + count
    fbank = 0 if wave < 800 else 1 + (wave - 400) // 160
    prefix = 2 if state.feature_count == 0 else state.feature_count
    splice = prefix + fbank - 4 if fbank else 0
    selected = list(range(state.offset, splice, 3))
    centers = [i if state.feature_count == 0 else state.total_fbank - state.feature_count + i + 2 for i in selected]
    old_prefix = ([0, 0] if state.feature_count == 0 else
                  list(range(state.total_fbank - state.feature_count, state.total_fbank)))
    indices = old_prefix + list(range(state.total_fbank, state.total_fbank + fbank))
    splices = [indices[i:i + 5] for i in selected]
    assert all(len(x) == 5 for x in splices)
    windows = [state.total_fbank * 160 + i * 160 for i in range(fbank)]
    state.total_samples += count
    state.wave_samples = wave - fbank * 160
    if fbank:
        remainder = (splice + (0 if state.offset == 0 else 3 - state.offset)) % 3
        state.offset = 0 if remainder == 0 else 3 - remainder
        state.feature_count = min(fbank, 4)
    result = dict(call_index=index, available_samples=state.total_samples, call_samples=count,
                  waveform_samples=wave, fbank_rows=fbank, splice_rows=splice,
                  selected_rows=len(selected), is_final_short=int(final_short),
                  wave_samples=state.wave_samples, feature_count=state.feature_count, offset=state.offset,
                  fbank_begin=state.total_fbank, model_begin=state.total_selected,
                  window_sample_starts=windows, splice_fbank_indices=splices, centers=centers)
    state.total_fbank += fbank
    state.total_selected += len(selected)
    return result


def stream_schedule(samples):
    assert isinstance(samples, int) and 0 <= samples < 2**63
    state = State()
    calls = [plan_call(state, min(4800, samples - start), i, samples - start < 4800)
             for i, start in enumerate(range(0, samples, 4800))]
    return dict(samples=samples, calls=calls, fbank_rows=state.total_fbank,
                model_rows=state.total_selected, canonical_calls=len(calls),
                acoustic_calls=sum(c['selected_rows'] > 0 for c in calls), residual_samples=state.wave_samples,
                eof_flush=False, finish_additional_frames=0, reset='one fresh state per stream')


def operation_budget(schedules):
    frames = sum(s['fbank_rows'] for s in schedules)
    rows = sum(s['model_rows'] for s in schedules)
    return dict(streams=len(schedules), pcm_samples=sum(s['samples'] for s in schedules),
                canonical_calls=sum(s['canonical_calls'] for s in schedules),
                acoustic_calls=sum(s['acoustic_calls'] for s in schedules),
                dft_frames=frames, decimal_fsmn_rows=rows,
                preprocessing_rne_operations=1601 * frames, integer_pcm_sum_additions=400 * frames,
                dft_real_component_products=205600 * frames, dft_real_component_additions=205600 * frames,
                power_products=514 * frames, power_additions=257 * frames,
                mel_products=501 * frames, mel_additions=501 * frames, decimal_ln_calls=240 * frames,
                cmvn_rne_operations=800 * rows, fbank_unique_round_certificates=80 * frames,
                model_weighted_products=388984 * rows, model_affine_calls=12 * rows,
                model_memory_calls=4 * rows, probability_decimal_exp_calls=6 * rows,
                probability_unique_round_certificates=6 * rows, diagnostic_display_scalar_conversions=3810 * rows,
                additional_model_evaluations_for_diagnostics=0)
