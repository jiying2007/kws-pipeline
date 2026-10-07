"""Retain existing Decimal stages with measured display-conversion enclosures."""
from decimal import Decimal as D
import math
import numpy as np
from softmax import contexts


def _text(value):
    text = str(value)
    assert len(text) <= 90, 'unreviewed Decimal serialization size: STOP'
    return text


def convert_matrix(values, model_error_bounds, width):
    assert len(values) == len(model_error_bounds)
    _, up, _ = contexts()
    array = np.empty((len(values), width), dtype=np.float64)
    conversion, combined, original = [], [], []
    for r, (row, raw_error) in enumerate(zip(values, model_error_bounds)):
        assert len(row) == width
        model_error = D(raw_error)
        assert model_error.is_finite() and model_error >= 0
        radius = D(0)
        for c, text in enumerate(row):
            center = D(text)
            assert center.is_finite()
            _text(center)
            display = float(center)
            assert math.isfinite(display), 'diagnostic binary64 overflow: STOP'
            lifted = D.from_float(display)
            error = max(up.subtract(lifted, center), up.subtract(center, lifted))
            assert error >= 0 and error.is_finite()
            radius = max(radius, error)
            array[r, c] = display
        original.append(_text(model_error))
        conversion.append(_text(radius))
        combined.append(_text(up.add(model_error, radius)))
    return array, original, conversion, combined


def retain(full, dimensions):
    rows = full['input_rows']
    assert 0 <= rows <= 64 and len(dimensions) == 21
    arrays, original, conversion, combined = {}, {}, {}, {}
    for index, width in enumerate(dimensions):
        name = f'stage{index}'
        assert len(full['stages'][name]) == len(full['stage_error_bounds'][name]) == rows
        arrays[name], original[name], conversion[name], combined[name] = convert_matrix(
            full['stages'][name], full['stage_error_bounds'][name], width)
    cache_errors = []
    if rows:
        assert len(full['final_cache']) == 4
        for layer in range(4):
            name, errors = f'stage{3 + 4 * layer}', []
            assert len(full['final_cache'][layer]) == 128
            for tap in range(11):
                index = rows - 11 + tap
                errors.append(original[name][index] if index >= 0 else '0')
                for channel in range(128):
                    cache = full['final_cache'][layer][channel]
                    assert len(cache) == 11
                    _text(D(cache[tap]))
                    expected = full['stages'][name][index][channel] if index >= 0 else '0'
                    assert D(cache[tap]) == D(expected), 'canonical cache/history mismatch'
            cache_errors.append(errors)
    record = dict(schema='a20-canonical-stage-display-diagnostics.v1', arrays=arrays,
                  model_error_bounds=original, conversion_error_bounds=conversion,
                  combined_error_bounds=combined, diagnostic_only=True,
                  accepted_for_final_raw_or_probability_gates=False,
                  representation='finite IEEE754 binary64 display of computed Decimal80 stage centers',
                  conversion_enclosure='upward maximum exact-binary64-lift minus Decimal-center absolute distance per stage/row',
                  combined_enclosure='each true mathematical stage scalar is within combined row radius of displayed scalar',
                  final_cache_representation='exact serialized computed Decimal80 states retained separately; mathematical cache uncertainty is per layer/tap projection radius',
                  additional_model_evaluations=0)
    return record, cache_errors
