"""Reconstructed exact-ratio binary32 RNE. No host FP32/FMA/FTZ authority."""
import math
from decimal import Decimal


class AmbiguousRounding(RuntimeError):
    """A complete mathematical enclosure does not select one binary32 value."""


def from_bits(value):
    assert isinstance(value, int) and 0 <= value < 2**32
    sign = -1.0 if value & 0x80000000 else 1.0
    exponent, mantissa = (value >> 23) & 255, value & 0x7fffff
    assert exponent != 255, 'finite binary32 only'
    significand, shift = (mantissa, -149) if exponent == 0 else (mantissa + (1 << 23), exponent - 150)
    return sign * math.ldexp(float(significand), shift)


def _nearest_integer(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    return quotient + (twice > denominator or (twice == denominator and quotient % 2 == 1))


def round_ratio_bits(numerator, denominator, negative_zero=False):
    assert isinstance(numerator, int) and isinstance(denominator, int) and denominator > 0
    sign = 0x80000000 if numerator < 0 or (numerator == 0 and negative_zero) else 0
    numerator = abs(numerator)
    if numerator == 0:
        return sign
    exponent = numerator.bit_length() - denominator.bit_length()
    below = (numerator < denominator << exponent) if exponent >= 0 else (numerator << -exponent < denominator)
    if below:
        exponent -= 1
    if exponent < -126:
        significand = _nearest_integer(numerator << 149, denominator)
        assert significand <= 1 << 23
        return sign | significand
    shift = 23 - exponent
    significand = (_nearest_integer(numerator << shift, denominator) if shift >= 0
                   else _nearest_integer(numerator, denominator << -shift))
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        raise OverflowError('canonical binary32 RNE overflow: STOP')
    assert 1 << 23 <= significand < 1 << 24
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def round_value_bits(value):
    if isinstance(value, Decimal):
        assert value.is_finite()
        numerator, denominator = value.as_integer_ratio()
        negative_zero = value.is_zero() and value.is_signed()
    else:
        assert math.isfinite(value)
        numerator, denominator = value.as_integer_ratio()
        negative_zero = value == 0 and math.copysign(1.0, value) < 0
    return round_ratio_bits(numerator, denominator, negative_zero)


def bits(value):
    return round_value_bits(value)


def sub(a, b):
    an, ad = a.as_integer_ratio()
    bn, bd = b.as_integer_ratio()
    negative_zero = a == 0 and b == 0 and bits(a) == 0x80000000 and bits(b) == 0
    return from_bits(round_ratio_bits(an * bd - bn * ad, ad * bd, negative_zero))


def mul(a, b):
    an, ad = a.as_integer_ratio()
    bn, bd = b.as_integer_ratio()
    negative_zero = bool((bits(a) ^ bits(b)) & 0x80000000)
    return from_bits(round_ratio_bits(an * bn, ad * bd, negative_zero))


def certify_interval(lower, upper):
    lo, hi = Decimal(lower), Decimal(upper)
    assert lo.is_finite() and hi.is_finite() and lo <= hi
    a, b = round_value_bits(lo), round_value_bits(hi)
    if a != b:
        raise AmbiguousRounding(f'RNE endpoint disagreement: {lo}, {hi}; {a:08x}, {b:08x}; STOP')
    return from_bits(a), dict(lower=str(lo), upper=str(hi), rne_bits_hex=f'{a:08x}',
                              proof='exact endpoint RNE bits equal; RNE is monotone',
                              escalation='none; ambiguity stops the run')
