"""Experimental finite binary32 add/multiply using integers only.

Each node rounds to nearest, ties to even, with gradual underflow.
Inputs must be exact Python ints encoding finite binary32; bool is rejected.
TypeError denotes non-int, ValueError denotes out-of-range/nonfinite input,
and OverflowError denotes a rounded nonfinite result. No host float is used.
Exact cancellation yields +0 except -0 + -0; multiplication uses XOR signs.

Adapted integer decomposition, rounding and packing from the portable affine
research oracle; validation, overflow handling and API documentation hardened.
Original applicable notice (retained verbatim):

New candidate, oracle, fixture and wrapper sources in this package are offered under MIT License.
Copyright (c) 2026 OpenAI-generated research contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies, and to permit persons to whom the Software is furnished to do so, subject to inclusion of this copyright and permission notice.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Reused safeguard sources retain their original provenance and any applicable original license; see SAFEGUARDS.md. This notice does not relicense preexisting project material.
"""

def _parts(bits):
    if type(bits) is not int:
        raise TypeError("binary32 bits must be an exact int")
    if not 0 <= bits <= 0xffffffff:
        raise ValueError("binary32 bits must be uint32")
    exponent = (bits >> 23) & 255
    if exponent == 255:
        raise ValueError("binary32 input must be finite")
    fraction = bits & 0x7fffff
    return (-1 if bits >> 31 else 1,
            (fraction | 0x800000) if exponent else fraction,
            exponent - 150 if exponent else -149)


def _round_int(n, shift):
    if shift <= 0:
        return n << -shift
    quotient, remainder = divmod(n, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + (remainder > half or (remainder == half and quotient & 1))


def _pack(n, exponent, negative_zero=False):
    if not n:
        return 0x80000000 if negative_zero else 0
    sign = 0x80000000 if n < 0 else 0
    n = abs(n)
    unit = max(n.bit_length() - 1 + exponent - 23, -149)
    mantissa = _round_int(n, unit - exponent)
    if not mantissa:
        return sign
    if mantissa >= 1 << 24:
        mantissa >>= 1
        unit += 1
    if unit > 104:
        raise OverflowError("binary32 rounded result is not finite")
    if mantissa < 1 << 23:
        return sign | mantissa
    return sign | ((unit + 150) << 23) | (mantissa - (1 << 23))


def mul(a, b):
    """Return separately rounded finite binary32 a * b as uint32 bits."""
    sa, ma, ea = _parts(a)
    sb, mb, eb = _parts(b)
    return _pack(sa * sb * ma * mb, ea + eb, bool((a ^ b) >> 31))


def add(a, b):
    """Return separately rounded finite binary32 a + b as uint32 bits."""
    sa, ma, ea = _parts(a)
    sb, mb, eb = _parts(b)
    exponent = min(ea, eb)
    return _pack((sa * ma << (ea - exponent)) +
                 (sb * mb << (eb - exponent)), exponent, a == b == 0x80000000)
