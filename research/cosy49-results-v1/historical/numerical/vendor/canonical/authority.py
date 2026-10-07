"""Recovered v3 semantics, with new source identity and no historical PASS claim."""
from enum import Enum
import copy


class ReferenceKind(str, Enum):
    CANONICAL_FP32_INTERFACE = 'canonical_fp32_interface'
    UNROUNDED_MATHEMATICAL_INTERVAL = 'unrounded_mathematical_interval'
    CERTIFIED_FP32_DECODER_INTERFACE = 'certified_fp32_decoder_interface'


AUTHORITIES = {
    'fbank': dict(kind=ReferenceKind.CANONICAL_FP32_INTERFACE.value, representation='IEEE754 binary32',
                  atol='1e-3', origin='unique RNE of full mathematical logfbank enclosure'),
    'spliced': dict(kind=ReferenceKind.CANONICAL_FP32_INTERFACE.value, representation='IEEE754 binary32',
                    atol='1e-3', origin='exact splice of canonical fbank interfaces'),
    'cmvn': dict(kind=ReferenceKind.CANONICAL_FP32_INTERFACE.value, representation='IEEE754 binary32',
                 atol='2e-4', origin='two separate RNE operations on canonical splice'),
    'raw_logits': dict(kind=ReferenceKind.UNROUNDED_MATHEMATICAL_INTERVAL.value,
                       representation='outward Decimal80 endpoints', atol='1e-4', rtol='1e-5',
                       origin='propagated mathematical FSMN of canonical PCM-derived CMVN'),
    'probabilities': dict(kind=ReferenceKind.UNROUNDED_MATHEMATICAL_INTERVAL.value,
                          representation='outward Decimal80 endpoints', atol='1e-5',
                          origin='bounded softmax from unrounded mathematical final logits'),
    'decoder_probabilities': dict(kind=ReferenceKind.CERTIFIED_FP32_DECODER_INTERFACE.value,
                                  representation='IEEE754 binary32',
                                  origin='unique RNE of full mathematical probability enclosure; not final-probability accuracy authority'),
}


def metadata():
    return dict(schema='a20-reference-authority-types.v1', quantities=copy.deepcopy(AUTHORITIES))


def validate_metadata(value):
    assert value == metadata(), 'wrong or changed canonical reference authority'


def require_kind(quantity, kind):
    assert AUTHORITIES[quantity]['kind'] == ReferenceKind(kind).value, ('wrong reference layer', quantity, kind)


def interface_reference(call, quantity, declared):
    validate_metadata(declared)
    require_kind(quantity, ReferenceKind.CANONICAL_FP32_INTERFACE)
    array = call[quantity]
    assert array.dtype.name == 'float32' and array.ndim == 2
    return array


def interval_reference(call, quantity, declared):
    validate_metadata(declared)
    require_kind(quantity, ReferenceKind.UNROUNDED_MATHEMATICAL_INTERVAL)
    return call['ideal_logits'] if quantity == 'raw_logits' else call['ideal_probabilities']
