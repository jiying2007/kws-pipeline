"""Versioned D1 full-text heads over the existing RNN/C1 encoder equations.

EncoderSpec v1 remains the five-output historical contract. This distinct type
binds a declared 2601-output vocabulary; its arithmetic is not target admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .contract import EncoderSpec, MAX_PARAMETER_ELEMENTS, integer, require

SCHEMA = "candidate-encoder-fulltext-spec-v2"


@dataclass(frozen=True)
class FullTextEncoderSpec(EncoderSpec):
    vocabulary_sha256: str = field(kw_only=True)

    def __post_init__(self):
        # Reuse every existing backbone constraint without relaxing v1's head.
        EncoderSpec(self.architecture, self.feature_dim, self.hidden_dim, 5,
                    self.kernels, self.dilations)
        integer(self.vocab_size, 2601, 2601, "vocab_size")
        require(type(self.vocabulary_sha256) is str and
                re.fullmatch(r"[0-9a-f]{64}", self.vocabulary_sha256), "vocabulary identity required")
        require(self.parameter_elements <= MAX_PARAMETER_ELEMENTS,
                "full-text head exceeds source-allocation safety cap, not target budget")

    def to_dict(self):
        return {**super().to_dict(), "schema": SCHEMA,
                "candidate_id": self.architecture + "-d1-ctc2601-v2",
                "vocabulary_sha256": self.vocabulary_sha256, "objective": "full-text-ctc"}


def rnn64_fulltext_spec(vocabulary_sha256):
    return FullTextEncoderSpec("rnn64_control", 32, 64, 2601,
                              vocabulary_sha256=vocabulary_sha256)


def c1_fulltext_spec(vocabulary_sha256):
    return FullTextEncoderSpec("causal_ds_tcn_ctc", 32, 64, 2601,
                              (3, 3, 3), (1, 2, 4), vocabulary_sha256=vocabulary_sha256)
