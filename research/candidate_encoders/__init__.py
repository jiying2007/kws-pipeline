"""Opt-in research encoder specifications; importing this package never imports Torch."""
from .contract import EncoderSpec, causal_ds_tcn_spec, rnn64_spec

__all__ = ['EncoderSpec', 'causal_ds_tcn_spec', 'rnn64_spec']
