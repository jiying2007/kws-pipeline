# Historical near-pronunciation decoder test inputs

These are exact historical source copies whose SHA-256 values are already
pinned by `research/near_pronunciation_v1/decoder_overlay.json`:

- `src/decoder.c`: `c1cef85ba33ab37dd2e278f7c4a1f0d398b128e5508e86cfff3933819e2b78e9`
- `tests/test_decoder.c`: `7d70c111c9292bc8697871508746e465477ed8142e69f77056c603dcccbe9109`
- `tools/kws_trace_io.c`: `2f66cf4e8181462dc35e4a5d45a2b3578411b712db1ef22fca048c86232fab96`
- `tools/kws_trace_io.h`: `b8a28a5257294ebebc44e52bd65f03d8f175ae855e0da4cc8422a363221a1349`
- `tools/kws_decoder_replay.c`: `86ff25540faa33c54b275ca943bdb8bfc7b96893c4542da2c78fd1340bae8483`

The decoder and decoder-test copies precede the non-default `longest` / `grace`
pending-expiry fix. The trace I/O copies precede the exclusive-create writer API
used by single-pass WAV trace capture; they are exact copies of the public
[`tools/kws_trace_io.c`](https://github.com/jiying2007/kws-pipeline/blob/a7015ff61947b423ed0162b692a39f7590255941/tools/kws_trace_io.c)
and [`tools/kws_trace_io.h`](https://github.com/jiying2007/kws-pipeline/blob/a7015ff61947b423ed0162b692a39f7590255941/tools/kws_trace_io.h)
at that commit, verified against the original overlay pins above. No historical
overlay, derived-source hash, preparer, verifier, experiment receipt or scientific
pin is changed.

The replay copy precedes optional native execution completion and is retained
byte-for-byte from public commit `4b9d5e05f51d73e356f39e80b01fa0e94ab5ee94`.
Its digest matches the existing overlay pin; current replay changes must not
rewrite historical source identity.

`tests/prepare_frozen_decoder_source.py` stages the entire pinned source closure
in a fresh temporary directory. Only these five explicitly listed files use
retained copies. Every other input comes from the current tree and must still
match its historical hash; any drift is an error, never an automatic pin update.

The CLI stdout test uses current source for all shipping tools, and the staged
historical closure only for the two research child-admission modes. The separate
historical verifier uses the same staged closure. Its parity and synthetic
verification results apply to that historical decoder only. Current decoder
regressions and source-bound recalibration requirements remain separate.

All checks use invented logits and test-only providers, without acoustic model
execution, downloads, training, or campaign replay.
