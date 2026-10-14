# Historical near-pronunciation decoder test inputs

These are exact copies of the two pre-pending-expiry files whose SHA-256 values
are already pinned by `research/near_pronunciation_v1/decoder_overlay.json`:

- `src/decoder.c`: `c1cef85ba33ab37dd2e278f7c4a1f0d398b128e5508e86cfff3933819e2b78e9`
- `tests/test_decoder.c`: `7d70c111c9292bc8697871508746e465477ed8142e69f77056c603dcccbe9109`

They were retained from the source snapshot before the non-default
`longest` / `grace` pending-expiry fix. No historical overlay, derived-source
hash, preparer, verifier, experiment receipt or scientific pin is changed.

`tests/prepare_frozen_decoder_source.py` stages the entire pinned source closure
in a fresh temporary directory. Only these two explicitly listed files use
retained copies. Every other input comes from the current tree and must still
match its historical hash; any drift is an error, never an automatic pin update.

The CLI stdout test uses current source for all shipping tools, and the staged
historical closure only for the two research child-admission modes. The separate
historical verifier uses the same staged closure. Its parity and synthetic
verification results apply to that historical decoder only. Current decoder
regressions and source-bound recalibration requirements remain separate.

All checks use invented logits and test-only providers, without acoustic model
execution, downloads, training, or campaign replay.
