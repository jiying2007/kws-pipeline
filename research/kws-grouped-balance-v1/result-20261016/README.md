# Original grouped A/B evidence

Source commit: `942b3c1495fa4e4007ca01604ca8a65ed2cd5bcf`; tree `60d8d4a4d28a80c576abeead37c4e8d64a510c07`. Enabled plan and executed source manifest are preserved byte-for-byte. Acceptance and result-hash manifest are original verifier outputs. Raw result ZIP is split into ordered parts bound by parts-manifest.json; verification reconstructs only in memory and does not execute models.

The fixed A600/B600 endpoints are primary. A400/A500 are descriptive, not selected alternatives. Probe is exposed, machine-labelled, non-test and never used for selection. Preset grouping is an identity proxy, not independent-speaker proof. No human gold, true recall, false-activation rate, generalization, shipping or reusable-weight claim. Original acceptance verified saved probabilities with float64 CTC; the public archive checker checks integrity/receipts/token statistics only, not independent CTC or process authenticity.

Run `python research/kws-grouped-balance-v1/verify_archived_result.py` from the repository root. No Torch, model, optimizer, ASR or TTS calls.
