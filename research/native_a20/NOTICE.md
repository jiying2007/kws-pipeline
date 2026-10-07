# Source provenance and licenses

The FSMN and streaming components derive from this repository's isolated
`research/donor_fsmn` implementation of WeKws commit
`6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`. Original FSMN authors are Yueyue Nyy
and Jing Du. The original streaming decoder is Copyright (c) 2023 Jing Du
(thuduj12@163.com). The implementation preserves the pinned decoder behavior
and the reviewed inclusive final-start suffix-search fix.

Apache License 2.0 is retained in LICENSE and
baseline/native/stream/LICENSE.wekws. The Hamming frontend derives from the
separately maintained research/donor_fbank implementation and pinned torchaudio
Kaldi features; preserve baseline/native/donor_fbank/LICENSE.torchaudio
(BSD-2-Clause). SHA256 code derives from the same Apache-2.0 repository.

This is a modified research implementation: six-class model geometry,
binary64 accumulation and FFT workspace, exact identity checking and explicit
caller-owned streaming state. PROVENANCE.json and SOURCE_MANIFEST.json bind
public files and unchanged selected C sources. Build/export/test wrappers are
adapted for repo-relative, caller-supplied paths; that transport adaptation is
not represented as byte-identical historical execution evidence.

Model lineage is the separately trained A20 endpoint descended from official
`iic/speech_charctc_kws_phone-xiaoyun`. The provider model-card declaration of
Apache License 2.0 applies to the distributed model; it does not establish
rights in unknown pretraining audio or arbitrary synthesized audio. Model/data
assets require their own exact provenance and redistribution inventory.

No code here is promoted to the shipping build, model registry, RNN ABI or
deployment path. Host sizes/timing and exposed numerical fixtures do not qualify
SSC305/A32, final hardware/AFE, real-world FAR/FRR or acoustic performance.
