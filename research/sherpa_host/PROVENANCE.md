# Source, attribution and research policy

New CLI, launcher, tests and build/materialization glue follow the repository's
Apache-2.0 LICENSE. No new upstream copyright holder is invented. The extracted
wrapper retains Xiaomi's original notice and the 19 byte-exact source spans
identified in `wrapper-extraction.json`. The original Sherpa license is already
at `../sherpa_pcm/vendor/LICENSE.sherpa-onnx`, and is also copied as `source/LICENSE`
when materializing. That existing header/license remains unchanged.

Sherpa source is pinned to
[11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf](https://github.com/k2-fsa/sherpa-onnx/tree/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf).
The original decoder SHA256 is
`872a9e074e626ce66914a4a1bb48948790851c281907340ea319d9817f888edb`;
the derivative is `195484ef46410bc44626b85d303ae2f0c5cb22668313e8ed171b3ebc0a55d8d0`.
`patches/bounded-viterbi.patch` is the entire upstream-to-candidate modification,
SHA256 `0f0ee458042b18962564f00566bfec17a4bf5dc88c98e26f64dfb55fa915750f`.

This research policy keeps Normalize and suffix scoring, groups bounded future
state, and keeps the strictly larger whole-path maximum. Ties keep the earlier
entry; merged paths do not use LogAdd. It is not equivalent to the original
official algorithm. Historical strict official parity failed; the successful
145-stream gate compares the same Viterbi policy before/after engineering
extraction. Neither quality nor near-phonetic/intent rejection was improved.

The retained version getters report Sherpa 1.13.8 and Git string 8c8e275d from
the upstream generated version source. These strings differ from the source
commit above and are not replaced or used instead of source/content hashes.

Exact selected sources and per-file hashes retain kaldi-decoder 0.3.0,
kaldi-native-fbank 1.22.3, KISS commit febd4caeed32e33ad8b2e0bb5ea77542c40f18ec,
kaldifst 1.8.0, OpenFst 1.8.5-2026-07-09, Eigen 5.0.1, simple-sentencepiece 0.7,
nlohmann-json 3.12.0 license/search-slot material and ORT 1.28.2 headers.
`licenses.json` fixes every retained license/notice file. Full Eigen
MPL/BSD/Apache/MINPACK declarations, KISS notices and ORT ThirdPartyNotices are
preserved in materialized source. Inclusion of a notice does not imply every
file in that component has all listed licenses.

The fixed x86 ORT is a prebuilt wheel library, not a locally reproduced ORT
build. Matching-version header/license inputs came from the maintainer ARM
archive; the Microsoft MIT LICENSE and full ThirdPartyNotices independently
matched official ORT commit
[33ca9628233dc8f002435e868d4c2e9f82766ca1](https://github.com/microsoft/onnxruntime/tree/33ca9628233dc8f002435e868d4c2e9f82766ca1).
Only their explicitly selected header/text bytes are consumed; no ARM binary.

The external model is pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01,
ModelScope revision 3787015f084cb241dfa0e4ba237703a2d4322d50. Its original README
declares Apache License 2.0; that revision has no separate upstream LICENSE.
Staging retains the original README, a clearly identified standard license
text, and the supplied provenance record. Encoder remains INT8, decoder/joiner
FP32, and tokens unchanged. The two-keyword profile is local research configuration.

Source text and compact result identities are the contribution. No model,
runtime ELF, audio, corpus, private process trace or historical data archive is
included. Source/model license statements do not grant rights to arbitrary
audio or establish commercial suitability, runtime quality or board acceptance.
