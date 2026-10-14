# Public small-package boundary

This package contains preparation/audit/orchestration source, synthetic tests, and frozen numerical/path/hash receipts. It is **not a fully self-contained audio or executable bundle**.

## Included and unchanged receipts

The original acquisition, audit, build, diagnostic invocation/stats/completion/summary receipts are copied byte-for-byte. `model-scope-audit.json` and `positive-control-review/review.json` add interpretation without editing original execution evidence. The current `source-plan.json` records the observed archive hash and completed diagnostic stage; the initial plan remains `versions/v1/source-plan.json`. The successful full audit's producer is retained as `versions/v4/alimeeting_eval.py`; the current top-level source is v5, whose metadata rescan matched all eight actual TextGrid results exactly.

## Deliberately omitted bodies

- Official 3,673,718,355-byte Eval archive, all original/derived WAVs, and TextGrid/transcript bodies
- Incomplete first audit's derived audio and full metadata-inspection copies
- CMake wheel/tool installation, compiler/linker executables, ELF runner, libraries/objects, full local CMake build directory
- Historical 1.67MB per-record identity index and the 676KB pre/post whole-source verification files; their hashes/counts remain in published summaries/receipts
- Upstream 2383-leaf source manifest/full source snapshot and original model/checkpoint/keyword-pack bodies; original repository source/assets are identified by exact commit/tree/file hashes, not duplicated here
- Credentials, private corpora, new TTS/ASR audio, or private model weights

The original C source needed for this build was available and verified in the execution workspace at P4 commit `11b93abe92427c3ece2a01f9a497cdb74bab0062`, tree `b8ae29e82097c38e92dbb1bb1cd4db10ade6cdc6`. One separately disclosed historical research-archive leaf was unavailable and unrelated to compilation. `build_frozen_runner.py` is retained as producer source with fixed historical paths; it requires the referenced original source/manifests/toolchain and is not a portable one-command rebuild promise. Build receipts include absolute historical paths as provenance; copying them does not claim those paths exist on a CI runner.

Public metadata can verify receipt/hash relationships and numerical consistency. Repeating acoustic intake or inference requires obtaining the explicitly identified external bodies, revalidating them, and separately constructing a valid execution plan. Do not silently rewrite immutable receipts to point to replacement audio or binaries.

## Offline synthetic test interfaces

All tests below run without original audio, model inference, Torch, network access, or the historical identity index:

- `python3 test_alimeeting_eval.py`: 36 tests; tiny synthetic WAV/tar/TextGrid fixtures, mocked download responses
- `python3 test_diagnostic_runner.py`: 8 tests; numerical/stat/event validators, no model invocation
- `python3 overlap/test_compare_overlap.py`: 14 tests; fabricated identity fixtures only

The same commands support `-O` and `-OO`; all also passed isolated execution in the preparation workspace. These 58 tests establish tooling contracts only, not acoustic performance. Any P6 aggregate/receipt tests are additional and must be reported separately.

`diagnostic_runner.py --execute` is not a test-suite step. It requires the exact reviewed input/audit/build bindings and refuses to overwrite an existing output directory. No CI test should invoke it against real data. The recorded one-pass diagnostic has zero triggers over 4.205059722 input hours, but labels remain unknown, accepted-negative exposure is zero, and current same-binary positive-response control remains absent. No FAR, recall, FRR, accuracy improvement, sealed qualification, or shipping approval follows.

## License

The corpus source is OpenSLR119 AliMeeting, licensed on the official data page as CC BY-SA4.0: https://www.openslr.org/119/ and https://creativecommons.org/licenses/by-sa/4.0/ . Preserve attribution and source/license links. The code repository's license does not replace the corpus license. No corpus audio/transcript redistribution is part of this small package. The archive SHA is an observed complete-download checksum, not a publisher-supplied SHA.
