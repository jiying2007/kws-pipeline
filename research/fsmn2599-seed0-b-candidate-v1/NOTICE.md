# Attribution and rights scope

This is a metadata-only candidate record. No trained weights, original model
weights, source code, generated audio or source-corpus audio are included.
It does not alter the repository's existing [Apache-2.0 license](../../LICENSE).

## Base model and implementation

The checkpoint derives from `thuduj12/kws_wenwen_fsmn_ctc`, published on
[ModelScope](https://modelscope.cn/models/thuduj12/kws_wenwen_fsmn_ctc).
The fixed-revision model card explicitly declares Apache License 2.0; the
same-revision official file listing binds the original `avg_30.pt`, configuration,
CMVN and tokens. It contains no separate LICENSE or NOTICE file. Their source
URLs, exact revision and retained byte identities are in [source-manifest.json](source-manifest.json).
This model-specific declaration is used directly; a code license is not being
used to guess the model-weight license.

The architecture reference is [WeKws](https://github.com/wenet-e2e/wekws/tree/6a45aeb994dd81c0969ff877a5a7c46d60ed0c86),
commit `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`, Apache-2.0.
Retain the relevant author attribution: FSMN implementation, yueyue.nyy (2022)
and Jing Du (2023); KWS model implementation, Binbin Zhang (2021) and Jing Du
(2023); streaming keyword spotter, Jing Du (2023) and WeKws contributors.
No separate upstream NOTICE file was found in that commit's recursive tree.
The candidate modification is head-weight/head-bias fine-tuning only; original
encoder/CMVN tensor hashes were retained. This does not claim authorship of the
base weights, implementation or original pretraining material.

For any later authorized binary redistribution, retain the applicable
Apache-2.0 text, attribution and prominent modification notices, plus any NOTICE
required by the components actually distributed. The present metadata record is
not itself a completed model redistribution/promotion package. The
[Apache-2.0 redistribution conditions](https://www.apache.org/licenses/LICENSE-2.0)
remain applicable.

## Synthetic training lineage

The 26 Qwen training recordings are stock-preset synthetic outputs from the
Qwen team model
[Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/README.md),
revision `85e237c12c027371202489a0ec509ded67b5e4b5`. Its fixed model card declares
Apache-2.0. A separate audio-output license is not specified there. That absence
is not treated as an automatic prohibition on a trained-weight release, and is
also not represented as a universal output-rights or personality-rights grant.
No voice-clone material is part of this head's training set. Stock-preset names
describe model options, not verified real-speaker identities or endorsements.

## MSWC attribution

The 120 published spoken-word training recordings come from MLCommons and the
Multilingual Spoken Words Corpus contributors. The
[official dataset page](https://mlcommons.org/datasets/multilingual-spoken-words/)
states [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) and
describes keyword spotting and spoken-term search as intended uses. Preserve
dataset attribution, source/license links and applicable modification notices
when redistributing covered material. Do not attempt to identify speakers, as
required by the official download conditions. This work selected a small subset,
decoded audio for research features and used the inherited word labels; source
audio and individual speaker identifiers are not redistributed here.

CC-BY-4.0 is recorded as the source-data license, not automatically imposed as
a blanket new license on the trained checkpoint. Neither that license nor a
model-card declaration grants every privacy/publicity right or warrants all
unknown pretraining sources. No new third-party agreement or model license is
accepted or created by this documentation.
