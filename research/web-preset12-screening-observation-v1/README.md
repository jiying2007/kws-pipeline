# Web preset12 research screening observation

This text-only archive retains an independently reviewed aggregate of twelve web-preset synthetic clips, screened once each by Qwen3-ASR-0.6B and sherpa-onnx SenseVoice. Counts compare recognition text with **generation requests**, not verified spoken words, acoustic accuracy, KWS performance or correct-label rates. No inference was repeated for this publication.

## Reviewed aggregate

The exact product keywords are **你好小窝 / 小窝小窝**. The reviewed comparison removes Unicode category-P punctuation only; raw outputs remain unchanged and no homophones are folded.

| Subset | Items | SenseVoice request matches | Qwen request matches |
| --- | ---: | ---: | ---: |
| Product-target requests | 4 | 2 | 2 |
| Near-sounding requests | 8 | 7 | 6 |
| Total | 12 | 9 | 8 |

The two recognizers agreed on 10/12 items. Agreement is weak machine evidence, including when both outputs differ from the request. Preserve distinctions among 窝 / 屋 / 哦 / 欧. Acoustic completeness remains **UNKNOWN**, and `training_admitted=false`.

[research-aggregate.json](research-aggregate.json) is the byte-exact reviewed public summary: **1,436 bytes**, SHA-256 `e3ee18e04eccaf0786b372f24a0e9f568d16219cc1732d12a2b00708e5706b4f`. Its independent-review PASS describes the prior review of execution originals and corrected target grouping. The repository contains neither those private originals nor actual per-item transcripts, so public tests do not independently recompute the twelve-item aggregates or authenticate that earlier review. No per-item content is invented to fill this gap.

## Actual identities and observation limits

The research Qwen run used CPU FP32/eager, `max_new_tokens=512`, empty context, automatic language (`None`) and no aligner. It captured generated token IDs and EOS, but not the full effective-config observations required by the existing native-v2 contract. That contract remains fixed at 256; the research run is not relabeled to match it.

SenseVoice used sherpa-onnx 1.13.8 and a QUInt8 conversion. It is a different runtime/model condition from historical FunASR FP32 and lacks native-v2 raw CTC/metadata/lexical boundary observations. No missing fields are manufactured. Two recognizers also do not meet the unchanged production three-distinct-family policy. A third recognizer cannot repair these identity/observability gaps or establish human truth, and no third execution is authorized here.

The production fixed-punctuation rule, research `normalized_actual` (Unicode punctuation plus whitespace), and this report's punctuation-only review comparison are separate policies. They agree on this batch's punctuation; this observation does not change any normalization rule.

Qwen's recorded group wall time is approximately 171.679448 seconds; child CPU time is approximately 170.058712 seconds. Sampled maximum RSS is 6,281,240,576 bytes, **not a guaranteed hard peak**. Twelve attempts were consumed, with zero retries, one generation call per input, 9–10 new tokens, EOS 151645 and no 512-token cap hit. Decoder termination does not prove acoustic completeness. SenseVoice ran before Qwen; an inherited contradictory start-template phrase was corrected append-only, without rewriting original evidence.

## Existing-boundary regressions

Nine invented text/metadata tests were added to the existing quality-guard suite. They exercise current `tools/speech_asr_consensus.py`, `tools/asr_native_contracts.py` and research `quality_gates.label_preparation`; no baseline source copies, new saved-result adapter, admission controller or runtime is introduced.

The mocks check raw-text preservation, distinct confusables, rejection of two recognizers under a three-family policy, weak agreement without training eligibility, disagreement retention, rejection of shared non-request text, unknown missing results, unchanged native identities and exact product keywords. They do not test actual audio, establish transcript truth, or rerun historical batches.

Run the new tests with:

```sh
python3 -S -B -m unittest discover -s research/experiment_quality_guards/tests -p test_existing_boundaries.py -v
```

Repeat with `-O` and `-OO`. The existing required CI and standalone offline quality-guard workflow include these tests in all three modes and retain their test-inventory checks. Source-retention maintenance records the added public files and current workflow changes; historical source provenance remains intact.
