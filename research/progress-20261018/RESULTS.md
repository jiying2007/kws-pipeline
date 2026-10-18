# Research progress: speech generation, weak import and C parity

Independent weak20 execution/JSON/byte review passed; initial four archived packages were read back (210 payload identities). No new model selection, threshold change, release, or product-admission decision is made by this report.

## Fixed comparison identity

The published [seed0 B candidate card](../fsmn2599-seed0-b-candidate-v1/MODEL_CARD.md) retains its original model identity and limitations. The experiments below do not replace that card or turn its checkpoint into a best/deployed model.

## Larger TTS bounded pilot

A 1.7B TTS pilot used eight generated utterances across two voices and instruction on/off conditions, followed by 16 blind ASR calls. For this small batch, the two ASR outputs both matched intended wording after deleting only Unicode punctuation and whitespace for 4/4 Dylan utterances and 0/4 Serena utterances. Each instruction condition had 2/4 matches. These are batch-specific agreement observations, not measured accuracy, human gold, or evidence of an instruction benefit. The recognizers agreed with each other on 5/8 items; one shared result differed from the intended wording. Agreement between two recognizers does not establish truth. This pilot used one fixed seed per voice/text pair, matched across instruction conditions; it cannot establish improvement over the earlier smaller TTS model.

Fixed model revisions: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at `0c0e3051f131929182e2c023b9537f8b1c68adfe`; `Qwen/Qwen3-ASR-0.6B` at `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; `FunAudioLLM/SenseVoiceSmall` at `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`.

Evidence SHA256 commitments (underlying evidence is privately retained, not included): asr_result `957c08319a10f045368809ac4251715345b94011c9373ceac830dfed9cc70c41`; asr_evidence_manifest `ce4828219bc06dde65517a5b4cfeab290978788be571431146c44c12bfac27f5`; independent_review `7905d08f0f68a8ee3ad2cb49dea4075120953c6d20e07715e7956a7ec4f71ebe`.

Immutable local code-closure SHA256 values, not upstream Git commits: tts_92_files `b5bd95e287790b642388c3b58963fe3e16b9787fae7fb30c4eb94209f1c37a60`; asr_bridge_118_files `0a580a60d14f8ce81c6320c002558f84d6cbc88443db63ed0f748b4be3f45e8f`. Package pins: qwen-tts 0.1.1, qwen-asr 0.0.6, funasr 1.4.16. TTS/vocoder used CPU bf16 without float32 fallback; ASR used CPU float32. Chinese/zh was language conditioning, not independent detection. SenseVoice dither reproducibility was not established.

## Last-block CTC experiments

The last-block-unfrozen arm had lower fulltext CTC NLL/token than the matched frozen arm on 49/50 labeled samples: training 26/26, development 16/17 and diagnostic 7/7. These are paired loss comparisons, not recognition accuracy. Among 23 inherited known target-presence clips, matching candidates decreased from 23 to 22: one historically accepted positive was lost, with no new known positive candidate. One weak noncanonical candidate and one unknown candidate also appeared. The unfreezing change was not adopted; the fixed original frozen seed0 B remains the research control. Better sequence fitting alone does not establish improved wake behavior. Public anonymous aggregate SHA256: f0559396ceecbd44bb5f7b6159c6e9d8b2f7cf2b6f43d659a655ce90d9992ad6.

Both arms used the same original 146 recordings, 880 presentations, 220 steps and pair-loss coefficient 0.1; the intervention unfreezes the last block at learning rate 1e-5. Source freeze SHA256: 384861266ca5c3d3c8681c2d6f827ec8d7c3be627a97f6f61726b89c3a7dded7. The frozen full-model SHA256 equals the published seed0 B a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77; the unfrozen full-model SHA256 is beb4b0b67af2e17cc447bc583f216a29e2bbd4646549d4c4757fd20f91f56a85.

## Weak20 increment

Preparation selected 20 machine-fulltext-consensus items from 32 candidates: 11 training and nine development items, with 12 quarantined. The treatment preserved the original 146 recordings, 880 original presentations and fixed 220 optimizer steps, adding 220 low-weight auxiliary presentations at weight 0.5 with the original batch denominator of four (effective auxiliary coefficient 0.125). Global gradient clipping was active on 194/220 steps. It did not modify the original seed0 B checkpoint.

Across 884 compared clips, the 32 known matching-candidate clips remained 32; no known candidate was gained, and one unknown candidate disappeared. Crucially, a historically single-user whole-phrase-confirmed TTS positive remained present but regressed from score 0.966317785 to 0.238842981, with its decision/alignment final frame changing from 138 to 135. Greedy text remained correct; these are not the same four-node event. Unchanged candidate-presence counts therefore do not establish preserved positive confidence or event structure.

New20 full-greedy exact matches rose from 13/20 to 15/20, with training 7/11→9/11 and development unchanged at 6/9; ordinary-word exact matches rose from 93/640 to 94/640. Fulltext CTC decreased on 60/70 and increased on 10/70. Noncanonical weak-development event scores moved in both directions, including an increase of about 0.060. There is no demonstrated consistent separation, operating-point improvement or independent FAR improvement. The original seed0 B research control is retained; the increment is not promoted.

Author aggregate SHA256: f95677724177ef3a5a95fd0fa3a10c69d26a27b65d0545132c62343a73f95040. Independent aggregate SHA256: 087a81b59d2422113f458f416fc3f92a220ed9edff6cb2fcb33c30f154266432; review PASS. Source freeze SHA256: 92410324128cc3f30dde5e9cc1849bd77543d2552d8b7ae8ee2187dd03fa823e. Treatment head SHA256: 89ec8169a4270bce4358a6ba91a6f9159f5784f37fd60699d8daa2e15926b926; treatment full model SHA256: 531d2cd724f5e3b790fb669eb6bb48c3b0a4e6e0956a6e886c46030aed17e746. These weights are not included.

Weak agreement labels and a small development split do not establish human-gold accuracy, independent FAR/FRR, final qualification, or broad generalization.

## End-to-end C diagnostic

On a fixed three-recording, 63.84-second historical diagnostic, two target events matched in all non-score fields and all three recording summaries matched. The 60-second background segment remained at zero events in both implementations. Both target scores differed in strict binary64, with absolute differences from 1.370397836319981e-7 to 4.175485994739603e-7. All three complete-posterior cumulative hashes differed; 2,027/2,155 rows differed in the sparse selected fields. No new independent negative hours were collected. This does not clear strict numerical parity: the acoustic/frontend numerical failures remain unresolved. It must not be conflated with the separately published [decoder-only exact-posterior contract](../fsmn2599-three-window-c/README.md).

Public aggregate evidence SHA256: bfad6a27e8f53e9d19fda08de79160c6e08925ad684bf931f1be78df84b574ff. Its fixed chain checkpoint SHA256 is 0182290585dce024f48637604ffe0901a16e4a4744c73ebddb9f68ea27ae4ad4; exported float32 payload SHA256 is 662f3092ee160a92f4b3d5a377f557c38f641ca2289c57bd36af8155311ae2a8. This is a separate identity from the published seed0 B candidate.

## Model storage boundary

The current local research checkpoint remains privately retained. Repository [models/README.md](../../models/README.md) states that failed runs, per-round checkpoints and unselected research candidates remain ephemeral workflow artifacts rather than Git commits. Existing model-training and development workflows can retain their own generated candidates as Actions artifacts, but the inspected workflows and currently available API tools do not provide an established safe intake for this already-local checkpoint. This is not a blanket claim that every repository artifact is forbidden. No research release, policy exception, credential, shipping-registry entry, or weight upload is introduced here.

## Publication boundary

This report contains only aggregate observations, source/evidence hashes and model identities. No raw audio, weights, private item identities, per-item transcripts/results, or private download references are included. Retained evidence hashes identify evidence; they are not a claim that private inputs are publicly reproducible.
