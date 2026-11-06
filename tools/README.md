# Engineering utilities

Use this index to find the existing command or shared helper. Paths are stable;
this organization does not create wrappers, relocate scripts, or activate a run.
Some Python modules are shared libraries rather than standalone commands.

- [Build and SDK usage](../README.md#build-and-install)
- [Research documentation and frozen historical contracts](../docs/README.md)
- [Isolated research modules and their tests](../research/README.md)
- [Contributor checks](../CONTRIBUTING.md#required-local-checks)

The real-time library remains under `src/` and `include/`. Python dependencies
and corpus/model processing belong to offline tools. Scripts for measurement, training and private data retain their own source, rights
and input gates; an index entry grants no new execution authority. Historical
qualification-format tools below are for offline evidence replay and numerical
boundary testing, not product approval or publication.

## Runtime command-line programs and C helpers

- [`kws_board_bench.c`](kws_board_bench.c)
- [`kws_bench_statistics.h`](kws_bench_statistics.h)
- [`kws_decoder_path_replay.c`](kws_decoder_path_replay.c)
- [`kws_decoder_replay.c`](kws_decoder_replay.c)
- [`kws_feature_dump.c`](kws_feature_dump.c)
- [`kws_posterior_dump.c`](kws_posterior_dump.c)
- [`kws_raw_stream.c`](kws_raw_stream.c)
- [`kws_timeline.h`](kws_timeline.h)
- [`kws_trace_io.c`](kws_trace_io.c)
- [`kws_trace_io.h`](kws_trace_io.h)
- [`kws_trace_slice.c`](kws_trace_slice.c)
- [`kws_wav.c`](kws_wav.c)
- [`sha256.c`](sha256.c)
- [`sha256.h`](sha256.h)
- [`tool_io.c`](tool_io.c)
- [`tool_io.h`](tool_io.h)

## Keyword and corpus identity

- [`private_candidate_host_smoke.py`](private_candidate_host_smoke.py): explicit local-only research transport checks; default NOT_RUN, no private input in CI. [Usage and limits](../docs/private-candidate-host-smoke.md).

- [`compile_keywords.py`](compile_keywords.py)
- [`corpus_identity.py`](corpus_identity.py)
- [`source_identity.py`](source_identity.py) — Shared recording/source-family namespace; unrelated identity domains remain separate.
- [`keyword_set_identity.py`](keyword_set_identity.py)
- [`kws_vocab.py`](kws_vocab.py)
- [`validate_research_keyword_pair.py`](validate_research_keyword_pair.py)
- [`verify_corpus_identity.py`](verify_corpus_identity.py)

## Speech-like source preparation and admission

- [`authorized_voice_clone_batch.py`](authorized_voice_clone_batch.py) — Offline, rights-scoped bounded multi-speaker cloning intents; no automatic training or qualification. [Private-generation guide](../docs/AUTHORIZED_VOICE_CLONING.md).
- [`voice_clone_effect_audit.py`](voice_clone_effect_audit.py) — Verify actual private clone WAV/PCM and source-family yield; optional 3-ASR *screen* is never a training or shipping admission. [Effect-driven data redesign](../docs/DATA_EFFECTIVENESS_REDESIGN_2026-11-02.md).
- [`kws_data_bulk_effect_boundary.py`](kws_data_bulk_effect_boundary.py) — Independent, non-admitting PCM readback of canonical `kws-data` bulk Qwen clone/ASR screens; no duplicate model execution. [Cross-repo runbook](../docs/KWS_DATA_BULK_EFFECT_BOUNDARY.md).
- [`phase_a_field_plan_preflight.py`](phase_a_field_plan_preflight.py) — Compare frozen private `kws-data` field-pilot shards with the source-locked research-only human coverage quotas; always zero real-audio qualification. [Coverage and sharding runbook](../docs/PHASE_A_FIELD_PLAN_PREFLIGHT.md).
- [`plan_phase_a_capture_supplement.py`](plan_phase_a_capture_supplement.py) — Produce private, bounded source-bound playback/double-talk/motion wake and long-negative field requests from real Phase-A deficits; requests never count as audio evidence. [Capture worksheet](../docs/PHASE_A_CRITICAL_CAPTURE_SUPPLEMENT.md).
- [`prioritize_kws_data_feedback.py`](prioritize_kws_data_feedback.py) — Turn genuine independent bulk-source screening gaps into ranked, non-authorizing development work; no TTS/ASR/training run. [Feedback runbook](../docs/KWS_DATA_FEEDBACK_PRIORITIES.md).
- [`asr_native_contracts.py`](asr_native_contracts.py) — Validate frozen native-ASR configuration and capture metadata without running recognition.
- [`asr_source_resolver.py`](asr_source_resolver.py) — Verify retained evidence against code-owned source roots; production train roots are currently empty.
- [`attach_speech_like_labels.py`](attach_speech_like_labels.py)
- [`bootstrap_speech_like_stage_a.py`](bootstrap_speech_like_stage_a.py)
- [`generate_speech_like_command_provider.py`](generate_speech_like_command_provider.py)
- [`materialize_speech_like_base_index.py`](materialize_speech_like_base_index.py)
- [`materialize_speech_like_provider.py`](materialize_speech_like_provider.py)
- [`run_speech_like_corpus_generation.py`](run_speech_like_corpus_generation.py)
- [`speech_like_corpus_plan.py`](speech_like_corpus_plan.py)
- [`speech_label_admission.py`](speech_label_admission.py) — Validate supplied actual-label review receipts; it does not authenticate a listener.
- [`source_lineage_admission.py`](source_lineage_admission.py) — Current source-alias binding on top of the existing receipt checks, preserving historical validator pins.
- [`speech_asr_consensus.py`](speech_asr_consensus.py) — Check prospective three-ASR machine-label consistency; pending external authentication is never training admission or qualification gold.
- [`speech_like_vits_resample_adapter.py`](speech_like_vits_resample_adapter.py)
- [`validate_speech_like_synthetic.py`](validate_speech_like_synthetic.py)
- [`verify_speech_like_backend_bundle.py`](verify_speech_like_backend_bundle.py)
- [`verify_speech_like_runtime_bundle.py`](verify_speech_like_runtime_bundle.py)

## Saved-speech research queue

- [`import_saved_speech_result.py`](import_saved_speech_result.py) — Import preserved native results without running ASR or TTS.
- [`speech_data_queue.py`](speech_data_queue.py) — Validate saved evidence, isolate uncertain items and export machine-weak research datasets.
- [Research queue and weak-loader guide](../docs/speech-data-queue.md) — Research-only inputs; machine agreement is not human gold or product qualification.

## Source-language-conditioned saved results

- [`import_language_conditioned_result.py`](import_language_conditioned_result.py) — Preserve and import existing language-conditioned result archives offline.
- [`language_conditioned_vote.py`](language_conditioned_vote.py) — Validate archived native token, prompt, source and PCM bindings.
- [`speech_history_evidence.py`](speech_history_evidence.py) — Retain prior attempts and check bounded source-rights evidence.
- [Language-conditioned weak-import guide](../docs/language-conditioned-weak-import.md) — Source language is supplied conditioning, not independent language detection; research weak labels only.

## Bounded batch saved-result candidates

- [`batch_weak_import.py`](batch_weak_import.py) — Offline bounded native batch verification and full-transcript research-candidate metadata/PCM inspection.
- [`batch-source-contract.json`](batch-source-contract.json) — Public source/model identities and explicit tokenizer metadata map.
- [Bounded batch weak-import guide](../docs/batch-weak-import.md) — Restricted Dylan/Serena source review, original split and full transcript preservation; current experiment admission remains false.

## Active research runtime

Active host Python and the `core`, `tts`, and `asr` dependency roles are defined
in [`research-runtime.json`](../configs/runtime/research-runtime.json). Use
[`research_runtime.py`](research_runtime.py) to check the interpreter, inspect a
role's package pins, or run a script through the checked entry. See the
[research runtime guide](../docs/RESEARCH_RUNTIME.md) for environment setup and
the separate model-smoke validation status.

- [`setup_asr_runtime.py`](setup_asr_runtime.py) — Install the shared ASR role from retained package, source and model hashes.
- [`active_asr_batch.py`](active_asr_batch.py) — Bind existing audio and resume a bounded private ASR batch without repeating completed calls.
- [`active_asr_selective.py`](active_asr_selective.py) — Run two audio-only primary recognizers and the predeclared selective Whisper audit with explicit unrequested observations.
- [`active_tts_expand.py`](active_tts_expand.py) — Continue the frozen D1 roster in bounded jobs while retaining raw failures and preventing repeated generation calls.
- [`active_tts_continuation.py`](active_tts_continuation.py) — Continue only never-claimed cells after a preserved timeout, pause for verified recognition, and keep returned audio separate from consumed unknown calls.
- [`research_tts_partial.py`](research_tts_partial.py) — Verify the original timed-out generation job in its exact isolated source environment without a model call.
- [`research_tts_handoff.py`](research_tts_handoff.py) — Replay the actual recognition seal with the retained data verifier before allowing the final generation jobs.
- [Accounted TTS checkpoint contract](../docs/tts-accounted-checkpoint-v2.md) — Bind partial returns and completed continuation jobs to immutable owners.
- [`active_asr_worker.py`](active_asr_worker.py) — Run one native recognizer process and retain its raw output and call claim.
- [`summarize_active_asr_batch.py`](summarize_active_asr_batch.py) — Re-read the complete model/clip matrix and produce the file inventory and resource summary.
- [Active ASR batch guide](../docs/ACTIVE_ASR_BATCH.md) — Decoder conditions, private output layout and weak-label boundaries.
- [`run_zipformer_system.py`](run_zipformer_system.py) — Execute a pinned native Zipformer system on verified shared PCM, preserving per-clip claims and results.
- [`run_d1_zipformer_system.py`](run_d1_zipformer_system.py) — Bind native D1 observations to a real completed preparation, retain every raw event, and separate known-label train/development diagnostics.
- [`run_d1_zipformer_extension.py`](run_d1_zipformer_extension.py) — Account for all 256 planned cells, reuse the original 64 native observations, and call the fixed system once for each additional observable cell.
- [`run_d1_zipformer_continuation.py`](run_d1_zipformer_continuation.py) — Continue only never-claimed native inputs after an immutable timeout, preserve consumed unknown predictions, and bind one separately budgeted attempt.
- [`research_d1_native_coverage.py`](research_d1_native_coverage.py) — Verify the complete planned roster against observed rows, explicit returned exclusions, and consumed unknown calls without inventing audio or labels.
- [`research_d1_previous_native.py`](research_d1_previous_native.py) — Read original native and queue evidence in a fresh isolated process with its exact frozen source hashes.
- [Native accounted D1 guide](../docs/research-d1-zipformer-accounted-v3.md) — Authenticate partial observation provenance and report returned audio separately from consumed unknown calls.
- [D1 partial observation inputs](../docs/research-d1-partial-observation-v3.md) — Keep the fixed training recipe and original development comparison while admitting only actual observed audio.
- [Historical native D1 extension guide](../docs/research-d1-zipformer-extension-v2.md) — Separate original and new execution identities, retain failures, and report the common development subset independently.
- [`research_job_lock.py`](research_job_lock.py) — Share one explicit local heavy-model slot across direct and supervised workers.
- [`research_queue_supervisor.py`](research_queue_supervisor.py) — Supervise an immutable local queue with heartbeat evidence and preserve failed or unknown model claims.
- [`research_d2_generate.py`](research_d2_generate.py) — Generate only declared D2 additions and authenticate native returns before ASR export.
- [`research_d2_campaign.py`](research_d2_campaign.py) — Complete each bounded generation, selective-ASR and weak-view chunk before advancing the same frozen campaign.
- [`research_d2_checkpoints.py`](research_d2_checkpoints.py) — Save and verify per-phase evidence capsules; require external-save receipts and preserve ambiguous launch intents across restore.
- [`research_d2_recovery.py`](research_d2_recovery.py) — Reconcile authenticated archived D2 receipts after runtime loss and exclude unrecorded outcomes from resubmission.
- [`research_context_generate.py`](research_context_generate.py) — Run independently declared context supplements without repeating D1 or D2 cells.
- [`research_context_v2_generate.py`](research_context_v2_generate.py) — Generate an explicit fixed-vocabulary context revision in separate bounded chunks while retaining its original parent.
- [`research_context_v3_generate.py`](research_context_v3_generate.py) — Generate new full-text requests for 24 added identities; require externally saved launch and terminal evidence for each bounded chunk.
- [`research_context_v3_campaign.py`](research_context_v3_campaign.py) — Close all nine context-v3 chunks through generation, selective ASR, complete weak-text replay and fixed full2601 projection, with durable acknowledgements between model phases.
- [`research_context_partial.py`](research_context_partial.py) — Authenticate only the complete native returns of a preserved timed-out context job without generating another utterance.
- [`research_fit_budget_compare.py`](research_fit_budget_compare.py) — Verify paired tiny-fit input, optimizer and consumed-step controls when only the declared exposure budget changes; constructs no model.
- [Native Zipformer system guide](../docs/ZIPFORMER_SYSTEM.md) — Fixed profile, actual D0 execution and limits on interpreting native events.
- [`setup_tts_runtime.py`](setup_tts_runtime.py) — Install and verify the isolated TTS environment and fixed Base model assets.
- [`active_tts_smoke.py`](active_tts_smoke.py) — Freeze eight source-bound D1 cells and retain raw generation output before technical checks.
- [`active_tts_pair.py`](active_tts_pair.py) — Run the separate predeclared reference-mode comparison with source and rights revalidation.
- [`research_no_network.py`](research_no_network.py) — Disable inference telemetry and restrict a new Linux worker before importing native model libraries.

[`research_python_entry.py`](research_python_entry.py) runs preserved research
drivers after shared Python startup. It removes only the running interpreter's
single library directory before entering the driver; preload, audit, empty and
additional loader paths remain errors. The preserved driver guards stay intact.

[`research_queue.py`](research_queue.py) freezes explicitly budgeted local jobs,
advances on verified prerequisite outputs and preserves claims, logs and terminal
receipts. The shared `research_job_lock.py` helper uses one configurable
heavy-model flock with direct or supervised training/evaluation entrypoints.
The [local queue guide](../docs/LOCAL_RESEARCH_QUEUE.md) explains completion,
unknown outcomes, actual elapsed-time accounting and the inherited process-group
contract. The remote hourly watchdog is **not deployed**; these local tools do
not create a GitHub schedule or retrospectively claim earlier model executions.

## Disabled research plan metadata

- [`validate_parallel_candidate_plan.py`](validate_parallel_candidate_plan.py) checks the
  closed, disabled [`parallel candidate plan`](../configs/research/parallel-candidate-screen-2026-10-10.json).
  It reads JSON only, never authenticates source/data/license claims, and always reports
  execution, numerical admission and shipping readiness as false. Pending identities,
  seeds and budgets require a separately reviewed executable preregistration; this
  checker is not an execution gate. The earlier fixed-architecture effect-chain plan
  is retained unchanged. No model, probe, training or qualification command is added.

## Diagnostics, replay and comparisons

- [`build_decoder_boundary_product_references.py`](build_decoder_boundary_product_references.py)
- [`build_decoder_boundary_references.py`](build_decoder_boundary_references.py)
- [`build_training_diagnostics.py`](build_training_diagnostics.py)
- [`compare_dataset_iterations.py`](compare_dataset_iterations.py)
- [`dataset_iteration_inputs.py`](dataset_iteration_inputs.py) — Preserve and verify the input bundle used to rescore a saved system comparison.
- [`dataset_iteration_policy.py`](dataset_iteration_policy.py) — Apply the versioned FAR confidence-bound and comparison policy without changing historical cards.
- [`select_multilingual_kws_models.py`](select_multilingual_kws_models.py) — Rank only paired same-corpus zh/en development KWS model results by FAR/FRR confidence and CPU/RSS/latency Pareto, never shipping. [Model tournament](../docs/KWS_MULTILINGUAL_MODEL_TOURNAMENT.md).
- [`diagnose_acoustic_alignment.py`](diagnose_acoustic_alignment.py)
- [`diagnose_acoustic_boundary_segmentation.py`](diagnose_acoustic_boundary_segmentation.py)
- [`diagnose_boundary_separability.py`](diagnose_boundary_separability.py)
- [`diagnose_ctc_label_prior_gradient.py`](diagnose_ctc_label_prior_gradient.py)
- [`diagnose_ctc_token_state_blank_gradient.py`](diagnose_ctc_token_state_blank_gradient.py)
- [`diagnose_decoder_policy_replay.py`](diagnose_decoder_policy_replay.py)
- [`diagnose_decoder_retention_recalibrated_curve.py`](diagnose_decoder_retention_recalibrated_curve.py)
- [`diagnose_decoder_search_path_decomposition.py`](diagnose_decoder_search_path_decomposition.py)
- [`diagnose_decoder_selected_path_provenance.py`](diagnose_decoder_selected_path_provenance.py)
- [`diagnose_frozen_far_source_replay.py`](diagnose_frozen_far_source_replay.py)
- [`diagnose_full_speech_objective_gradient.py`](diagnose_full_speech_objective_gradient.py)
- [`diagnose_keyword_ctc_competition_gradient.py`](diagnose_keyword_ctc_competition_gradient.py)
- [`diagnose_keyword_ctc_sequence_competition.py`](diagnose_keyword_ctc_sequence_competition.py)
- [`diagnose_kws_threshold_operating_curve.py`](diagnose_kws_threshold_operating_curve.py)
- [`diagnose_sequence_margin_runtime_gap.py`](diagnose_sequence_margin_runtime_gap.py)
- [`diagnose_sequence_margin_runtime_gap_decomposition.py`](diagnose_sequence_margin_runtime_gap_decomposition.py)
- [`plan_far_stream.py`](plan_far_stream.py)
- [`event_readback_contract.py`](event_readback_contract.py): validate saved per-clip identity bindings; no execution authentication, acoustic metrics or qualification.
- [`requalify_decoder_boundary_grid.py`](requalify_decoder_boundary_grid.py)
- [`report_offline_effect.py`](report_offline_effect.py): compare saved events by evidence tier under frozen identities; no inference, production qualification or shipping approval.
- [`summarize_frozen_far_context_path.py`](summarize_frozen_far_context_path.py)

## Research measurements and historical qualification-format replay

- [`collect_runtime_soak.py`](collect_runtime_soak.py)
- [`collect_target_evidence.py`](collect_target_evidence.py)
- [`final_afe_identity.py`](final_afe_identity.py)
- [`qualification_common.py`](qualification_common.py)
- [`qualification_gate.py`](qualification_gate.py) — Historical-format offline replay only; its retained `shipping_approved` assertion is not a research acceptance gate. Exact original `--references`, `--detections` and `--board-audio` remain required to reproduce an old result.
- [`qualification_manifest.py`](qualification_manifest.py) — Retired product-format manifest producer; explicitly requires `--historical-replay-only`, never current research acceptance.
- [`qualification_metrics.py`](qualification_metrics.py)
- [`run_dataset_iteration.py`](run_dataset_iteration.py)
- [`run_final_afe_corpus.py`](run_final_afe_corpus.py)
- [`runtime_soak_contract.py`](runtime_soak_contract.py) — Versioned single-core CPU evidence validation.
- [`score_real_human_qualification.py`](score_real_human_qualification.py) — Retained strict historical Phase-A numeric scorer, `--historical-replay-only` required; independent current human/AFE research is separately rights- and evidence-bound.
- [`score_target_cohort.py`](score_target_cohort.py) — Retained physical multi-DUT evidence math, not Shipping promotion.
- [`score_target_dut_qualification.py`](score_target_dut_qualification.py) — Historical deployment-format replay; no current active Actions publisher.
- [`seal_real_human_corpus.py`](seal_real_human_corpus.py)
- [`validate_real_human_corpus.py`](validate_real_human_corpus.py)
- [`validate_real_human_development_corpus.py`](validate_real_human_development_corpus.py)

## Model and training provenance

- [`export_artifacts.py`](export_artifacts.py) — Reject input/output aliases and stage model/provenance writes before per-file replacement.

- [`materialize_model_registry.py`](materialize_model_registry.py) — Inert fail-closed tombstone retained solely because a frozen historical CI workflow still syntax-compiles this path. No GitHub Release, Shipping or model import behavior; original source remains in Git history.
- [`model_family_resource_estimate.py`](model_family_resource_estimate.py)
- [`model_provenance.py`](model_provenance.py)
- [`verify_model_promotion_source.py`](verify_model_promotion_source.py) — Authenticate the governed training producer and safely verify/extract its exact artifact carrier.
- [`verify_model_promotion_bundle.py`](verify_model_promotion_bundle.py) — Retired product-promotion manifest/checksum CLI, `--historical-replay-only` required; original strict source and metric tests retained.
- [`verify_model_registry.py`](verify_model_registry.py) — Current Git-registered immutable research comparator verifier; no Release import.
- [`verify_research_model_assets.py`](verify_research_model_assets.py)

## Repository verification and maintenance

- [`candidate_numeric_ci_changes.py`](candidate_numeric_ci_changes.py) — Select bounded candidate numeric checks while preserving existing product and training CI scopes; no model execution.
- Historical commercial release-branch cleanup is retired from the active tree; its exact source remains available in Git history. Research branch maintenance uses the current repository governance process.
- [`check_bench_signal.py`](check_bench_signal.py)
- [`check_deferred_verdicts.py`](check_deferred_verdicts.py)
- [`check_gcov.py`](check_gcov.py)
- [`check_reproducible_sdk.py`](check_reproducible_sdk.py)
- [`check_runtime_purity.py`](check_runtime_purity.py)
- [`check_workflow_path_filters.py`](check_workflow_path_filters.py)
- [`gen_parameter_limits.py`](gen_parameter_limits.py)
- [`generate_sbom.py`](generate_sbom.py)
- [`validate_sbom.py`](validate_sbom.py) — validate the generated SDK SPDX profile and installed file identities
- [`install_arm_dependencies.py`](install_arm_dependencies.py) — Bound APT setup on the ephemeral ARM CI runner; retry download-only acquisition at most once and never retry installation.
- [`prepare_eval_context_fixtures.py`](prepare_eval_context_fixtures.py)
- [`research_ci_changes.py`](research_ci_changes.py) — Select complete PR/main changes for required research CI; no model execution.
- [`run_research_source_checks.py`](run_research_source_checks.py)
- [`statistical_bounds.py`](statistical_bounds.py)
- [`test_inventory.py`](test_inventory.py)
- [`verify_development_source_binding.py`](verify_development_source_binding.py)
- [`verify_durable_trace.py`](verify_durable_trace.py) — Offline validation of the fixed PR450 evidence pointer.
- [`verify_frozen_replay_build_contract.py`](verify_frozen_replay_build_contract.py)
- [`verify_research_consolidation.py`](verify_research_consolidation.py)
- [`verify_research_publication.py`](verify_research_publication.py)
- [`verify_research_sources.py`](verify_research_sources.py)

## Verified test execution and current regression

- [`checked_unittest.py`](checked_unittest.py) — Enforce reviewed test IDs, counts, modes and outcomes; no automatic contract rewriting.
- [`far_regression_receipt.py`](far_regression_receipt.py) — Recompute saved synthetic continuous-FAR evidence, bind exact current research source/model/config identities, and reject older-green fallbacks; no product-release authorization or model training. See [research FAR regression](../docs/RESEARCH_FAR_REGRESSION.md).

### Data/model effect iteration

- [inventory_effect_data.py](inventory_effect_data.py): hash-verified recovered audio inventory and development-only split planning; does not run models.
- [audit_historical_base_metadata.py](audit_historical_base_metadata.py): read-only historical train/calibration lineage audit; never promotes diagnostic-only data.

- [link_retention_events.py](link_retention_events.py): bounded research-only episode/event sidecar linkage; association is not acoustic truth or production qualification.
