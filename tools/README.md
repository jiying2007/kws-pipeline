# Engineering utilities

Use this index to find the existing command or shared helper. Paths are stable;
this organization does not create wrappers, relocate scripts, or activate a run.
Some Python modules are shared libraries rather than standalone commands.

- [Build and SDK usage](../README.md#build-and-install)
- [Documentation and product contracts](../docs/README.md)
- [Isolated research modules and their tests](../research/README.md)
- [Contributor checks](../CONTRIBUTING.md#required-local-checks)

The real-time library remains under `src/` and `include/`. Python dependencies
and corpus/model processing belong to offline tools. Scripts for measurement,
training, private data or promotion retain their own input and permission gates;
an entry in this index grants no new execution authority. Product-evidence tools
make collected measurements auditable and do not substitute for those measurements.

## Runtime command-line programs and C helpers

- [`kws_board_bench.c`](kws_board_bench.c)
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

- [`compile_keywords.py`](compile_keywords.py)
- [`corpus_identity.py`](corpus_identity.py)
- [`keyword_set_identity.py`](keyword_set_identity.py)
- [`kws_vocab.py`](kws_vocab.py)
- [`validate_shipping_keywords.py`](validate_shipping_keywords.py)
- [`verify_corpus_identity.py`](verify_corpus_identity.py)

## Speech-like source preparation and admission

- [`attach_speech_like_labels.py`](attach_speech_like_labels.py)
- [`bootstrap_speech_like_stage_a.py`](bootstrap_speech_like_stage_a.py)
- [`generate_speech_like_command_provider.py`](generate_speech_like_command_provider.py)
- [`materialize_speech_like_base_index.py`](materialize_speech_like_base_index.py)
- [`materialize_speech_like_provider.py`](materialize_speech_like_provider.py)
- [`run_speech_like_corpus_generation.py`](run_speech_like_corpus_generation.py)
- [`speech_like_corpus_plan.py`](speech_like_corpus_plan.py)
- [`speech_label_admission.py`](speech_label_admission.py) — Validate supplied actual-label review receipts; it does not authenticate a listener.
- [`speech_like_vits_resample_adapter.py`](speech_like_vits_resample_adapter.py)
- [`validate_speech_like_synthetic.py`](validate_speech_like_synthetic.py)
- [`verify_speech_like_backend_bundle.py`](verify_speech_like_backend_bundle.py)
- [`verify_speech_like_runtime_bundle.py`](verify_speech_like_runtime_bundle.py)

## Diagnostics, replay and comparisons

- [`build_decoder_boundary_product_references.py`](build_decoder_boundary_product_references.py)
- [`build_decoder_boundary_references.py`](build_decoder_boundary_references.py)
- [`build_training_diagnostics.py`](build_training_diagnostics.py)
- [`compare_dataset_iterations.py`](compare_dataset_iterations.py)
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
- [`requalify_decoder_boundary_grid.py`](requalify_decoder_boundary_grid.py)
- [`summarize_frozen_far_context_path.py`](summarize_frozen_far_context_path.py)

## Product evidence and qualification

- [`collect_runtime_soak.py`](collect_runtime_soak.py)
- [`collect_target_evidence.py`](collect_target_evidence.py)
- [`final_afe_identity.py`](final_afe_identity.py)
- [`qualification_common.py`](qualification_common.py)
- [`qualification_gate.py`](qualification_gate.py)
- [`qualification_manifest.py`](qualification_manifest.py)
- [`qualification_metrics.py`](qualification_metrics.py)
- [`run_dataset_iteration.py`](run_dataset_iteration.py)
- [`run_final_afe_corpus.py`](run_final_afe_corpus.py)
- [`runtime_soak_contract.py`](runtime_soak_contract.py) — Versioned single-core CPU evidence validation.
- [`score_real_human_qualification.py`](score_real_human_qualification.py)
- [`score_target_cohort.py`](score_target_cohort.py)
- [`score_target_dut_qualification.py`](score_target_dut_qualification.py)
- [`seal_real_human_corpus.py`](seal_real_human_corpus.py)
- [`validate_real_human_corpus.py`](validate_real_human_corpus.py)
- [`validate_real_human_development_corpus.py`](validate_real_human_development_corpus.py)

## Model and training provenance

- [`kws_landing_status.py`](kws_landing_status.py)
- [`materialize_model_registry.py`](materialize_model_registry.py)
- [`model_family_resource_estimate.py`](model_family_resource_estimate.py)
- [`model_provenance.py`](model_provenance.py)
- [`verify_model_promotion_bundle.py`](verify_model_promotion_bundle.py)
- [`verify_model_registry.py`](verify_model_registry.py)
- [`verify_model_release.py`](verify_model_release.py)

## Repository verification and maintenance

- [`cleanup_bootstrap_branch.sh`](cleanup_bootstrap_branch.sh) — Exact-trigger-SHA cleanup after successful publication.
- [`check_bench_signal.py`](check_bench_signal.py)
- [`check_deferred_verdicts.py`](check_deferred_verdicts.py)
- [`check_gcov.py`](check_gcov.py)
- [`check_reproducible_sdk.py`](check_reproducible_sdk.py)
- [`check_runtime_purity.py`](check_runtime_purity.py)
- [`check_workflow_path_filters.py`](check_workflow_path_filters.py)
- [`gen_parameter_limits.py`](gen_parameter_limits.py)
- [`generate_sbom.py`](generate_sbom.py)
- [`validate_sbom.py`](validate_sbom.py) — validate the generated SDK SPDX profile and installed file identities
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
