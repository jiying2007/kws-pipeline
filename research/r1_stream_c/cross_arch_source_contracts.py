"""Standard-library-only source contracts for the R1 cross-architecture lane."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "research/r1_stream_c/cross_arch_validate.py"
RECOVERY = ROOT / "research/r1_stream_c/cross_arch_recover_inputs.py"
CAPTURE = ROOT / "research/r1_stream_c/cross_arch_capture.c"
BUILD = ROOT / "research/r1_stream_c/build_cross_arch.sh"
DOC = ROOT / "research/r1_stream_c/CROSS_ARCH.md"


def read(path):
    return path.read_text(encoding="utf-8")


def parse_assignments(source):
    tree = ast.parse(source)
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        result[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return tree, result


class CrossArchSourceContracts(unittest.TestCase):
    def setUp(self):
        self.validator_text = read(VALIDATOR)
        self.recovery_text = read(RECOVERY)
        self.capture_text = read(CAPTURE)
        self.build_text = read(BUILD)
        self.doc_text = read(DOC)

    def test_validator_is_valid_python_without_importing_optional_packages(self):
        tree = ast.parse(self.validator_text)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "main"
                            for node in tree.body))
        self.assertIn("numpy", self.validator_text)
        self.assertNotIn("exec(", self.validator_text)

    def test_model_payload_size_is_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["MODEL_BYTES"], 3028860)

    def test_frontend_payload_size_is_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["TABLE_BYTES"], 83840)

    def test_checkpoint_identity_is_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["CHECKPOINT_SHA256"],
                         "99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571")

    def test_model_and_frontend_hashes_are_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["MODEL_SHA256"],
                         "e9a3c3360bab0485b309eb2a3ef50869a8e917f50f28dbfd30673f57a88c86bb")
        self.assertEqual(values["TABLE_SHA256"],
                         "49c42226c617fad81c5edb683a6332bdb82b52e37040f724eac23522a504db49")

    def test_feature_numerical_bounds_are_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["FEATURE_ATOL"], 1e-8)
        self.assertEqual(values["FEATURE_RTOL"], 1e-8)
        self.assertEqual(values["FEATURE_MAX_ABS"], 1e-8)

    def test_logit_numerical_bounds_are_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["LOGIT_ATOL"], 1e-8)
        self.assertEqual(values["LOGIT_RTOL"], 1e-8)
        self.assertEqual(values["LOGIT_MAX_ABS"], 1e-6)

    def test_posterior_numerical_bounds_are_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["POSTERIOR_ATOL"], 1e-8)
        self.assertEqual(values["POSTERIOR_RTOL"], 1e-8)
        self.assertEqual(values["POSTERIOR_MAX_ABS"], 1e-8)

    def test_capture_header_and_record_geometry_are_fixed(self):
        _, values = parse_assignments(self.validator_text)
        self.assertEqual(values["MAGIC"], b"R1DREF1\0")
        self.assertIn('FEATURE_STRUCT = struct.Struct("<400d")', self.validator_text)
        self.assertIn('LOGIT_STRUCT = struct.Struct("<2601d")', self.validator_text)
        self.assertIn("FRAME_BYTES = CLOCK_STRUCT.size + FEATURE_STRUCT.size + LOGIT_STRUCT.size",
                      self.validator_text)

    def test_input_manifest_schema_is_fixed(self):
        self.assertIn('"kws-r1-arm-input-recovery-v1"', self.validator_text)
        self.assertIn('"kws-r1-arm-input-recovery-v1"', self.recovery_text)

    def test_input_manifest_is_bounded_to_256_rows(self):
        self.assertRegex(self.validator_text, r"len\(records\)\s*<=\s*256")
        self.assertRegex(self.validator_text, r"0\s*<=\s*index\s*<\s*256")

    def test_only_declared_train_and_development_roles_are_accepted(self):
        self.assertIn('("train", "development")', self.validator_text)
        self.assertIn('("train", "development")', self.recovery_text)

    def test_duplicate_source_ids_are_rejected(self):
        self.assertIn("source_id not in ids", self.validator_text)
        self.assertIn("source_id not in ids", self.recovery_text)

    def test_pcm_size_and_hash_are_verified_before_replay(self):
        self.assertIn('sha256_file(pcm) == record["pcm_sha256"]', self.validator_text)
        self.assertIn('sha_bytes(pcm) == binding["pcm_sha256"]', self.recovery_text)

    def test_original_capsule_and_model_card_are_hash_pinned(self):
        _, values = parse_assignments(self.recovery_text)
        self.assertEqual(values["CAPSULE_SHA256"],
                         "84048654b73de0d6da955472eacaff608b7a54534d0f48c035b56c9b46b9a6bc")
        self.assertEqual(values["MODEL_CARD_SHA256"],
                         "2079fce70bcd27d2991ce558807b3ef9707309051e6915946fc01821f8350936")

    def test_original_pcm_format_is_not_resampled_or_relabelled(self):
        self.assertIn("(1, 2, 16000)", self.recovery_text)
        self.assertIn("WAV PCM length mismatch", self.recovery_text)
        self.assertNotIn("resample(", self.recovery_text)

    def test_recovery_is_offline_and_does_not_invoke_external_tools(self):
        for forbidden in ("urllib", "requests", "subprocess", "http.client"):
            self.assertNotIn(forbidden, self.recovery_text)
        self.assertIn('"raw_audio_published": False', self.recovery_text)

    def test_recovery_and_validation_refuse_overwrite(self):
        self.assertIn("output must be a fresh absolute directory", self.recovery_text)
        self.assertIn("output must be a fresh absolute directory", self.validator_text)
        self.assertIn('open("xb")', self.recovery_text)
        self.assertIn('open("xb")', self.validator_text)

    def test_architecture_output_paths_are_distinct(self):
        self.assertIn('args.output / "x86"', self.validator_text)
        self.assertIn('args.output / "arm"', self.validator_text)

    def test_source_dependency_and_delivery_clocks_are_checked(self):
        self.assertIn("source dependency clock", self.validator_text)
        self.assertIn("delivery clock", self.validator_text)
        self.assertIn("source clock mismatch", self.validator_text)

    def test_eof_does_not_add_synthetic_frames(self):
        self.assertIn('"synthetic_eof") is False', self.validator_text)
        self.assertIn("r1d_finish(stream)", self.capture_text)
        self.assertIn('\\\"synthetic_eof\\\":false', self.capture_text)
        self.assertNotIn("pad", self.capture_text.lower())

    def test_feature_logit_and_posterior_checks_are_separate(self):
        self.assertIn("feature numerical contract failed", self.validator_text)
        self.assertIn("logit numerical contract failed", self.validator_text)
        self.assertIn("posterior numerical contract failed", self.validator_text)

    def test_argmax_and_greedy_sequence_are_checked(self):
        self.assertIn("cross-architecture frame argmax mismatch", self.validator_text)
        self.assertIn("saved greedy sequence mismatch", self.validator_text)
        self.assertIn("greedy_sequence", self.validator_text)

    def test_capture_driver_uses_the_fixed_stream_api(self):
        self.assertIn("r1d_create", self.capture_text)
        self.assertIn("r1d_feed", self.capture_text)
        self.assertIn("r1d_finish", self.capture_text)
        self.assertIn("r1d_state_bytes", self.capture_text)

    def test_capture_driver_contains_no_threshold_or_wake_label_logic(self):
        self.assertNotIn("threshold", self.capture_text.lower())
        self.assertNotIn("wake event", self.capture_text.lower())
        self.assertNotIn("keyword", self.capture_text.lower())

    def test_build_script_compiles_both_targets_with_strict_fp_flags(self):
        self.assertIn("arm-linux-gnueabihf-gcc", self.build_text)
        self.assertIn("gcc", self.build_text)
        self.assertIn("-fno-fast-math", self.build_text)
        self.assertIn("-ffp-contract=off", self.build_text)
        self.assertIn("-mfpu=neon-vfpv4", self.build_text)
        self.assertIn("-mfloat-abi=hard", self.build_text)

    def test_build_script_keeps_weights_and_audio_out_of_build(self):
        self.assertIn("numerics.py", self.build_text)
        self.assertIn("--mode reference64", self.build_text)
        self.assertIn("weights", self.build_text)
        self.assertIn("qemu-arm-static", self.build_text)
        self.assertIn("model_weights_loaded", self.build_text)
        self.assertIn("private_audio_loaded", self.build_text)

    def test_documentation_preserves_emulator_and_device_boundary(self):
        self.assertIn("not a physical SSC305", self.doc_text)
        self.assertIn("not a calibrated event FAR/FRR", self.doc_text)
        self.assertIn("original FP32", self.doc_text)
        self.assertIn("build_cross_arch.sh", self.doc_text)


if __name__ == "__main__":
    unittest.main()
