"""Static exact-profile gate; no package import, installation or execution."""
HISTORICAL_VERSIONS = {
    "torch": "2.12.1", "numpy": "1.26.4", "transformers": "4.57.6",
    "qwen-asr": "0.0.6", "funasr": "1.4.16", "kaldi-native-fbank": "1.22.3",
}
CPU_VERSIONS = {**HISTORICAL_VERSIONS, "torch": "2.12.1+cpu"}
CPU_WHEEL = {
    "filename": "torch-2.12.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
    "sha256": "ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df",
    "bytes": 192268841,
}
PROFILES = {"historical_pypi_build_cpu_execution": HISTORICAL_VERSIONS,
            "official_cpu_build_native_fbank": CPU_VERSIONS}

def require_runtime_profile(actual_versions, *, profile, torch_wheel=None,
                            frontend_has_torchaudio=False, frontend_has_knf=True):
    if profile not in PROFILES:
        raise RuntimeError("Unknown explicit runtime profile")
    if type(actual_versions) is not dict or actual_versions != PROFILES[profile]:
        raise RuntimeError("Exact runtime version/build mismatch; suffixes are significant")
    if frontend_has_torchaudio is not False or frontend_has_knf is not True:
        raise RuntimeError("Reviewed native-fbank recipe required; no frontend substitution")
    if profile == "official_cpu_build_native_fbank" and torch_wheel != CPU_WHEEL:
        raise RuntimeError("The exact CPU wheel identity must match its retained hash lock")
    return {"profile": profile, "versions": dict(actual_versions),
            "frontend_backend": "kaldi-native-fbank", "model_execution_authorized": False,
            "remaining_gate": "Verify all installed package bodies, source locks and run contract separately"}
