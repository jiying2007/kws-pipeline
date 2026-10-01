// New kws-pipeline host research glue. Apache-2.0; see LICENSE and PROVENANCE.md.
// Uses the unchanged Sherpa public header and only the selected KWS C API.
#include "options.hpp"
#include "profile.hpp"
#include "session.hpp"
#include <cstdlib>
#include <iostream>
#include <signal.h>

namespace kws_cli {
class CApi {
 public:
  const SherpaOnnxKeywordSpotter* CreateSpotter(const SherpaOnnxKeywordSpotterConfig* c) {
    const char* sherpa_version = SherpaOnnxGetVersionStr();
    const char* ort_version = SherpaOnnxGetOnnxruntimeVersionStr();
    if (!sherpa_version || !ort_version || std::strcmp(sherpa_version, "1.13.8") ||
        std::strcmp(ort_version, "1.28.2"))
      throw std::runtime_error("runtime version differs from the fixed profile");
    return SherpaOnnxCreateKeywordSpotter(c);
  }
  void DestroySpotter(const SherpaOnnxKeywordSpotter* k) { SherpaOnnxDestroyKeywordSpotter(k); }
  const SherpaOnnxOnlineStream* CreateStream(const SherpaOnnxKeywordSpotter* k) {
    return SherpaOnnxCreateKeywordStream(k);
  }
  void DestroyStream(const SherpaOnnxOnlineStream* s) { SherpaOnnxDestroyOnlineStream(s); }
  void Accept(const SherpaOnnxOnlineStream* s, int32_t rate, const float* f, int32_t n) {
    SherpaOnnxOnlineStreamAcceptWaveform(s, rate, f, n);
  }
  void Finished(const SherpaOnnxOnlineStream* s) { SherpaOnnxOnlineStreamInputFinished(s); }
  int32_t Ready(const SherpaOnnxKeywordSpotter* k, const SherpaOnnxOnlineStream* s) {
    return SherpaOnnxIsKeywordStreamReady(k, s);
  }
  void Decode(const SherpaOnnxKeywordSpotter* k, const SherpaOnnxOnlineStream* s) {
    SherpaOnnxDecodeKeywordStream(k, s);
  }
  const SherpaOnnxKeywordResult* GetResult(const SherpaOnnxKeywordSpotter* k,
                                         const SherpaOnnxOnlineStream* s) {
    return SherpaOnnxGetKeywordResult(k, s);
  }
  void Reset(const SherpaOnnxKeywordSpotter* k, const SherpaOnnxOnlineStream* s) {
    SherpaOnnxResetKeywordStream(k, s);
  }
  void DestroyResult(const SherpaOnnxKeywordResult* r) { SherpaOnnxDestroyKeywordResult(r); }
};

inline void SetEnvironment() {
  for (const char* key : {"ORT_DISABLE_TELEMETRY", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                          "NUMEXPR_NUM_THREADS"}) {
    if (::setenv(key, "1", 1)) throw std::runtime_error("cannot set fixed runtime environment");
  }
  if (::setenv("OMP_DYNAMIC", "FALSE", 1) || ::setenv("MKL_DYNAMIC", "FALSE", 1))
    throw std::runtime_error("cannot set fixed runtime environment");
}

inline std::string KitRoot() {
  std::array<char, 4096> path{};
  const ssize_t n = ::readlink("/proc/self/exe", path.data(), path.size());
  if (n <= 0 || static_cast<size_t>(n) >= path.size())
    throw std::runtime_error("cannot resolve CLI executable path");
  const std::string executable(path.data(), static_cast<size_t>(n));
  const std::string suffix = "/bin/kws";
  if (executable.size() <= suffix.size() ||
      executable.compare(executable.size() - suffix.size(), suffix.size(), suffix))
    throw std::runtime_error("CLI must reside at kit-root/bin/kws");
  return executable.substr(0, executable.size() - suffix.size());
}

}  // namespace kws_cli

int main(int argc, char** argv) {
  try {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "IEEE float32 is required");
    struct sigaction action{};
    action.sa_handler = SIG_IGN;
    if (::sigemptyset(&action.sa_mask) || ::sigaction(SIGPIPE, &action, nullptr))
      throw std::runtime_error("cannot make stdout pipe errors detectable");
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cerr << "Usage: run-kws [--wav | --pcm-s16le-16000-mono] [--] FILE\n"
                   "WAV default: PCM1 mono 16000 Hz s16le. One regular file, at most 1 GiB.\n";
      return std::cerr ? 0 : 1;
    }
    const auto options = kws_cli::ParseOptions(argc, argv);
    kws_cli::FileSource input(options.file);  // Keeps caller-relative pathname semantics.
    kws_cli::SetEnvironment();
    const std::string root = kws_cli::KitRoot();
    kws_cli::CheckProfile(root);
    kws_cli::FixedConfig config(root);
    kws_cli::FdSink sink(STDOUT_FILENO);
    kws_cli::CApi api;
    (void)kws_cli::RunSession(input, options.format, config.value, api, sink);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "kws: " << e.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "kws: unexpected failure\n";
    return 1;
  }
}
