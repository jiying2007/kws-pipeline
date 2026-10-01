// New kws-pipeline host research glue. Apache-2.0; see LICENSE and PROVENANCE.md.
#ifndef KWS_HOST_CLI_OPTIONS_HPP_
#define KWS_HOST_CLI_OPTIONS_HPP_
#include "audio.hpp"
namespace kws_cli {
struct Options { InputFormat format = InputFormat::wav; std::string file; };
inline Options ParseOptions(int argc, const char* const* argv) {
  Options result;
  bool literal = false, format_given = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (!literal && arg == "--") { literal = true; continue; }
    if (!literal && (arg == "--wav" || arg == "--pcm-s16le-16000-mono")) {
      if (format_given) throw std::runtime_error("specify the input format only once");
      result.format = arg == "--wav" ? InputFormat::wav : InputFormat::pcm_s16le_16000_mono;
      format_given = true;
    } else {
      if (!literal && !arg.empty() && arg[0] == '-')
        throw std::runtime_error("unknown option; use -- before a filename beginning with -");
      if (!result.file.empty() || arg.empty())
        throw std::runtime_error("specify exactly one nonempty input filename");
      result.file = arg;
    }
  }
  if (result.file.empty()) throw std::runtime_error("specify exactly one input filename");
  return result;
}
}  // namespace kws_cli
#endif
