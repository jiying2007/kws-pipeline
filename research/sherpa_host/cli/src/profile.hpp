// New kws-pipeline host research glue. Apache-2.0; see LICENSE and PROVENANCE.md.
#ifndef KWS_HOST_CLI_PROFILE_HPP_
#define KWS_HOST_CLI_PROFILE_HPP_
#include "audio.hpp"
#include "sherpa-onnx/c-api/c-api.h"
namespace kws_cli {
// Canonical, immutable descriptor, byte-matched before model creation. It is
// deliberately not an open-ended JSON configuration or a parameter search API.
inline constexpr char kProfile[] = R"PROFILE({
  "profile": "host-kws-research-v1",
  "provider": "cpu",
  "sample_rate": 16000,
  "feature_dim": 80,
  "num_threads": 1,
  "max_active_paths": 4,
  "num_trailing_blanks": 1,
  "keywords_score": 1,
  "keywords_threshold": 0.25,
  "feed_samples": 320,
  "automatic_padding_samples": 0,
  "encoder": "models/encoder.int8.onnx",
  "decoder": "models/decoder.onnx",
  "joiner": "models/joiner.onnx",
  "tokens": "models/tokens.txt",
  "keywords": "config/keywords.txt"
}
)PROFILE";

inline void CheckProfile(const std::string& root) {
  FileSource file(root + "/config/profile.json");
  constexpr size_t bytes = sizeof(kProfile) - 1;
  if (file.size() != bytes) throw std::runtime_error("fixed profile differs from this CLI");
  std::array<unsigned char, bytes> actual{};
  file.Read(0, actual.data(), actual.size());
  if (std::memcmp(actual.data(), kProfile, bytes))
    throw std::runtime_error("fixed profile differs from this CLI");
}

struct FixedConfig {
  std::string encoder, decoder, joiner, tokens, keywords;
  SherpaOnnxKeywordSpotterConfig value{};
  explicit FixedConfig(const std::string& root)
      : encoder(root + "/models/encoder.int8.onnx"),
        decoder(root + "/models/decoder.onnx"),
        joiner(root + "/models/joiner.onnx"),
        tokens(root + "/models/tokens.txt"),
        keywords(root + "/config/keywords.txt") {
    value.feat_config.sample_rate = 16000;
    value.feat_config.feature_dim = 80;
    value.model_config.transducer = {encoder.c_str(), decoder.c_str(), joiner.c_str()};
    value.model_config.tokens = tokens.c_str();
    value.model_config.num_threads = 1;
    value.model_config.provider = "cpu";
    value.model_config.debug = 0;
    value.keywords_file = keywords.c_str();
    value.max_active_paths = 4;
    value.num_trailing_blanks = 1;
    value.keywords_score = 1.0f;
    value.keywords_threshold = 0.25f;
  }
  // c_str pointers refer to this object's own strings for the full call lifetime.
  FixedConfig(const FixedConfig&) = delete;
  FixedConfig& operator=(const FixedConfig&) = delete;
};
}  // namespace kws_cli
#endif
