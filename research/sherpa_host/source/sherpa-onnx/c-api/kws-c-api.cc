// sherpa-onnx/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

// Mechanically extracted subset; original function/structure bytes are retained.
#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/version.h"

#define SHERPA_ONNX_OR(x, y) (x ? x : y)

const char *SherpaOnnxGetVersionStr() { return sherpa_onnx::GetVersionStr(); }

const char *SherpaOnnxGetGitSha1() { return sherpa_onnx::GetGitSha1(); }

const char *SherpaOnnxGetOnnxruntimeVersionStr() {
  return sherpa_onnx::GetOnnxruntimeVersionStr();
}

// On WASM builds, multi-threading is not supported. Clamp num_threads to 1.
static int32_t GetNumThreads(int32_t num_threads) {
  int32_t n = SHERPA_ONNX_OR(num_threads, 1);
#if SHERPA_ONNX_ENABLE_WASM
  if (n > 1) {
    SHERPA_ONNX_LOGE(
        "WASM does not support multi-threading. "
        "Changing num_threads from %d to 1.",
        n);
    n = 1;
  }
#endif
  return n;
}

struct SherpaOnnxOnlineStream {
  std::unique_ptr<sherpa_onnx::OnlineStream> impl;
  explicit SherpaOnnxOnlineStream(std::unique_ptr<sherpa_onnx::OnlineStream> p)
      : impl(std::move(p)) {}
};

void SherpaOnnxDestroyOnlineStream(const SherpaOnnxOnlineStream *stream) {
  if (!stream) return;
  delete stream;
}

void SherpaOnnxOnlineStreamAcceptWaveform(const SherpaOnnxOnlineStream *stream,
                                          int32_t sample_rate,
                                          const float *samples, int32_t n) {
  if (!stream) return;
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

void SherpaOnnxOnlineStreamInputFinished(const SherpaOnnxOnlineStream *stream) {
  if (!stream) return;
  stream->impl->InputFinished();
}

struct SherpaOnnxKeywordSpotter {
  std::unique_ptr<sherpa_onnx::KeywordSpotter> impl;
};

static sherpa_onnx::KeywordSpotterConfig GetKeywordSpotterConfig(
    const SherpaOnnxKeywordSpotterConfig *config) {
  sherpa_onnx::KeywordSpotterConfig spotter_config;

  spotter_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);
  spotter_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  spotter_config.model_config.transducer.encoder =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");
  spotter_config.model_config.transducer.decoder =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");
  spotter_config.model_config.transducer.joiner =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  spotter_config.model_config.paraformer.encoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.encoder, "");
  spotter_config.model_config.paraformer.decoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.decoder, "");

  spotter_config.model_config.zipformer2_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer2_ctc.model, "");

  spotter_config.model_config.nemo_ctc.model =
      SHERPA_ONNX_OR(config->model_config.nemo_ctc.model, "");

  spotter_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  if (config->model_config.tokens_buf &&
      config->model_config.tokens_buf_size > 0) {
    spotter_config.model_config.tokens_buf = std::string(
        config->model_config.tokens_buf, config->model_config.tokens_buf_size);
  }

  spotter_config.model_config.num_threads =
      GetNumThreads(config->model_config.num_threads);
  spotter_config.model_config.provider_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  if (spotter_config.model_config.provider_config.provider.empty()) {
    spotter_config.model_config.provider_config.provider = "cpu";
  }

  spotter_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  spotter_config.model_config.debug = config->model_config.debug;

  spotter_config.max_active_paths = SHERPA_ONNX_OR(config->max_active_paths, 4);

  spotter_config.num_trailing_blanks =
      SHERPA_ONNX_OR(config->num_trailing_blanks, 1);

  spotter_config.keywords_score = SHERPA_ONNX_OR(config->keywords_score, 1.0);

  spotter_config.keywords_threshold =
      SHERPA_ONNX_OR(config->keywords_threshold, 0.25);

  spotter_config.keywords_file = SHERPA_ONNX_OR(config->keywords_file, "");
  if (config->keywords_buf && config->keywords_buf_size > 0) {
    spotter_config.keywords_buf =
        std::string(config->keywords_buf, config->keywords_buf_size);
  }

  if (spotter_config.model_config.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", spotter_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", spotter_config.ToString().c_str());
#endif
  }

  return spotter_config;
}

const SherpaOnnxKeywordSpotter *SherpaOnnxCreateKeywordSpotter(
    const SherpaOnnxKeywordSpotterConfig *config) {
  auto spotter_config = GetKeywordSpotterConfig(config);
  if (!spotter_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaOnnxKeywordSpotter *spotter = new SherpaOnnxKeywordSpotter;

  spotter->impl = std::make_unique<sherpa_onnx::KeywordSpotter>(spotter_config);

  return spotter;
}

void SherpaOnnxDestroyKeywordSpotter(const SherpaOnnxKeywordSpotter *spotter) {
  if (!spotter) return;
  delete spotter;
}

const SherpaOnnxOnlineStream *SherpaOnnxCreateKeywordStream(
    const SherpaOnnxKeywordSpotter *spotter) {
  if (!spotter) return nullptr;
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(spotter->impl->CreateStream());
  return stream;
}

int32_t SherpaOnnxIsKeywordStreamReady(const SherpaOnnxKeywordSpotter *spotter,
                                       const SherpaOnnxOnlineStream *stream) {
  if (!spotter || !stream) return 0;
  return spotter->impl->IsReady(stream->impl.get());
}

void SherpaOnnxDecodeKeywordStream(const SherpaOnnxKeywordSpotter *spotter,
                                   const SherpaOnnxOnlineStream *stream) {
  if (!spotter || !stream) return;
  spotter->impl->DecodeStream(stream->impl.get());
}

void SherpaOnnxResetKeywordStream(const SherpaOnnxKeywordSpotter *spotter,
                                  const SherpaOnnxOnlineStream *stream) {
  if (!spotter || !stream) return;
  spotter->impl->Reset(stream->impl.get());
}

const SherpaOnnxKeywordResult *SherpaOnnxGetKeywordResult(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream) {
  if (!spotter || !stream) return nullptr;
  const sherpa_onnx::KeywordResult &result =
      spotter->impl->GetResult(stream->impl.get());
  const auto &keyword = result.keyword;

  auto r = new SherpaOnnxKeywordResult;
  memset(r, 0, sizeof(SherpaOnnxKeywordResult));

  r->start_time = result.start_time;

  // copy keyword
  char *pKeyword = new char[keyword.size() + 1];
  std::copy(keyword.begin(), keyword.end(), pKeyword);
  pKeyword[keyword.size()] = 0;
  r->keyword = pKeyword;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *pTokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = pTokens + pos;
      memcpy(pTokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens = pTokens;
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty()) {
      r->timestamps = new float[result.timestamps.size()];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaOnnxDestroyKeywordResult(const SherpaOnnxKeywordResult *r) {
  if (r) {
    delete[] r->keyword;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

