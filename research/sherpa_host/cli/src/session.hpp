// New kws-pipeline host research glue. Apache-2.0; see LICENSE and PROVENANCE.md.
// Feed/drain/result/reset order follows the frozen reference runner; see PROVENANCE.md.
#ifndef KWS_HOST_CLI_SESSION_HPP_
#define KWS_HOST_CLI_SESSION_HPP_
#include "audio.hpp"
#include "json.hpp"
#include <memory>

namespace kws_cli {
// Both the real CLI and the model-free tests instantiate this exact helper.
// Api is a compile-time adapter, not a replacement decoder or a dynamic loader.
template <class Source, class Api, class Sink>
Summary RunSession(const Source& source, InputFormat format,
                   const SherpaOnnxKeywordSpotterConfig& config,
                   Api& api, Sink& sink) {
  const AudioLayout layout = InspectAudio(source, format);  // Before model creation.
  Summary summary{layout.file_bytes, 0, 0, 0};
  {
    auto destroy_spotter = [&api](const SherpaOnnxKeywordSpotter* p) { api.DestroySpotter(p); };
    std::unique_ptr<const SherpaOnnxKeywordSpotter, decltype(destroy_spotter)>
        spotter(api.CreateSpotter(&config), destroy_spotter);
    if (!spotter) throw std::runtime_error("keyword model creation failed");
    auto destroy_stream = [&api](const SherpaOnnxOnlineStream* p) { api.DestroyStream(p); };
    std::unique_ptr<const SherpaOnnxOnlineStream, decltype(destroy_stream)>
        stream(api.CreateStream(spotter.get()), destroy_stream);
    if (!stream) throw std::runtime_error("keyword stream creation failed");
    auto drain = [&](bool eof) {
      while (api.Ready(spotter.get(), stream.get())) {
        api.Decode(spotter.get(), stream.get());
        auto destroy_result = [&api](const SherpaOnnxKeywordResult* p) { api.DestroyResult(p); };
        std::unique_ptr<const SherpaOnnxKeywordResult, decltype(destroy_result)>
            result(api.GetResult(spotter.get(), stream.get()), destroy_result);
        if (!result) throw std::runtime_error("keyword result creation failed");
        ValidateResult(*result);
        if (*result->keyword) {
          sink.Write(EventJson(*result, summary.audio_samples, eof));
          summary.events = CheckedAdd(summary.events, 1);
          api.Reset(spotter.get(), stream.get());
        }
        // Result destruction follows Reset for a reported event, as in runner.cc.
      }
    };
    AudioBlocks<Source> blocks(source, layout);
    std::array<float, kBlockSamples> samples{};
    for (size_t n; (n = blocks.Next(samples)) != 0;) {
      api.Accept(stream.get(), 16000, samples.data(), static_cast<int32_t>(n));
      summary.audio_samples = CheckedAdd(summary.audio_samples, n);
      summary.feed_calls = CheckedAdd(summary.feed_calls, 1);
      drain(false);
    }
    api.Finished(stream.get());  // Exactly once on a fully read input.
    drain(true);
    source.CheckSize();
  }  // Stream then spotter are destroyed before the completion record.
  sink.Write(SummaryJson(summary));
  return summary;
}
}  // namespace kws_cli
#endif
