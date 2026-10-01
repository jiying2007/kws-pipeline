// New kws-pipeline host research glue. Apache-2.0; see LICENSE and PROVENANCE.md.
// See PROVENANCE.md. JSON strings preserve valid UTF-8 and reject invalid UTF-8.
#ifndef KWS_HOST_CLI_JSON_HPP_
#define KWS_HOST_CLI_JSON_HPP_

#include "audio.hpp"
#include "sherpa-onnx/c-api/c-api.h"
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

namespace kws_cli {
constexpr size_t kMaxResultTextBytes = 65536;
constexpr size_t kMaxEventBytes = 1048576;
constexpr int32_t kMaxResultTokens = 1024;

inline void ValidateUtf8(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const auto a = static_cast<unsigned char>(s[i]);
    if (a <= 0x7f) { ++i; continue; }
    size_t n;
    uint32_t value;
    uint32_t minimum;
    if (a >= 0xc2 && a <= 0xdf) { n = 2; value = a & 0x1f; minimum = 0x80; }
    else if (a >= 0xe0 && a <= 0xef) { n = 3; value = a & 0x0f; minimum = 0x800; }
    else if (a >= 0xf0 && a <= 0xf4) { n = 4; value = a & 7; minimum = 0x10000; }
    else throw std::runtime_error("result contains invalid UTF-8");
    if (n > s.size() - i) throw std::runtime_error("result contains truncated UTF-8");
    for (size_t j = 1; j < n; ++j) {
      const auto b = static_cast<unsigned char>(s[i + j]);
      if ((b & 0xc0) != 0x80) throw std::runtime_error("result contains invalid UTF-8 continuation");
      value = (value << 6) | (b & 0x3f);
    }
    if (value < minimum || value > 0x10ffff ||
        (value >= 0xd800 && value <= 0xdfff))
      throw std::runtime_error("result contains invalid UTF-8 code point");
    i += n;
  }
}

inline std::string ResultText(const char* text) {
  if (!text) throw std::runtime_error("result contains a null string");
  size_t n = 0;
  while (n <= kMaxResultTextBytes && text[n]) ++n;
  if (n > kMaxResultTextBytes) throw std::runtime_error("result string exceeds safety limit");
  std::string s(text, n);
  ValidateUtf8(s);
  return s;
}

class Json {
 public:
  void Add(const std::string& value) {
    if (value.size() > kMaxEventBytes - text_.size())
      throw std::runtime_error("result JSON exceeds safety limit");
    text_ += value;
  }
  void String(const std::string& value) {
    ValidateUtf8(value);
    static constexpr char hex[] = "0123456789abcdef";
    Add("\"");
    for (unsigned char c : value) {
      if (c == '"') Add("\\\"");
      else if (c == '\\') Add("\\\\");
      else if (c < 0x20) {
        char escaped[] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
        Add(std::string(escaped, sizeof(escaped)));
      } else Add(std::string(1, static_cast<char>(c)));
    }
    Add("\"");
  }
  void Number(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("result contains a non-finite number");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    if (!out) throw std::runtime_error("JSON number serialization failed");
    Add(out.str());
  }
  const std::string& text() const { return text_; }
 private:
  std::string text_;
};

inline void ValidateResult(const SherpaOnnxKeywordResult& r) {
  (void)ResultText(r.keyword);
  if (r.count < 0 || r.count > kMaxResultTokens || !std::isfinite(r.start_time) ||
      (r.count && (!r.tokens_arr || !r.timestamps)))
    throw std::runtime_error("malformed keyword result");
  for (int32_t i = 0; i < r.count; ++i) {
    (void)ResultText(r.tokens_arr[i]);
    if (!std::isfinite(r.timestamps[i]))
      throw std::runtime_error("result contains a non-finite timestamp");
  }
}

inline std::string EventJson(const SherpaOnnxKeywordResult& r,
                             uint64_t available_samples, bool eof) {
  ValidateResult(r);
  if (!*r.keyword || !available_samples)
    throw std::runtime_error("empty event or invalid sample availability");
  Json out;
  out.Add("{\"type\":\"event\",\"keyword\":");
  out.String(ResultText(r.keyword));
  out.Add(",\"tokens\":[");
  for (int32_t i = 0; i < r.count; ++i) {
    if (i) out.Add(",");
    out.String(ResultText(r.tokens_arr[i]));
  }
  out.Add("],\"timestamps\":[");
  for (int32_t i = 0; i < r.count; ++i) {
    if (i) out.Add(",");
    out.Number(r.timestamps[i]);
  }
  out.Add("],\"start_time\":");
  out.Number(r.start_time);
  out.Add(",\"available_samples\":" + std::to_string(available_samples));
  out.Add(eof ? ",\"eof\":true}\n" : ",\"eof\":false}\n");
  return out.text();
}

struct Summary {
  uint64_t input_bytes = 0;
  uint64_t audio_samples = 0;
  uint64_t feed_calls = 0;
  uint64_t events = 0;
};
inline std::string SummaryJson(const Summary& s) {
  return "{\"type\":\"complete\",\"input_bytes\":" + std::to_string(s.input_bytes) +
         ",\"audio_samples\":" + std::to_string(s.audio_samples) +
         ",\"feed_calls\":" + std::to_string(s.feed_calls) +
         ",\"events\":" + std::to_string(s.events) + ",\"eof\":true}\n";
}

// Unbuffered write: every byte is checked, so there is no deferred stdout flush.
// The caller must ignore SIGPIPE before the first write to make EPIPE catchable.
template <class Write>
void WriteAll(int fd, const std::string& line, Write write) {
  size_t done = 0;
  while (done < line.size()) {
    const ssize_t n = write(fd, line.data() + done, line.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0 || static_cast<size_t>(n) > line.size() - done)
      throw std::runtime_error("stdout write failed");
    done += static_cast<size_t>(n);
  }
}

class FdSink {
 public:
  explicit FdSink(int fd) : fd_(fd) {}
  void Write(const std::string& line) {
    WriteAll(fd_, line, [](int fd, const char* bytes, size_t n) { return ::write(fd, bytes, n); });
  }
 private:
  int fd_;
};
}  // namespace kws_cli
#endif
