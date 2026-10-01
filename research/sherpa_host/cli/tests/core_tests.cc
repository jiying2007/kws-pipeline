// Model-free CLI helper tests. No model or native KWS implementation is linked.
// New kws-pipeline test glue. Apache-2.0; see the repository LICENSE.
#include "audio.hpp"
#include "json.hpp"
#include "options.hpp"
#include "profile.hpp"
#include "session.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include <signal.h>

// Fake handles complete the public header's opaque declarations only in this TU.
struct SherpaOnnxKeywordSpotter {};
struct SherpaOnnxOnlineStream {};

namespace {
using namespace kws_cli;
using Bytes = std::vector<unsigned char>;
void Need(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template<class F> void Reject(F function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  Need(rejected, "expected rejection did not occur");
}
void U16(Bytes& b, uint16_t x) { b.push_back(x & 255); b.push_back(x >> 8); }
void U32(Bytes& b, uint32_t x) { for (int i = 0; i < 4; ++i) b.push_back((x >> (i * 8)) & 255); }
void Set32(Bytes& b, size_t at, uint32_t x) {
  for (int i = 0; i < 4; ++i) b.at(at + i) = (x >> (i * 8)) & 255;
}
Bytes Format(bool extended = false) {
  Bytes b;
  U16(b, 1); U16(b, 1); U32(b, 16000); U32(b, 32000); U16(b, 2); U16(b, 16);
  if (extended) U16(b, 0);
  return b;
}
Bytes Chunk(const char* id, const Bytes& payload, bool pad = true) {
  Bytes out(id, id + 4); U32(out, static_cast<uint32_t>(payload.size()));
  out.insert(out.end(), payload.begin(), payload.end());
  if (pad && (payload.size() & 1)) out.push_back(0xa5);
  return out;
}
Bytes Wave(const std::vector<Bytes>& chunks) {
  Bytes out{'R','I','F','F',0,0,0,0,'W','A','V','E'};
  for (const auto& c : chunks) out.insert(out.end(), c.begin(), c.end());
  Set32(out, 4, static_cast<uint32_t>(out.size() - 8));
  return out;
}
struct MemorySource {
  Bytes bytes;
  uint64_t declared_size;
  mutable size_t largest_read = 0, read_count = 0;
  size_t fail_read = 0;
  bool size_changed = false;
  explicit MemorySource(Bytes b) : bytes(std::move(b)), declared_size(bytes.size()) {}
  uint64_t size() const { return declared_size; }
  void CheckSize() const {
    if (size_changed) throw std::runtime_error("fake source length changed");
  }
  void Read(uint64_t offset, unsigned char* out, size_t n) const {
    CheckSize();
    ++read_count;
    if (fail_read && read_count == fail_read) throw std::runtime_error("fake truncated read");
    largest_read = std::max(largest_read, n);
    if (CheckedAdd(offset, n) > bytes.size()) throw std::runtime_error("fake truncated read");
    std::memcpy(out, bytes.data() + offset, n);
  }
};
void ParserTests() {
  const auto fmt = Chunk("fmt ", Format());
  const auto data = Chunk("data", Bytes{0, 128, 255, 127});
  const auto base = Wave({fmt, data});
  auto valid = [&](Bytes b, uint64_t samples) {
    MemorySource source(std::move(b));
    const auto layout = InspectAudio(source, InputFormat::wav);
    Need(layout.data_bytes == samples * 2, "wrong parsed sample count");
    Need(source.largest_read <= 18, "header scan buffer became unbounded");
  };
  auto invalid = [&](Bytes b) {
    MemorySource source(std::move(b));
    Reject([&] { InspectAudio(source, InputFormat::wav); });
    Need(source.largest_read <= 18, "invalid chunk caused a large header read");
  };
  valid(base, 2);
  valid(Wave({Chunk("fmt ", Format(true)), data}), 2);
  valid(Wave({data, Chunk("JUNK", Bytes{1,2,3}), fmt}), 2);
  valid(Wave({Chunk("JUNK", {}), fmt, data, Chunk("LIST", {1})}), 2);
  invalid({}); invalid(Wave({})); invalid(Wave({fmt})); invalid(Wave({data}));
  invalid(Wave({fmt, fmt, data})); invalid(Wave({fmt, data, data}));
  invalid(Wave({fmt, Chunk("data", {})}));
  invalid(Wave({fmt, Chunk("data", {1})}));
  invalid(Wave({fmt, data, Chunk("JUNK", {1}, false)}));
  invalid(Wave({fmt, data, Bytes{'J','U','N','K'}}));
  for (size_t i = 0; i < base.size(); ++i) invalid(Bytes(base.begin(), base.begin() + i));
  for (const auto& magic : {"RIFX", "RF64"}) {
    auto b = base; std::memcpy(b.data(), magic, 4); invalid(b);
  }
  { auto b = base; b[8] = 'X'; invalid(b); }
  { auto b = base; b.push_back(0); invalid(b); }
  { auto b = base; Set32(b, 4, UINT32_MAX); invalid(b); }
  { auto b = base; Set32(b, 4, 0); invalid(b); }
  { auto b = base; Set32(b, 16, UINT32_MAX); invalid(b); }
  { auto b = base; b.pop_back(); Set32(b, 4, b.size() - 8); invalid(b); }
  for (size_t size : {size_t(0), size_t(15), size_t(17), size_t(19), size_t(40)}) {
    auto f = Format(); f.resize(size); invalid(Wave({Chunk("fmt ", f), data}));
  }
  { auto f = Format(true); f[16] = 1; invalid(Wave({Chunk("fmt ", f), data})); }
  // Format, channels, sample rate, byte rate, block align and bit depth.
  for (size_t offset : {size_t(0), size_t(2), size_t(4), size_t(8), size_t(12), size_t(14)}) {
    auto f = Format(); f[offset] ^= 1; invalid(Wave({Chunk("fmt ", f), data}));
  }
  { auto f = Format(); f[0] = 0xfe; f[1] = 0xff; invalid(Wave({Chunk("fmt ", f), data})); }
  MemorySource empty({}), odd({1}), raw({0, 0});
  Reject([&] { InspectAudio(empty, InputFormat::pcm_s16le_16000_mono); });
  Reject([&] { InspectAudio(odd, InputFormat::pcm_s16le_16000_mono); });
  Need(InspectAudio(raw, InputFormat::pcm_s16le_16000_mono).data_bytes == 2, "PCM length");
  raw.declared_size = kMaxInputBytes;
  Need(InspectAudio(raw, InputFormat::pcm_s16le_16000_mono).data_bytes == kMaxInputBytes,
       "exact 1 GiB bound was rejected");  // Fake metadata only; no huge allocation.
  raw.declared_size = kMaxInputBytes + 1;
  Reject([&] { InspectAudio(raw, InputFormat::pcm_s16le_16000_mono); });
  Need(CheckedAdd(UINT64_MAX - 1, 1) == UINT64_MAX, "checked-add exact boundary");
  Reject([] { CheckedAdd(UINT64_MAX, 1); });
  Reject([] { CheckedAdd(UINT64_MAX - 1, 2); });
}

struct FakeApi {
  std::vector<std::string>& actions;
  int spotters = 0, streams = 0, results = 0, resets = 0, finishes = 0;
  int pending = 0, event_every = 0, decoded = 0;
  bool null_spotter = false, null_stream = false, null_result = false;
  bool invalid_utf8 = false, nonfinite = false;
  std::string fail;
  std::vector<std::vector<float>> feeds;
  float stamp = 0.125f;
  const char* token[1] = {"你"};
  explicit FakeApi(std::vector<std::string>& a) : actions(a) {}
  void Action(const std::string& name) {
    actions.push_back(name);
    if (name == fail) throw std::runtime_error("fake API exception");
  }
  const SherpaOnnxKeywordSpotter* CreateSpotter(const SherpaOnnxKeywordSpotterConfig* c) {
    Action("create_spotter");
    Need(c->model_config.num_threads == 1, "thread count drift");
    if (null_spotter) return nullptr;
    ++spotters; return new SherpaOnnxKeywordSpotter;
  }
  void DestroySpotter(const SherpaOnnxKeywordSpotter* p) {
    actions.push_back("destroy_spotter"); --spotters; delete p;
  }
  const SherpaOnnxOnlineStream* CreateStream(const SherpaOnnxKeywordSpotter*) {
    Action("create_stream");
    if (null_stream) return nullptr;
    ++streams; return new SherpaOnnxOnlineStream;
  }
  void DestroyStream(const SherpaOnnxOnlineStream* p) {
    actions.push_back("destroy_stream"); --streams; delete p;
  }
  void Accept(const SherpaOnnxOnlineStream*, int32_t rate, const float* samples, int32_t n) {
    Action("accept"); Need(rate == 16000 && n > 0 && n <= 320, "invalid feed geometry");
    feeds.emplace_back(samples, samples + n); pending = 1;
  }
  void Finished(const SherpaOnnxOnlineStream*) { Action("finished"); ++finishes; pending = 1; }
  int32_t Ready(const SherpaOnnxKeywordSpotter*, const SherpaOnnxOnlineStream*) {
    Action("ready"); return pending;
  }
  void Decode(const SherpaOnnxKeywordSpotter*, const SherpaOnnxOnlineStream*) {
    Action("decode"); --pending; ++decoded;
  }
  const SherpaOnnxKeywordResult* GetResult(const SherpaOnnxKeywordSpotter*, const SherpaOnnxOnlineStream*) {
    Action("get_result");
    if (null_result) return nullptr;
    auto* result = new SherpaOnnxKeywordResult{}; ++results;
    result->keyword = event_every && decoded % event_every == 0 ? "你好\"\\\n" : "";
    if (invalid_utf8) result->keyword = "\xc0\xaf";
    result->start_time = nonfinite ? std::numeric_limits<float>::infinity() : 0.25f;
    result->tokens_arr = token; result->timestamps = &stamp; result->count = 1;
    return result;
  }
  void Reset(const SherpaOnnxKeywordSpotter*, const SherpaOnnxOnlineStream*) { Action("reset"); ++resets; }
  void DestroyResult(const SherpaOnnxKeywordResult* result) {
    actions.push_back("destroy_result"); --results; delete result;
  }
  void Clean() const { Need(spotters == 0 && streams == 0 && results == 0, "owned API object leaked"); }
};
struct FakeSink {
  std::vector<std::string>& actions;
  std::vector<std::string> lines;
  int calls = 0, fail_call = 0;
  explicit FakeSink(std::vector<std::string>& a) : actions(a) {}
  void Write(const std::string& line) {
    if (++calls == fail_call) { actions.push_back("write_failed"); throw std::runtime_error("fake write failure"); }
    actions.push_back(line.find("\"complete\"") != std::string::npos ? "complete" : "event");
    lines.push_back(line);
  }
};
Bytes Pcm321() {
  Bytes out;
  for (size_t i = 0; i < 321; ++i) U16(out, static_cast<uint16_t>(i == 0 ? 32768 : i == 1 ? 65535 : i == 2 ? 0 : i == 3 ? 1 : 32767));
  return out;
}
void SessionTests() {
  FixedConfig config("/test kit");
  Need(config.value.feat_config.sample_rate == 16000 && config.value.feat_config.feature_dim == 80 &&
       config.value.max_active_paths == 4 && config.value.num_trailing_blanks == 1 &&
       config.value.keywords_score == 1.0f && config.value.keywords_threshold == 0.25f &&
       config.value.model_config.debug == 0, "fixed model profile drift");
  for (int event_every : {0, 1, 3}) {
    std::vector<std::string> actions;
    MemorySource source(Pcm321()); FakeApi api(actions); FakeSink sink(actions); api.event_every = event_every;
    const auto summary = RunSession(source, InputFormat::pcm_s16le_16000_mono, config.value, api, sink);
    api.Clean();
    Need(summary.audio_samples == 321 && summary.feed_calls == 2 && api.feeds.size() == 2,
         "stream geometry changed");
    Need(api.feeds[0].size() == 320 && api.feeds[1].size() == 1 && api.finishes == 1,
         "last short block padded or EOF repeated");
    Need(api.feeds[0][0] == -1.0f && api.feeds[0][1] == -1.0f / 32768.0f &&
         api.feeds[0][2] == 0.0f && api.feeds[0][3] == 1.0f / 32768.0f &&
         api.feeds[0][4] == 32767.0f / 32768.0f, "s16le conversion changed");
    std::vector<std::string> expected{"create_spotter", "create_stream"};
    int round = 0;
    for (const char* entry : {"accept", "accept", "finished"}) {
      ++round;
      expected.insert(expected.end(), {entry, "ready", "decode", "get_result"});
      if (event_every && round % event_every == 0) expected.insert(expected.end(), {"event", "reset"});
      expected.insert(expected.end(), {"destroy_result", "ready"});
    }
    expected.insert(expected.end(), {"destroy_stream", "destroy_spotter", "complete"});
    Need(actions == expected, "feed/decode/result/output/reset/destruction order drift");
    const int expected_events = event_every ? 3 / event_every : 0;
    Need(summary.events == static_cast<uint64_t>(expected_events) && api.resets == expected_events, "event/reset count");
    if (event_every == 1) {
      Need(sink.lines[0].find("\"available_samples\":320,\"eof\":false") != std::string::npos, "first availability");
      Need(sink.lines[1].find("\"available_samples\":321,\"eof\":false") != std::string::npos, "last short block availability");
      Need(sink.lines[2].find("\"available_samples\":321,\"eof\":true") != std::string::npos, "EOF availability");
    }
    if (event_every == 3)
      Need(sink.lines[0].find("\"available_samples\":321,\"eof\":true") != std::string::npos, "EOF-only event");
    Need(source.largest_read <= 640, "feed read became unbounded");
  }
  // Every error below must unwind acquired handles and omit successful completion.
  for (const std::string failure : {"bad_input", "null_spotter", "null_stream", "null_result", "invalid_utf8",
       "nonfinite", "read", "event_write", "completion_write", "accept", "ready", "decode", "get_result", "reset", "finished"}) {
    std::vector<std::string> actions;
    MemorySource source(failure == "bad_input" ? Bytes{} : Pcm321());
    FakeApi api(actions); FakeSink sink(actions);
    api.event_every = failure == "completion_write" ? 0 : 1;
    api.null_spotter = failure == "null_spotter";
    api.null_stream = failure == "null_stream";
    api.null_result = failure == "null_result";
    api.invalid_utf8 = failure == "invalid_utf8";
    api.nonfinite = failure == "nonfinite";
    source.fail_read = failure == "read" ? 2 : 0;
    if (failure == "event_write" || failure == "completion_write") sink.fail_call = 1;
    api.fail = failure;
    Reject([&] { RunSession(source, InputFormat::pcm_s16le_16000_mono, config.value, api, sink); });
    api.Clean();
    Need(std::find(actions.begin(), actions.end(), "complete") == actions.end(), "error claimed completion");
    if (failure == "bad_input") Need(actions.empty(), "model created before input validation");
    if (failure == "event_write") Need(api.resets == 0 && api.finishes == 0, "write failure continued decoding");
    if (failure == "read") Need(api.finishes == 0 && sink.lines.size() == 1, "mid-read failure lost partial-event semantics");
  }
}

void JsonTests() {
  Json text;
  std::string controls;
  for (int i = 0; i < 32; ++i) controls += static_cast<char>(i);
  text.String(controls + "\"\\你好😀");
  Need(text.text().find("\\u0000") != std::string::npos && text.text().find("\\u001f") != std::string::npos &&
       text.text().find("\\\"\\\\你好😀") != std::string::npos, "JSON string escaping");
  for (const std::string& bad : {std::string("\x80"), std::string("\xc0\xaf"), std::string("\xe0\x80\xaf"),
       std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"), std::string("\xf5\x80\x80\x80"),
       std::string("\xc2"), std::string("\xe2\x28\xa1")}) {
    Reject([&] { Json out; out.String(bad); });
  }
  ValidateUtf8(std::string("\xc2\x80\xdf\xbf\xe0\xa0\x80\xed\x9f\xbf\xef\xbf\xbf\xf0\x90\x80\x80\xf4\x8f\xbf\xbf"));
  for (float value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
    Reject([&] { Json out; out.Number(value); });
  }
  float timestamps[] = {0.125f}; const char* tokens[] = {"你"};
  SherpaOnnxKeywordResult result{};
  result.keyword = "你好\"\\\n"; result.count = 1; result.tokens_arr = tokens;
  result.timestamps = timestamps; result.start_time = 0.25f;
  Need(EventJson(result, 320, false) ==
      "{\"type\":\"event\",\"keyword\":\"你好\\\"\\\\\\u000a\",\"tokens\":[\"你\"],\"timestamps\":[0.125],\"start_time\":0.25,\"available_samples\":320,\"eof\":false}\n",
      "event schema/value preservation mismatch");
  result.count = -1; Reject([&] { ValidateResult(result); });
  result.count = 1; result.tokens_arr = nullptr; Reject([&] { ValidateResult(result); });
  result.tokens_arr = tokens; result.timestamps = nullptr; Reject([&] { ValidateResult(result); });
  result.timestamps = timestamps; timestamps[0] = std::numeric_limits<float>::quiet_NaN();
  Reject([&] { ValidateResult(result); });
  Reject([] { Json out; out.Add(std::string(kMaxEventBytes + 1, 'x')); });
  std::string too_long(kMaxResultTextBytes + 1, 'x'); Reject([&] { ResultText(too_long.c_str()); });
}

void OptionsTests() {
  const char* a[] = {"kws", "--pcm-s16le-16000-mono", "--", "-audio with spaces.pcm"};
  const auto parsed = ParseOptions(4, a);
  Need(parsed.file == "-audio with spaces.pcm" && parsed.format == InputFormat::pcm_s16le_16000_mono, "argument mutation");
  const char* b[] = {"kws", "a b.wav"};
  Need(ParseOptions(2, b).file == "a b.wav", "relative path mutation");
  const char* c[] = {"kws", "--wav", "--wav", "f.wav"}; Reject([&] { ParseOptions(4, c); });
  const char* d[] = {"kws", "a.wav", "b.wav"}; Reject([&] { ParseOptions(3, d); });
  const char* e[] = {"kws", "--threshold", "0.2"}; Reject([&] { ParseOptions(3, e); });
  const char* f[] = {"kws", "--"}; Reject([&] { ParseOptions(2, f); });
}

void FileAndOutputTests() {
  // Deterministic fake syscall schedules exercise the same WriteAll helper used
  // by FdSink, including EINTR followed by partial writes and partial failure.
  std::string captured;
  int write_calls = 0;
  WriteAll(123, "abcdef\n", [&](int fd, const char* bytes, size_t n) -> ssize_t {
    Need(fd == 123, "write descriptor changed");
    if (++write_calls == 1) { errno = EINTR; return -1; }
    const size_t count = std::min<size_t>(2, n);
    captured.append(bytes, count); return static_cast<ssize_t>(count);
  });
  Need(captured == "abcdef\n" && write_calls == 5, "short-write/EINTR handling");
  for (int error : {EPIPE, EIO, ENOSPC}) {
    int call = 0;
    Reject([&] { WriteAll(123, "abcdef\n", [&](int, const char*, size_t) -> ssize_t {
      if (++call == 1) return 2;
      errno = error; return -1;
    }); });
    Need(call == 2, "write failure was retried or ignored");
  }
  Reject([] { WriteAll(123, "x", [](int, const char*, size_t) -> ssize_t { return 0; }); });
  Reject([] { FileSource file("/dev/null"); });
  Reject([] { FileSource file("/tmp"); });
  const char* tmpdir = std::getenv("TMPDIR");
  Need(tmpdir && *tmpdir, "TMPDIR must name the existing test scratch directory");
  struct stat tmpdir_stat{};
  Need(::stat(tmpdir, &tmpdir_stat) == 0 && S_ISDIR(tmpdir_stat.st_mode),
       "TMPDIR must name the existing test scratch directory");
  std::string path_storage = std::string(tmpdir) + "/kws-cli-test-XXXXXX";
  char* path = path_storage.data();
  const int raw_fd = ::mkstemp(path);
  Need(raw_fd >= 0, "cannot create test file");
  Fd test_fd(raw_fd);
  struct RemoveFile { const char* p; ~RemoveFile() { ::unlink(p); } } remove{path};
  Reject([&] { FileSource file(path); });
  const auto pcm = Pcm321();
  Need(::write(test_fd.get(), pcm.data(), pcm.size()) == static_cast<ssize_t>(pcm.size()), "cannot write fixture");
  FileSource file(path);
  const auto layout = InspectAudio(file, InputFormat::pcm_s16le_16000_mono);
  AudioBlocks<FileSource> blocks(file, layout); std::array<float, kBlockSamples> samples{};
  Need(blocks.Next(samples) == 320, "real file block mismatch");
  Need(::ftruncate(test_fd.get(), 640) == 0, "cannot truncate fixture");
  Reject([&] { blocks.Next(samples); });
  const std::string fifo = std::string(path) + ".fifo";
  Need(::mkfifo(fifo.c_str(), 0600) == 0, "cannot create FIFO fixture");
  RemoveFile remove_fifo{fifo.c_str()};
  Reject([&] { FileSource input(fifo); });
  Reject([] { FdSink output(-1); output.Write("x\n"); });
  struct sigaction ignore{}, old{}; ignore.sa_handler = SIG_IGN;
  Need(::sigemptyset(&ignore.sa_mask) == 0 && ::sigaction(SIGPIPE, &ignore, &old) == 0, "SIGPIPE setup");
  int pipe_fds[2]; Need(::pipe(pipe_fds) == 0, "cannot create output pipe");
  ::close(pipe_fds[0]); Fd pipe_writer(pipe_fds[1]);
  Reject([&] { FdSink output(pipe_writer.get()); output.Write("x\n"); });
  Need(::sigaction(SIGPIPE, &old, nullptr) == 0, "SIGPIPE restore");
  Fd full(::open("/dev/full", O_WRONLY | O_CLOEXEC));
  Reject([&] { FdSink output(full.get()); output.Write("x\n"); });
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Need(argc == 2, "test requires CLI source root for frozen profile/fixture checks");
    ParserTests(); SessionTests(); JsonTests(); OptionsTests(); FileAndOutputTests();
    CheckProfile(argv[1]);
    FileSource fixture(std::string(argv[1]) + "/tests/fixtures/valid tiny.wav");
    Need(InspectAudio(fixture, InputFormat::wav).data_bytes == 10, "static WAV fixture mismatch");
    std::cout << "model-free core tests passed\n";
    return std::cout ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "test failure: " << e.what() << '\n'; return 1;
  }
}
