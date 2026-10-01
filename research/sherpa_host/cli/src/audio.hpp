// New kws-pipeline host research glue. Apache-2.0; see LICENSE and PROVENANCE.md.
// See PROVENANCE.md. This is not a replacement for the upstream public header.
#ifndef KWS_HOST_CLI_AUDIO_HPP_
#define KWS_HOST_CLI_AUDIO_HPP_

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kws_cli {
constexpr uint64_t kMaxInputBytes = UINT64_C(1073741824);
constexpr size_t kBlockSamples = 320;

inline uint64_t CheckedAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::runtime_error("integer overflow in input bounds");
  return a + b;
}

inline uint16_t Le16(const unsigned char* p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
inline uint32_t Le32(const unsigned char* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {
    if (fd_ < 0) throw std::runtime_error("file descriptor operation failed");
  }
  ~Fd() { if (fd_ >= 0) ::close(fd_); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int get() const { return fd_; }
 private:
  int fd_;
};

// One open regular file, reused for the header scan and all PCM reads. O_NONBLOCK
// prevents an accidental FIFO argument from blocking before the regular-file check.
class FileSource {
 public:
  explicit FileSource(const std::string& path)
      : fd_(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)) {
    struct stat st{};
    if (::fstat(fd_.get(), &st) || !S_ISREG(st.st_mode) || st.st_size < 0)
      throw std::runtime_error("input must be a readable regular file");
    size_ = static_cast<uint64_t>(st.st_size);
    if (!size_ || size_ > kMaxInputBytes)
      throw std::runtime_error("input must contain 1 to 1073741824 bytes");
  }
  uint64_t size() const { return size_; }
  void CheckSize() const {
    struct stat st{};
    if (::fstat(fd_.get(), &st) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) != size_)
      throw std::runtime_error("input length changed during processing");
  }
  void Read(uint64_t offset, unsigned char* out, size_t n) const {
    CheckSize();
    if (CheckedAdd(offset, n) > size_ ||
        offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
      throw std::runtime_error("input read outside validated bounds");
    size_t done = 0;
    while (done < n) {
      const uint64_t at = CheckedAdd(offset, done);
      if (at > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::runtime_error("input offset cannot be represented");
      const ssize_t got = ::pread(fd_.get(), out + done, n - done,
                                  static_cast<off_t>(at));
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) throw std::runtime_error("input read failed or was truncated");
      done += static_cast<size_t>(got);
    }
    CheckSize();
  }
 private:
  Fd fd_;
  uint64_t size_ = 0;
};

enum class InputFormat { wav, pcm_s16le_16000_mono };
struct AudioLayout {
  uint64_t file_bytes;
  uint64_t data_offset;
  uint64_t data_bytes;
};

template <class Source>
AudioLayout InspectAudio(const Source& source, InputFormat format) {
  const uint64_t bytes = source.size();
  if (!bytes || bytes > kMaxInputBytes)
    throw std::runtime_error("input must contain 1 to 1073741824 bytes");
  source.CheckSize();
  if (format == InputFormat::pcm_s16le_16000_mono) {
    if (bytes & 1) throw std::runtime_error("s16le PCM byte count must be even");
    return {bytes, 0, bytes};
  }
  if (bytes < 12) throw std::runtime_error("truncated WAV RIFF header");
  std::array<unsigned char, 18> b{};
  source.Read(0, b.data(), 12);
  if (std::memcmp(b.data(), "RIFF", 4) ||
      std::memcmp(b.data() + 8, "WAVE", 4))
    throw std::runtime_error("input requires little-endian RIFF/WAVE");
  const uint64_t end = CheckedAdd(8, Le32(b.data() + 4));
  if (end != bytes) throw std::runtime_error("RIFF length differs from file length");
  bool have_fmt = false, have_data = false;
  AudioLayout layout{bytes, 0, 0};
  for (uint64_t pos = 12; pos < end;) {
    const uint64_t payload = CheckedAdd(pos, 8);
    if (payload > end) throw std::runtime_error("truncated WAV chunk header");
    source.Read(pos, b.data(), 8);
    const uint64_t length = Le32(b.data() + 4);
    const uint64_t payload_end = CheckedAdd(payload, length);
    const uint64_t next = CheckedAdd(payload_end, length & 1);
    if (payload_end > end || next > end)
      throw std::runtime_error("truncated WAV chunk or odd-chunk padding");
    if (!std::memcmp(b.data(), "fmt ", 4)) {
      if (have_fmt) throw std::runtime_error("duplicate WAV fmt chunk");
      have_fmt = true;
      if (length != 16 && length != 18)
        throw std::runtime_error("WAV fmt must be 16 bytes or 18 with cbSize zero");
      source.Read(payload, b.data(), static_cast<size_t>(length));
      if (Le16(b.data()) != 1 || Le16(b.data() + 2) != 1 ||
          Le32(b.data() + 4) != 16000 || Le32(b.data() + 8) != 32000 ||
          Le16(b.data() + 12) != 2 || Le16(b.data() + 14) != 16 ||
          (length == 18 && Le16(b.data() + 16) != 0))
        throw std::runtime_error("WAV requires PCM1, mono, 16000 Hz, s16le, byteRate 32000, blockAlign 2");
    } else if (!std::memcmp(b.data(), "data", 4)) {
      if (have_data) throw std::runtime_error("duplicate WAV data chunk");
      have_data = true;
      if (!length || (length & 1))
        throw std::runtime_error("WAV data must be nonempty with an even byte count");
      layout.data_offset = payload;
      layout.data_bytes = length;
    }
    // Unknown chunks are skipped by checked offset, never allocated by size.
    // A pad byte may have any value, but must exist and be readable.
    if (length & 1) source.Read(payload_end, b.data(), 1);
    pos = next;
  }
  if (!have_fmt || !have_data) throw std::runtime_error("WAV requires exactly one fmt and one data chunk");
  source.CheckSize();
  return layout;
}

template <class Source>
class AudioBlocks {
 public:
  AudioBlocks(const Source& source, AudioLayout layout)
      : source_(source), layout_(layout) {}
  size_t Next(std::array<float, kBlockSamples>& samples) {
    if (used_ == layout_.data_bytes) { source_.CheckSize(); return 0; }
    const size_t bytes = static_cast<size_t>(std::min<uint64_t>(
        kBlockSamples * 2, layout_.data_bytes - used_));
    std::array<unsigned char, kBlockSamples * 2> raw{};
    source_.Read(CheckedAdd(layout_.data_offset, used_), raw.data(), bytes);
    for (size_t i = 0; i < bytes / 2; ++i) {
      const uint16_t encoded = Le16(raw.data() + 2 * i);
      // Avoid implementation-defined narrowing from unsigned to int16_t.
      const int32_t signed_sample = encoded <= 32767 ? encoded : int32_t(encoded) - 65536;
      samples[i] = static_cast<float>(signed_sample) / 32768.0f;
    }
    used_ = CheckedAdd(used_, bytes);
    return bytes / 2;
  }
 private:
  const Source& source_;
  AudioLayout layout_;
  uint64_t used_ = 0;
};
}  // namespace kws_cli
#endif
