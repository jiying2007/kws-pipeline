"""Strict, stdlib-only PCM16 WAV binding for calibration.

The supported format is deliberately the exact 44-byte canonical RIFF/WAVE
header inspected in the 16 recovered inputs. No implicit decoding, metadata
chunks, resampling, trimming, channel mixing, or runtime model imports occur.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import islice
import os
import re
import stat
import struct
from typing import Iterable

SCHEMA = "cosyvoice12-pcm-binding-v1"
MAX_WAV_BYTES = 16 * 1024 * 1024
SAMPLE_RATE = 16000
_HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")
_BASENAME = re.compile(r"[A-Za-z0-9_-]+\.wav\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_ID = re.compile(r"clip-[0-9]{6}\Z")


class BindingError(ValueError):
    """An input did not satisfy its predeclared binding or strict format."""


def _integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise BindingError(f"invalid {name}")


def _sha(value: object, name: str) -> None:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise BindingError(f"invalid {name}")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pcm_descriptor(descriptor: object) -> None:
    """Validate a label-free declaration's schema and internal relationships.

    This does not establish that its referenced WAV/sample bytes exist or were
    consumed by a model. Byte acquisition and verification remain separate.
    """
    fields = {"schema", "opaque_id", "wav", "raw_pcm16", "float32_pcm",
              "raw_pcm16_descriptor_sha256", "float32_pcm_descriptor_sha256", "operations"}
    if type(descriptor) is not dict or set(descriptor) != fields:
        raise BindingError("unexpected PCM descriptor fields")
    if descriptor["schema"] != SCHEMA:
        raise BindingError("unexpected PCM descriptor schema")
    opaque_id = descriptor["opaque_id"]
    if type(opaque_id) is not str or not _OPAQUE_ID.fullmatch(opaque_id):
        raise BindingError("invalid PCM descriptor opaque ID")
    wav = descriptor["wav"]
    if type(wav) is not dict or set(wav) != {"sha256", "bytes", "header_bytes", "format"}:
        raise BindingError("unexpected WAV descriptor fields")
    _sha(wav["sha256"], "WAV SHA256")
    _integer(wav["bytes"], "WAV bytes", 46, MAX_WAV_BYTES)
    _integer(wav["header_bytes"], "WAV header bytes", 44, 44)
    if wav["format"] != "RIFF-WAVE-fmt16-PCM1-data-only":
        raise BindingError("unexpected WAV descriptor format")
    common = {"frame_count", "sample_rate_hz", "channels", "shape", "sample_order",
              "encoding", "dtype", "bytes", "sha256"}
    frames = []
    for name, width, encoding, dtype in (("raw_pcm16", 2, "signed-int16-le", "<i2"),
                                         ("float32_pcm", 4, "ieee754-binary32-le", "<f4")):
        pcm = descriptor[name]
        required = common | ({"normalization"} if width == 4 else set())
        if type(pcm) is not dict or set(pcm) != required:
            raise BindingError("unexpected sample descriptor fields")
        count = pcm["frame_count"]
        _integer(count, "PCM frame count", 1, (MAX_WAV_BYTES - 44) // 2)
        _integer(pcm["sample_rate_hz"], "PCM sample rate", SAMPLE_RATE, SAMPLE_RATE)
        _integer(pcm["channels"], "PCM channels", 1, 1)
        _integer(pcm["bytes"], "PCM bytes", width * count, width * count)
        shape = pcm["shape"]
        if type(shape) is not list or len(shape) != 1 or type(shape[0]) is not int or shape[0] != count:
            raise BindingError("PCM shape differs from frame count")
        if (pcm["sample_order"] != "unchanged" or pcm["encoding"] != encoding
                or pcm["dtype"] != dtype):
            raise BindingError("unexpected PCM encoding/order")
        if width == 4 and pcm["normalization"] != "signed_int16 / 32768.0":
            raise BindingError("unexpected PCM normalization")
        _sha(pcm["sha256"], "PCM SHA256")
        digest = descriptor[name + "_descriptor_sha256"]
        _sha(digest, "sample descriptor SHA256")
        if sha256(_canonical(pcm)) != digest:
            raise BindingError("sample descriptor digest mismatch")
        frames.append(count)
    if frames[0] != frames[1] or wav["bytes"] != 44 + 2 * frames[0]:
        raise BindingError("WAV/PCM frame and byte counts disagree")
    operations = descriptor["operations"]
    if (type(operations) is not dict or set(operations) != {"trim", "resample", "mix"}
            or any(value is not False for value in operations.values())):
        raise BindingError("unexpected PCM transformation declaration")


@dataclass(frozen=True)
class WaveExpectation:
    """Private file binding. Do not pass this object to an ASR decoder."""
    opaque_id: str
    basename: str
    file_sha256: str
    file_bytes: int
    frame_count: int
    sample_rate_hz: int = SAMPLE_RATE
    channels: int = 1
    bits_per_sample: int = 16

    def __post_init__(self) -> None:
        if type(self.opaque_id) is not str or not _OPAQUE_ID.fullmatch(self.opaque_id):
            raise BindingError("opaque_id must be cv3- plus 16 lowercase hex characters")
        if type(self.basename) is not str or not _BASENAME.fullmatch(self.basename):
            raise BindingError("basename must be a single ASCII .wav filename")
        _sha(self.file_sha256, "file SHA256")
        _integer(self.file_bytes, "file bytes", 46, MAX_WAV_BYTES)
        _integer(self.frame_count, "frame count", 1, (MAX_WAV_BYTES - 44) // 2)
        for value, name, required in ((self.sample_rate_hz, "sample rate", SAMPLE_RATE),
                                      (self.channels, "channels", 1),
                                      (self.bits_per_sample, "sample width", 16)):
            _integer(value, name, required, required)
        if self.file_bytes != 44 + 2 * self.frame_count:
            raise BindingError("predeclared bytes and frames disagree")


@dataclass(frozen=True)
class BoundPCM:
    """Immutable arrays and label-free descriptor from one verified snapshot.

    Consumers may use np.frombuffer(pcm_float32_le, dtype='<f4') later. The view
    over immutable bytes is read-only. If a runtime needs a writable array, use
    .copy(order='C') separately per decoder, then verify shape/dtype/bytes hash
    before inference. No numpy integration has been run by this module.
    """
    opaque_id: str
    pcm16_le: bytes
    pcm_float32_le: bytes
    descriptor_json: bytes
    descriptor_sha256: str

    @property
    def descriptor(self) -> dict:
        """Fresh JSON object: caller mutation cannot alter the bound descriptor."""
        return json.loads(self.descriptor_json)


def _open_root(root: str) -> int:
    # Walk every component with dir_fd + O_NOFOLLOW, not just the final file.
    # Fail closed on platforms lacking the required POSIX guarantees.
    if (type(root) is not str or not root.startswith("/") or "\x00" in root
            or (root != "/" and any(p in ("", ".", "..") for p in root.split("/")[1:]))):
        raise BindingError("root must be a lexical canonical absolute directory")
    if not all(hasattr(os, n) for n in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK")):
        raise BindingError("required POSIX nofollow flags unavailable")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise BindingError("required POSIX dir_fd support unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    current = os.open("/", flags)
    try:
        for component in root.split("/")[1:] if root != "/" else ():
            nxt = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = nxt
        return current
    except BaseException:
        os.close(current)
        raise


def _stamp(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read_verified(root: str, expected: WaveExpectation) -> bytes:
    if type(expected) is not WaveExpectation:
        raise BindingError("expected must be a WaveExpectation")
    # Revalidate in case a caller bypassed frozen dataclass protections.
    expected.__post_init__()
    root_fd = file_fd = None
    try:
        root_fd = _open_root(root)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        file_fd = os.open(expected.basename, flags, dir_fd=root_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise BindingError("input is not a regular file")
        if before.st_nlink != 1:
            raise BindingError("hard-link aliases are forbidden")
        if before.st_size != expected.file_bytes:
            raise BindingError("file size differs from predeclared size")
        # At most declared size + one byte. Never unbounded read(), seek, or mmap.
        data = bytearray()
        while len(data) < expected.file_bytes + 1:
            block = os.read(file_fd, min(65536, expected.file_bytes + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        after = os.fstat(file_fd)
        named_after = os.stat(expected.basename, dir_fd=root_fd, follow_symlinks=False)
        if _stamp(before) != _stamp(after) or _stamp(after) != _stamp(named_after):
            raise BindingError("file identity or metadata changed during read")
        if len(data) != expected.file_bytes:
            raise BindingError("short read or extra bytes")
        result = bytes(data)
        if sha256(result) != expected.file_sha256:
            raise BindingError("file SHA256 differs from predeclared SHA256")
        return result
    except OSError as exc:
        raise BindingError(f"safe file read failed (errno {exc.errno})") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if root_fd is not None:
            os.close(root_fd)


def _parse_canonical_pcm16(wav: bytes, expected: WaveExpectation) -> bytes:
    """Accept only RIFF + fmt(16, PCM=1) + data, in that order, with no extras."""
    if type(wav) is not bytes or len(wav) != expected.file_bytes:
        raise BindingError("WAV bytes length/type differs from declaration")
    if len(wav) < _HEADER.size:
        raise BindingError("truncated WAV header")
    (riff, riff_bytes, wave, fmt, fmt_bytes, encoding, channels, sample_rate,
     byte_rate, block_align, bits, data, data_bytes) = _HEADER.unpack_from(wav)
    if (riff, wave, fmt, data) != (b"RIFF", b"WAVE", b"fmt ", b"data"):
        raise BindingError("unsupported chunk tags/order or non-RIFF/WAVE")
    if riff_bytes != len(wav) - 8:
        raise BindingError("RIFF size differs from exact file boundary")
    if fmt_bytes != 16 or encoding != 1:
        raise BindingError("only 16-byte fmt with integer PCM encoding is supported")
    if (channels, sample_rate, byte_rate, block_align, bits) != (1, SAMPLE_RATE, 32000, 2, 16):
        raise BindingError("requires internally consistent 16 kHz mono PCM16")
    if data_bytes != expected.frame_count * 2 or data_bytes != len(wav) - 44:
        raise BindingError("data size/frame count differs from exact file boundary")
    return wav[44:]


def bind_wave(root: str, expected: WaveExpectation) -> BoundPCM:
    """Read and bind exactly the predeclared file to immutable decoder samples."""
    wav = _read_verified(root, expected)
    raw = _parse_canonical_pcm16(wav, expected)
    output = bytearray(expected.frame_count * 4)
    for index, (sample,) in enumerate(struct.iter_unpack("<h", raw)):
        # Every int16 / 2**15 is exact in binary32, including -1 and +32767/32768.
        struct.pack_into("<f", output, index * 4, sample / 32768.0)
    converted = bytes(output)
    common = {"frame_count": expected.frame_count, "sample_rate_hz": SAMPLE_RATE,
              "channels": 1, "shape": [expected.frame_count], "sample_order": "unchanged"}
    raw_descriptor = {**common, "encoding": "signed-int16-le", "dtype": "<i2",
                      "bytes": len(raw), "sha256": sha256(raw)}
    float_descriptor = {**common, "encoding": "ieee754-binary32-le", "dtype": "<f4",
                        "normalization": "signed_int16 / 32768.0",
                        "bytes": len(converted), "sha256": sha256(converted)}
    descriptor = {
        "schema": SCHEMA, "opaque_id": expected.opaque_id,
        "wav": {"sha256": expected.file_sha256, "bytes": expected.file_bytes,
                "header_bytes": 44, "format": "RIFF-WAVE-fmt16-PCM1-data-only"},
        "raw_pcm16": raw_descriptor,
        "raw_pcm16_descriptor_sha256": sha256(_canonical(raw_descriptor)),
        "float32_pcm": float_descriptor,
        "float32_pcm_descriptor_sha256": sha256(_canonical(float_descriptor)),
        "operations": {"trim": False, "resample": False, "mix": False},
    }
    canonical = _canonical(descriptor)
    return BoundPCM(expected.opaque_id, raw, converted, canonical, sha256(canonical))


def _bounded_items(values: Iterable, kind: str) -> tuple:
    try:
        result = tuple(islice(iter(values), 1001))
    except TypeError as exc:
        raise BindingError(f"{kind} requires an iterable") from exc
    if not result or len(result) > 1000:
        raise BindingError(f"{kind} must contain 1..1000 inputs")
    return result


def bind_batch(root: str, expectations: Iterable[WaveExpectation]) -> tuple[BoundPCM, ...]:
    """Preserve declaration order; fail the whole batch on any alias/mismatch."""
    declared = _bounded_items(expectations, "batch")
    if any(type(item) is not WaveExpectation for item in declared):
        raise BindingError("batch members must be WaveExpectation objects")
    if len({e.opaque_id for e in declared}) != len(declared):
        raise BindingError("duplicate opaque IDs")
    if len({e.basename for e in declared}) != len(declared):
        raise BindingError("duplicate filenames/path aliases")
    return tuple(bind_wave(root, e) for e in declared)


def decoder_manifest(bindings: Iterable[BoundPCM]) -> dict:
    """Separate label-free descriptor manifest; contains no paths or transcripts.

    Only bound bytes plus opaque IDs should reach decoders. The private mapping
    from opaque ID to source basename belongs in the evaluator/orchestrator.
    """
    items = _bounded_items(bindings, "manifest")
    if any(type(b) is not BoundPCM for b in items):
        raise BindingError("manifest requires 1..1000 BoundPCM objects")
    if len({b.opaque_id for b in items}) != len(items):
        raise BindingError("duplicate manifest IDs")
    clips = []
    for bound in items:
        descriptor = bound.descriptor
        validate_pcm_descriptor(descriptor)
        if (sha256(bound.descriptor_json) != bound.descriptor_sha256
                or _canonical(descriptor) != bound.descriptor_json
                or descriptor.get("opaque_id") != bound.opaque_id
                or sha256(bound.pcm16_le) != descriptor["raw_pcm16"]["sha256"]
                or sha256(bound.pcm_float32_le) != descriptor["float32_pcm"]["sha256"]):
            raise BindingError("bound descriptor or samples changed")
        clips.append({"opaque_id": bound.opaque_id, "binding_sha256": bound.descriptor_sha256,
                      "descriptor": descriptor})
    return {"schema": "asr-decoder-inputs-v1", "order": [b.opaque_id for b in items],
            "clips": clips}
