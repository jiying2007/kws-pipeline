#!/usr/bin/env python3
"""Bind a promotion to GitHub API evidence and safely unpack its exact ZIP.

Offline only. The caller fetches the canonical workflow independently by filename,
not by a workflow ID supplied by the run. Repository ID is trusted runner context.
The extracted files remain untrusted model evidence for the bundle verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import unicodedata
import zipfile
import zlib

REPOSITORY = "jiying2007/kws-pipeline"
WORKFLOW_PATH = ".github/workflows/model-training.yml"
ARTIFACT_NAME = "xiaowo-torch-domain-model"
# Explicit promotion safety budget, not a claim about every future training run.
# The retained model-749187ec1d66 manifest records a 48,851,013-byte carrier.
# Larger future holdout/stream bundles require a reviewed limit change.
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEMBERS = 200_000
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
CHUNK = 64 * 1024


class SourceRejected(ValueError):
    """Only fixed, locally defined diagnostic messages belong in this exception."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceRejected(message)


def positive_int(value: object, label: str) -> int:
    require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def cli_id(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("expected a positive decimal integer")
    return int(value)


def load_json(path: Path) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise SourceRejected("nonfinite JSON constant")

    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "API JSON exceeds size limit")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def object_value(value: object, label: str) -> dict:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def select_artifact(args) -> dict:
    require(args.repository == REPOSITORY, "unexpected repository")
    positive_int(args.repository_id, "expected repository ID")
    positive_int(args.training_run_id, "expected training run ID")
    require(re.fullmatch(r"[0-9a-f]{40}", args.expected_head_sha) is not None,
            "expected head SHA must be 40 lowercase hexadecimal characters")
    run = object_value(load_json(args.run_json), "run")
    workflow = object_value(load_json(args.workflow_json), "canonical workflow")
    workflow_id = positive_int(workflow.get("id"), "canonical workflow ID")
    require(workflow.get("path") == WORKFLOW_PATH, "canonical workflow path mismatch")
    require(positive_int(run.get("id"), "run ID") == args.training_run_id, "run ID mismatch")
    require(positive_int(run.get("workflow_id"), "run workflow ID") == workflow_id,
            "run does not belong to canonical workflow")
    require(run.get("path") == WORKFLOW_PATH, "run workflow path mismatch")
    require(run.get("status") == "completed" and run.get("conclusion") == "success",
            "training run must be completed/success")
    require(run.get("head_sha") == args.expected_head_sha, "run head SHA mismatch")
    require(run.get("head_branch") == "main", "training run must use main")
    require(run.get("event") in ("push", "workflow_dispatch"), "unsupported training event")
    for key in ("repository", "head_repository"):
        repository = object_value(run.get(key), f"run {key}")
        require(positive_int(repository.get("id"), f"{key} ID") == args.repository_id,
                f"run {key} ID mismatch")
        require(repository.get("full_name") == REPOSITORY, f"run {key} full name mismatch")

    payload = load_json(args.artifacts_json)
    pages = payload if isinstance(payload, list) else [payload]
    require(bool(pages), "empty artifact pagination")
    rows = []
    total_count = None
    for page in pages:
        page = object_value(page, "artifact page")
        count = page.get("total_count")
        require(type(count) is int and count >= 0, "invalid artifact total_count")
        require(total_count is None or total_count == count, "artifact page counts changed")
        total_count = count
        require(isinstance(page.get("artifacts"), list), "artifact page lacks artifacts array")
        rows.extend(page["artifacts"])
    require(total_count == len(rows), "artifact pagination is incomplete or inconsistent")
    ids = set()
    matches = []
    for row in rows:
        row = object_value(row, "artifact")
        artifact_id = positive_int(row.get("id"), "artifact ID")
        require(artifact_id not in ids, "duplicate artifact ID across pages")
        ids.add(artifact_id)
        if row.get("name") == ARTIFACT_NAME:
            require(type(row.get("expired")) is bool, "artifact expired flag must be Boolean")
            if row["expired"] is False:
                matches.append(row)
    require(len(matches) == 1, f"expected exactly one live {ARTIFACT_NAME} artifact")
    artifact = matches[0]
    size = positive_int(artifact.get("size_in_bytes"), "artifact size")
    require(size <= MAX_ARCHIVE_BYTES, "artifact exceeds compressed size limit")
    digest = artifact.get("digest")
    require(isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None,
            "artifact requires a canonical sha256 digest")
    binding = object_value(artifact.get("workflow_run"), "artifact workflow_run")
    for field, expected in (("id", args.training_run_id),
                            ("repository_id", args.repository_id),
                            ("head_repository_id", args.repository_id)):
        require(positive_int(binding.get(field), f"artifact workflow_run {field}") == expected,
                f"artifact workflow_run {field} mismatch")
    require(binding.get("head_sha") == args.expected_head_sha, "artifact workflow_run SHA mismatch")
    require(binding.get("head_branch") == "main", "artifact workflow_run branch mismatch")
    return artifact


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    require(len(data) == size, "truncated ZIP record")
    return data


def directory_bounds(stream, size: int) -> tuple[int, int, int]:
    """Bound central-directory allocation before ZipFile constructs its index."""
    stream.seek(max(0, size - 65_557))
    tail = stream.read(65_557)
    offset = tail.rfind(b"PK\x05\x06")
    require(offset >= 0 and len(tail) - offset >= 22, "ZIP end record is missing")
    end_offset = size - len(tail) + offset
    _, disk, central_disk, disk_count, count, length, start, comment_length = struct.unpack(
        "<4s4H2IH", tail[offset:offset + 22])
    require(offset + 22 + comment_length == len(tail), "ZIP has trailing or malformed end data")
    require(disk == central_disk == 0 and disk_count == count, "multi-disk ZIP is unsupported")
    boundary = end_offset
    stream.seek(max(0, end_offset - 20))
    locator = stream.read(20)
    if locator[:4] == b"PK\x06\x07":
        _, zip64_disk, zip64_offset, disks = struct.unpack("<4sIQI", locator)
        require(zip64_disk == 0 and disks == 1 and zip64_offset + 56 == end_offset - 20,
                "invalid ZIP64 locator")
        stream.seek(zip64_offset)
        record = struct.unpack("<4sQ2H2I4Q", read_exact(stream, 56))
        signature, record_size, _, _, disk64, central_disk64, disk_count64, count64, length64, start64 = record
        require(signature == b"PK\x06\x06" and record_size == 44,
                "unsupported ZIP64 end record")
        require(disk64 == central_disk64 == 0 and disk_count64 == count64,
                "multi-disk ZIP64 is unsupported")
        require(count in (count64, 0xffff) and length in (length64, 0xffffffff)
                and start in (start64, 0xffffffff), "ZIP and ZIP64 end records disagree")
        count, length, start, boundary = count64, length64, start64, zip64_offset
    else:
        require(count != 0xffff and length != 0xffffffff and start != 0xffffffff,
                "ZIP64 end record is missing")
    require(0 < count <= MAX_MEMBERS, "ZIP member count exceeds limit or is empty")
    require(0 < length <= MAX_DIRECTORY_BYTES, "ZIP central directory exceeds size limit")
    require(start + length == boundary, "ZIP central directory offsets are inconsistent")
    # A forged EOCD count must not make ZipFile allocate more than MAX_MEMBERS.
    # Walk fixed-size central headers first, without retaining member objects.
    stream.seek(start)
    records = 0
    while stream.tell() < start + length:
        require(start + length - stream.tell() >= 46, "truncated ZIP central header")
        header = read_exact(stream, 46)
        require(header[:4] == b"PK\x01\x02", "invalid ZIP central record")
        name_size, extra_size, comment_size = struct.unpack("<3H", header[28:34])
        following = stream.tell() + name_size + extra_size + comment_size
        require(following <= start + length, "ZIP central record exceeds directory")
        records += 1
        require(records <= MAX_MEMBERS and records <= count, "ZIP member count mismatch")
        stream.seek(following)
    require(records == count, "ZIP central directory/member count mismatch")
    return start, length, count


def extra_fields(raw: bytes) -> dict[int, bytes]:
    fields = {}
    while raw:
        require(len(raw) >= 4, "truncated ZIP extra field")
        kind, size = struct.unpack("<HH", raw[:4])
        require(size <= len(raw) - 4 and kind not in fields, "invalid/duplicate ZIP extra field")
        # Never interpret link targets, alternate names, or platform attributes.
        require(kind in (0x0001, 0x5455, 0x000a, 0x7875), "unsupported ZIP extra field")
        fields[kind] = raw[4:4 + size]
        raw = raw[4 + size:]
    return fields


def safe_member(info: zipfile.ZipInfo) -> tuple[str, bool]:
    name = info.orig_filename  # filename alone loses the original NUL suffix.
    require(isinstance(name, str) and name == info.filename and "\x00" not in name,
            "ZIP member has a NUL or rewritten name")
    require(name and "\\" not in name and ":" not in name
            and not any(ord(c) < 32 or ord(c) == 127 for c in name), "unsafe ZIP member name")
    directory = name.endswith("/")
    relative = name[:-1] if directory else name
    path = PurePosixPath(relative)
    require(relative and not path.is_absolute() and all(p not in (".", "..") for p in path.parts)
            and str(path) == relative and unicodedata.normalize("NFC", relative) == relative,
            "ZIP path must be canonical and relative")
    require(len(path.parts) <= 64 and len(relative.encode("utf-8")) <= 1024
            and all(len(part.encode("utf-8")) <= 255 for part in path.parts),
            "ZIP member path exceeds length/depth limit")
    mode = stat.S_IFMT(info.external_attr >> 16)
    require(mode in ((0, stat.S_IFDIR) if directory else (0, stat.S_IFREG)),
            "ZIP links and nonregular members are forbidden")
    require(not (info.external_attr & (0x08 | 0x40 | 0x400)), "ZIP special DOS member is forbidden")
    require(directory or not (info.external_attr & 0x10), "ZIP directory name/type mismatch")
    require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
            "unsupported ZIP compression method")
    require(not (info.flag_bits & ~0x080e), "encrypted or unsupported ZIP flags")
    require(info.compress_type == zipfile.ZIP_DEFLATED or not (info.flag_bits & 0x06),
            "invalid stored ZIP flags")
    require(info.volume == 0, "multi-disk ZIP member is unsupported")
    require(info.file_size >= 0 and info.compress_size >= 0, "invalid ZIP member size")
    require(not directory or info.file_size == 0, "ZIP directory has file content")
    extra_fields(info.extra)
    return relative, directory


def member_data_offset(stream, info: zipfile.ZipInfo, boundary: int) -> int:
    stream.seek(info.header_offset)
    fields = struct.unpack("<4s5H3I2H", read_exact(stream, 30))
    signature, _, flags, method, _, _, crc, compressed, logical, name_length, extra_length = fields
    require(signature == b"PK\x03\x04", "invalid ZIP local header")
    require(flags == info.flag_bits and method == info.compress_type, "ZIP local/central flags differ")
    name = read_exact(stream, name_length).decode("utf-8" if flags & 0x800 else "cp437")
    require(name == info.orig_filename and "\x00" not in name, "ZIP local/central names differ")
    extras = extra_fields(read_exact(stream, extra_length))
    zip64 = extras.get(1, b"")
    for field in ("logical", "compressed"):
        value = logical if field == "logical" else compressed
        if value == 0xffffffff:
            require(len(zip64) >= 8, "missing local ZIP64 size")
            value, = struct.unpack("<Q", zip64[:8])
            zip64 = zip64[8:]
            if field == "logical":
                logical = value
            else:
                compressed = value
    expected = (info.CRC, info.compress_size, info.file_size)
    actual = (crc, compressed, logical)
    if flags & 8:
        require(all(a in (0, b) for a, b in zip(actual, expected)), "ZIP local sizes differ")
    else:
        require(actual == expected, "ZIP local sizes/CRC differ")
    data_offset = stream.tell()
    end = data_offset + info.compress_size
    require(end <= boundary, "ZIP member overlaps another record")
    gap = boundary - end
    if flags & 8:
        require(gap in (12, 16, 20, 24), "invalid ZIP data descriptor size")
        stream.seek(end)
        descriptor = read_exact(stream, gap)
        if gap in (16, 24):
            require(descriptor[:4] == b"PK\x07\x08", "invalid ZIP data descriptor signature")
            descriptor = descriptor[4:]
        actual = struct.unpack("<III" if len(descriptor) == 12 else "<IQQ", descriptor)
        require(actual == expected, "ZIP data descriptor differs from central directory")
    else:
        require(gap == 0, "unexpected bytes between ZIP records")
    return data_offset


def copy_member(stream, info, offset, output) -> None:
    """Decode raw compressed bytes; ZipExtFile silently truncates at file_size."""
    stream.seek(offset)
    remaining = info.compress_size
    count = crc = 0
    decoder = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None

    def consume(data):
        nonlocal count, crc
        count += len(data)
        require(count <= info.file_size, "ZIP actual expansion exceeds declared size")
        crc = zlib.crc32(data, crc)
        if output is not None:
            output.write(data)

    while remaining:
        data = read_exact(stream, min(CHUNK, remaining))
        remaining -= len(data)
        if decoder is None:
            consume(data)
        else:
            while data:
                consume(decoder.decompress(data, min(CHUNK, info.file_size - count + 1)))
                require(not decoder.unused_data, "ZIP deflate stream has trailing data")
                data = decoder.unconsumed_tail
    if decoder is not None:
        while not decoder.eof:
            data = decoder.decompress(b"", min(CHUNK, info.file_size - count + 1))
            if not data:
                break
            consume(data)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                "ZIP deflate stream is truncated or has trailing data")
    require(count == info.file_size and crc == info.CRC, "ZIP actual size/CRC mismatch")


def extract_artifact(archive: Path, output: Path, artifact: dict) -> None:
    created = False
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        with os.fdopen(os.open(archive, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), "artifact carrier must be a regular file")
            require(before.st_size == artifact["size_in_bytes"], "artifact carrier byte-size mismatch")
            digest = hashlib.sha256()
            hashed = 0
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                hashed += len(chunk)
                require(hashed <= artifact["size_in_bytes"], "artifact carrier grew during verification")
                digest.update(chunk)
            require(hashed == artifact["size_in_bytes"], "artifact carrier changed byte size")
            require("sha256:" + digest.hexdigest() == artifact["digest"], "artifact carrier SHA256 mismatch")
            start, length, count = directory_bounds(stream, before.st_size)
            with zipfile.ZipFile(stream) as archive_index:
                infos = archive_index.infolist()
                require(archive_index.start_dir == start and len(infos) == count,
                        "ZIP central directory/member count mismatch")
                require(sum(46 + len(i.orig_filename.encode('utf-8' if i.flag_bits & 0x800 else 'cp437'))
                            + len(i.extra) + len(i.comment) for i in infos) == length,
                        "ZIP central directory contains unexpected records")
                members = []
                names = {}
                total = 0
                for info in infos:
                    relative, directory = safe_member(info)
                    require(relative not in names, "duplicate ZIP member path")
                    names[relative] = directory
                    total += info.file_size
                    require(total <= MAX_UNCOMPRESSED_BYTES, "ZIP logical size exceeds limit")
                    members.append((info, relative, directory))
                for relative in names:
                    require(all(names.get(str(p), True) for p in PurePosixPath(relative).parents),
                            "ZIP file is an ancestor of another member")
                members.sort(key=lambda row: row[0].header_offset)
                require(members[0][0].header_offset == 0, "ZIP has a prefix or invalid first member offset")
                offsets = []
                for index, (info, _, _) in enumerate(members):
                    boundary = members[index + 1][0].header_offset if index + 1 < len(members) else start
                    require(info.header_offset < boundary, "ZIP members have duplicate/overlapping offsets")
                    offsets.append(member_data_offset(stream, info, boundary))
                output.mkdir(mode=0o700, parents=False, exist_ok=False)
                created = True
                for (info, relative, directory), offset in zip(members, offsets):
                    destination = output / relative
                    if directory:
                        destination.mkdir(parents=True, exist_ok=True)
                        copy_member(stream, info, offset, None)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as target:
                            copy_member(stream, info, offset, target)
            after = os.fstat(stream.fileno())
            require(all(getattr(before, field) == getattr(after, field)
                        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
                    "artifact carrier changed during verification")
    except BaseException:
        if created:
            shutil.rmtree(output)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("select", "extract"):
        subparser = phases.add_parser(phase)
        subparser.add_argument("--repository", required=True)
        subparser.add_argument("--repository-id", required=True, type=cli_id)
        subparser.add_argument("--training-run-id", required=True, type=cli_id)
        subparser.add_argument("--expected-head-sha", required=True)
        subparser.add_argument("--run-json", required=True, type=Path)
        subparser.add_argument("--workflow-json", required=True, type=Path)
        subparser.add_argument("--artifacts-json", required=True, type=Path)
        if phase == "extract":
            subparser.add_argument("--zip", required=True, type=Path)
            subparser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        artifact = select_artifact(args)
        if args.phase == "extract":
            extract_artifact(args.zip, args.output, artifact)
        else:
            print(f"artifact_id={artifact['id']}")
            print(f"artifact_digest={artifact['digest']}")
            print(f"artifact_size={artifact['size_in_bytes']}")
    except SourceRejected as exc:
        print(f"model promotion source rejected: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError, EOFError, NotImplementedError, RecursionError, struct.error,
            zipfile.BadZipFile, zlib.error) as exc:
        # Library errors may embed untrusted member names, paths, or terminal controls.
        print(f"model promotion source rejected: invalid/unreadable input ({type(exc).__name__})",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
