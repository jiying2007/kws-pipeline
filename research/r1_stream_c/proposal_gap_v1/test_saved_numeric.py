#!/usr/bin/env python3
"""Fictional tensor-file parser contracts. No Torch import or pickle execution."""
import io
import struct
import unittest
import warnings
import zipfile

from saved_numeric import metadata, read_argmax, tensor_hash


def text(value):
    raw = value.encode()
    return b"X" + struct.pack("<I", len(raw)) + raw


def integer(value):
    return b"J" + struct.pack("<i", value)


def sequence(values):
    return b"(" + b"".join(integer(v) for v in values) + b"t"


def tensor(key, shape, stride, elements):
    storage = b"(" + text("storage") + b"ctorch\nFloatStorage\n" + text(key) + text("cpu") + integer(elements) + b"tQ"
    return (b"ctorch._utils\n_rebuild_tensor_v2\n(" + storage + integer(0) +
            sequence(shape) + sequence(stride) + b"\x89ccollections\nOrderedDict\n)RtR")


def fixture(*, wrong_hash=False, invalid_stride=False, nan=False, duplicate=False, extra=False):
    frames = 2
    features = b"\0" * (frames * 400 * 4)
    values = [0.0] * (frames * 2601)
    values[1027] = 1.0
    values[2601 + 1401] = 1.0
    if nan:
        values[0] = float("nan")
    logits = struct.pack("<" + "f" * len(values), *values)
    fields = [("schema", text("kws-d1-development-numeric-evidence-v2")),
              ("source_id", text("fictional-source")),
              ("feature", tensor("0", (frames, 400), (400, 1), frames * 400)),
              ("logits", tensor("1", (frames, 1, 2601), (1, 5202, 1) if invalid_stride else (2601, 5202, 1), frames * 2601)),
              ("feature_sha256", text(tensor_hash((frames, 400), features))),
              ("logits_sha256", text("0" * 64 if wrong_hash else tensor_hash((frames, 1, 2601), logits)))]
    payload = b"\x80\x02}(" + b"".join(text(k) + v for k, v in fields) + b"u."
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in {"archive/data.pkl": payload, "archive/.format_version": b"1",
                          "archive/.storage_alignment": b"64", "archive/byteorder": b"little",
                          "archive/data/0": features, "archive/data/1": logits,
                          "archive/version": b"3\n", "archive/.data/serialization_id": b"fixture"}.items():
            archive.writestr(name, raw)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                archive.writestr("archive/data/0", features)
        if extra:
            archive.writestr("unexpected", b"no")
    return buffer.getvalue()


class NumericReaderTests(unittest.TestCase):
    def test_fictional_argmax(self):
        result = read_argmax(fixture())
        self.assertEqual(result["source_id"], "fictional-source")
        self.assertEqual(result["tokens"], [1027, 1401])

    def test_corrupted_tensor_hash(self):
        with self.assertRaisesRegex(ValueError, "tensor hashes"):
            read_argmax(fixture(wrong_hash=True))

    def test_stride_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            read_argmax(fixture(invalid_stride=True))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            read_argmax(fixture(nan=True))

    def test_closed_zip_roster(self):
        for args in ({"duplicate": True}, {"extra": True}):
            with self.assertRaisesRegex(ValueError, "member set"):
                read_argmax(fixture(**args))

    def test_no_callable_global(self):
        with self.assertRaisesRegex(ValueError, "inert global"):
            metadata(b"\x80\x02cbuiltins\neval\n.")

    def test_no_unknown_opcode(self):
        with self.assertRaisesRegex(ValueError, "opcode"):
            metadata(b"\x80\x02] .")

    def test_no_trailing_bytes(self):
        with self.assertRaisesRegex(ValueError, "trailing"):
            metadata(b"\x80\x02}.trailing")

    def test_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            metadata(b"\x80\x02}(" + text("x") + integer(1) + text("x") + integer(2) + b"u.")

    def test_metadata_size_bound(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            metadata(b"x" * 65537)


if __name__ == "__main__":
    unittest.main()
