#!/usr/bin/env python3
"""Read a small, closed saved-tensor format without unpickling or importing Torch.

GLOBAL/REDUCE are interpreted as three inert format descriptors; no callable,
module import, pickle virtual machine, audio decoder or acoustic model executes.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import math
import pickletools
import struct
import zipfile


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    count: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


def require(value, message):
    if not value:
        raise ValueError(message)


def metadata(raw):
    require(len(raw) <= 65536, "bounded tensor metadata required")
    mark, stack, memo = object(), [], {}
    allowed = {"torch._utils _rebuild_tensor_v2", "torch FloatStorage", "collections OrderedDict"}
    for count, (op, arg, position) in enumerate(pickletools.genops(raw)):
        require(count < 4096 and len(stack) < 512 and len(memo) < 512, "metadata bounds")
        name = op.name
        if name == "PROTO":
            require(position == 0 and arg == 2, "protocol 2 required")
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "MARK":
            stack.append(mark)
        elif name in ("BINUNICODE", "BININT", "BININT1", "BININT2", "LONG1"):
            require(type(arg) in (str, int), "primitive only")
            stack.append(arg)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name == "GLOBAL":
            require(arg in allowed, "unrecognized inert global descriptor")
            stack.append(Symbol(arg))
        elif name in ("BINPUT", "LONG_BINPUT"):
            require(type(arg) is int and 0 <= arg < 512 and arg not in memo and stack,
                    "bounded nonoverwritten memo")
            memo[arg] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET"):
            require(arg in memo, "missing memo")
            stack.append(memo[arg])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1])
            require(len(stack) >= n, "tuple underflow")
            values = tuple(stack[-n:]); del stack[-n:]; stack.append(values)
        elif name == "TUPLE":
            require(any(x is mark for x in stack), "tuple mark")
            start = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is mark)
            values = tuple(stack[start + 1:]); del stack[start:]; stack.append(values)
        elif name == "BINPERSID":
            value = stack.pop()
            require(type(value) is tuple and len(value) == 5 and value[0] == "storage" and
                    value[1] == Symbol("torch FloatStorage") and value[2] in ("0", "1") and
                    value[3] == "cpu" and type(value[4]) is int and 0 < value[4] <= 4000000,
                    "CPU float32 storage descriptor required")
            stack.append(Storage(value[2], value[4]))
        elif name == "REDUCE":
            require(len(stack) >= 2, "descriptor underflow")
            args, symbol = stack.pop(), stack.pop()
            if symbol == Symbol("collections OrderedDict"):
                require(args == (), "empty inert hooks descriptor")
                stack.append({})
            else:
                require(symbol == Symbol("torch._utils _rebuild_tensor_v2") and
                        type(args) is tuple and len(args) == 6 and isinstance(args[0], Storage) and
                        args[1] == 0 and type(args[2]) is tuple and type(args[3]) is tuple and
                        all(type(x) is int and x > 0 for x in args[2] + args[3]) and
                        args[4] is False and args[5] == {}, "closed tensor descriptor required")
                stack.append(Tensor(args[0], args[1], args[2], args[3]))
        elif name == "SETITEMS":
            require(any(x is mark for x in stack), "dictionary mark")
            start = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is mark)
            require(start > 0 and type(stack[start - 1]) is dict, "dictionary target")
            values = stack[start + 1:]
            require(len(values) % 2 == 0, "dictionary pairs")
            target = stack[start - 1]
            for key, value in zip(values[::2], values[1::2]):
                require(type(key) is str and key not in target, "unique string key")
                target[key] = value
            del stack[start:]
        elif name == "STOP":
            require(len(stack) == 1 and type(stack[0]) is dict and position == len(raw) - 1,
                    "single final dictionary without trailing bytes")
            return stack[0]
        else:
            raise ValueError("unsupported metadata opcode: " + name)
    raise ValueError("missing metadata STOP")


def tensor_hash(shape, raw):
    header = json.dumps({"dtype": "torch.float32", "shape": list(shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + raw).hexdigest()


def read_argmax(raw):
    require(len(raw) <= 20 * 1024 * 1024, "bounded saved numeric object required")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        expected = {"archive/data.pkl", "archive/.format_version", "archive/.storage_alignment",
                    "archive/byteorder", "archive/data/0", "archive/data/1", "archive/version",
                    "archive/.data/serialization_id"}
        require(len(names) == len(expected) and set(names) == expected, "closed numeric ZIP member set")
        require(all(i.file_size <= 16 * 1024 * 1024 and not i.flag_bits & 1
                    for i in archive.infolist()), "numeric member bounds")
        require(archive.read("archive/byteorder") == b"little", "little endian source required")
        record = metadata(archive.read("archive/data.pkl"))
        require(set(record) == {"schema", "source_id", "feature", "logits", "feature_sha256", "logits_sha256"},
                "closed saved record")
        require(record["schema"] == "kws-d1-development-numeric-evidence-v2", "numeric schema")
        feature, logits = record["feature"], record["logits"]
        require(isinstance(feature, Tensor) and isinstance(logits, Tensor), "inert tensors required")
        require(len(logits.shape) == 3 and logits.shape[1:] == (1, 2601) and
                logits.stride == (2601, logits.shape[0] * 2601, 1), "frozen contiguous logit geometry")
        frames = logits.shape[0]
        require(0 < frames <= 1500 and feature.shape == (frames, 400) and feature.stride == (400, 1),
                "frozen feature geometry")
        require(feature.storage.key == "0" and logits.storage.key == "1" and
                feature.storage.count == frames * 400 and logits.storage.count == frames * 2601,
                "exact nonaliased storage geometry")
        feature_raw, logit_raw = archive.read("archive/data/0"), archive.read("archive/data/1")
        require(len(feature_raw) == frames * 400 * 4 and len(logit_raw) == frames * 2601 * 4,
                "storage payload lengths")
        require(tensor_hash(feature.shape, feature_raw) == record["feature_sha256"] and
                tensor_hash(logits.shape, logit_raw) == record["logits_sha256"], "saved tensor hashes")
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", feature_raw)), "nonfinite feature")
        tokens = []
        unpacker = struct.Struct("<2601f")
        for row in range(frames):
            values = unpacker.unpack_from(logit_raw, row * 2601 * 4)
            require(all(math.isfinite(v) for v in values), "nonfinite logits")
            tokens.append(max(range(2601), key=values.__getitem__))
        return {"source_id": record["source_id"], "feature_sha256": record["feature_sha256"],
                "logits_sha256": record["logits_sha256"], "tokens": tokens}
