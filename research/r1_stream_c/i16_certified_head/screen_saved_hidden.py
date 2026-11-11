#!/usr/bin/env python3
"""Owner-only fixed INT16 full-OTHER head screen; no audio or network access.

Reads exact retained zip bytes without extracting PCM or changing any input.
Requires NumPy 2.3.5 / 1 BLAS thread for the registered diagnostic.
Only sanitized aggregate counts are printed and optionally atomically written.
This is NOT the original C par4 numeric gate or CPU benchmark.
"""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
import zipfile

ORIGIN_SHA = "23c9acf557d0a0c0c63d88c0b995d0d6c686b7a6297659c8a9570f9b101e7661"
ACTIVE_SHA = "5540f63c6ef87eb403568f008a0dc9b76985eb5185e549ea5288f5b67d3a6e05"
MODEL_SHA = "63f18045c64f95e0a57776a2860a2714fdf3fd8f21472d59fe865608030f3ed4"
PRECISE = (0, 1027, 1401, 1462, 2093)


def hashed_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def screen(reference, active, split):
    import numpy as np
    if np.__version__ != "2.3.5":
        raise ValueError("fixed NumPy 2.3.5 required")
    if hashed_file(reference) != ORIGIN_SHA or hashed_file(active) != ACTIVE_SHA:
        raise ValueError("owner source archive identity mismatch")
    with zipfile.ZipFile(reference) as zp, zipfile.ZipFile(active) as za:
        model = zp.read("weights/model-int8.r1hq8")
        if len(model) != 1960140 or hashlib.sha256(model).hexdigest() != MODEL_SHA:
            raise ValueError("original complete quantized head hash mismatch")
        p = json.loads(za.read("competitor-active-v1/execution/PREDICTIONS.json"))
        selection = json.loads(za.read("competitor-active-v1/SELECTION.json"))
        ids = selection["selected_other_ids"]
        if len(ids) != 113 or ids != sorted(set(ids)):
            raise ValueError("selection mismatch")
        other_ids = np.setdiff1d(np.arange(2601), PRECISE)
        qoff = 88 + 390474 * 4
        soff = qoff + 2601 * 140 + 4
        boff = soff + 2601 * 8
        q = np.frombuffer(model, dtype="i1", count=2601 * 140, offset=qoff)
        q = q.reshape(2601, 140)[other_ids]
        scale = np.frombuffer(model, dtype="<f8", count=2601, offset=soff)[other_ids]
        bias = np.frombuffer(model, dtype="<f4", count=2601, offset=boff)[other_ids].astype("f8")
        weights = q.T.astype("f8").copy(order="C")
        l1 = np.abs(q.astype("f8")).sum(axis=1)
        result = {"recordings": 0, "frames": 0, "one_candidate": 0,
                  "two_candidate": 0, "three_or_more": 0,
                  "missing_true_winner": 0, "interval_violations": 0,
                  "max_absolute_approx_error": 0.0, "max_candidate_count": 0,
                  "full_max_mismatches": 0, "bitwise_par4_score_matches": 0,
                  "bitwise_original_id_matches": 0, "exact_candidates_evaluated": 0}
        for row in p["records"]:
            if row["split"] != split:
                continue
            i = row["index"]
            name = "{:03d}".format(i)
            h = zp.read("competitor-model/extraction/hidden/" + name + ".r1h")
            a = za.read("competitor-active-v1/execution/predictions/" + name + ".r1a")
            if (hashlib.sha256(h).hexdigest() != row["hidden_sha256"] or
                    hashlib.sha256(a).hexdigest() != row["sha256"] or
                    h[:8] != b"R1HD140\x00" or a[:8] != b"R1ACTV1\x00"):
                raise ValueError("retained record hash or format mismatch")
            n = row["frames"]
            if len(h) != 64 + 1200 * n or len(a) != 64 + 40 * n:
                raise ValueError("frame byte counts changed")
            x = np.ndarray((n, 140), dtype="<f8", buffer=h, offset=96,
                           strides=(1200, 8))
            original_max = np.ndarray((n,), dtype="<f8", buffer=h,
                                      offset=1256, strides=(1200,))
            original_id = np.ndarray((n,), dtype="<u4", buffer=h,
                                     offset=92, strides=(1200,))
            if not np.isfinite(x).all():
                raise ValueError("nonfinite hidden")
            full = (x @ weights) * scale[None, :] + bias[None, :]
            maxs = full.max(axis=1)
            if np.max(np.abs(maxs - original_max)) > 1e-8:
                raise ValueError("saved full original OTHER max mismatch")
            peak = np.max(np.abs(x), axis=1)
            step = np.where(peak > 0, peak / 32767.0, 1.0)
            quant = np.rint(x / step[:, None]).clip(-32767, 32767)
            dot = quant @ weights  # integer-valued results exactly fit binary64
            approximate = (dot * step[:, None]) * scale[None, :] + bias[None, :]
            error = (step[:, None] / 2) * scale[None, :] * l1[None, :]
            error += 1e-6 + 1e-8 * np.abs(approximate)
            violations = np.abs(full - approximate) > error
            if violations.any():
                raise ValueError("approximate interval misses a full original row")
            best_lower = (approximate - error).max(axis=1)
            eligible = approximate + error >= best_lower[:, None]
            count = eligible.sum(axis=1)
            winners = np.argmax(full, axis=1)
            missed = ~eligible[np.arange(n), winners]
            if missed.any():
                raise ValueError("original OTHER maximum not in candidate set")
            original_idx = other_ids[winners]
            if not np.array_equal(original_idx, original_id):
                raise ValueError("original OTHER ID changed")
            # Recalculate only candidate rows in the original 4-lane
            # serial Python-double reduction order, not NumPy's BLAS.
            for fi in range(n):
                best = -float("inf")
                best_id = 4294967295
                xframe = x[fi]
                positions = np.flatnonzero(eligible[fi])
                for pos in positions:
                    weight = q[pos]
                    s0 = s1 = s2 = s3 = 0.0
                    for j in range(0, 140, 4):
                        s0 += float(xframe[j]) * int(weight[j])
                        s1 += float(xframe[j + 1]) * int(weight[j + 1])
                        s2 += float(xframe[j + 2]) * int(weight[j + 2])
                        s3 += float(xframe[j + 3]) * int(weight[j + 3])
                    exact = ((s0 + s1) + (s2 + s3)) * float(scale[pos]) + float(bias[pos])
                    original = int(other_ids[pos])
                    if exact > best or (exact == best and original < best_id):
                        best, best_id = exact, original
                if (struct.pack("<d", best) !=
                        struct.pack("<d", float(original_max[fi]))):
                    raise ValueError("candidate exact par4 result not byte-identical to retained teacher")
                if best_id != int(original_id[fi]):
                    raise ValueError("candidate original class ID differs from teacher")
                result["bitwise_par4_score_matches"] += 1
                result["bitwise_original_id_matches"] += 1
                result["exact_candidates_evaluated"] += len(positions)
            result["recordings"] += 1
            result["frames"] += n
            result["one_candidate"] += int(np.count_nonzero(count == 1))
            result["two_candidate"] += int(np.count_nonzero(count == 2))
            result["three_or_more"] += int(np.count_nonzero(count >= 3))
            result["max_candidate_count"] = max(result["max_candidate_count"], int(count.max()))
            result["max_absolute_approx_error"] = max(
                result["max_absolute_approx_error"],
                float(np.max(np.abs(full - approximate))))
        target = (96, 9079) if split == "train" else (34, 3176)
        if (result["recordings"], result["frames"]) != target:
            raise ValueError("unexpected split identity or exposure")
        return {"schema": "r1-i16-saved-head-screen-v1",
                "split": split, "model_sha256": MODEL_SHA,
                "reference_zip_sha256": ORIGIN_SHA,
                "active_zip_sha256": ACTIVE_SHA,
                "result": result, "owner_pcm_reads": 0,
                "backbone_forward_calls": 0, "source_only_not_C_bitwise": True,
                "independent_far_frr": False, "physical_ssc305": False,
                "model_promotion_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-archive", required=True, type=Path)
    p.add_argument("--active-archive", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("train", "development"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = screen(a.reference_archive, a.active_archive, a.split)
    data = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if a.output:
        with a.output.open("x", encoding="utf-8") as f:
            f.write(data)
    print(data, end="")


if __name__ == "__main__":
    main()
