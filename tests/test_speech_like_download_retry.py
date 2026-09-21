#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import pathlib
import sys
import tempfile
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import bootstrap_speech_like_stage_a as bootstrap  # noqa: E402


def _request_url(request: str | object) -> str:
    return str(getattr(request, "full_url", request))


def main() -> int:
    payload = b"verified-provider-asset"
    payload_sha = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory(prefix="speech-like-download-retry-") as tmp:
        root = pathlib.Path(tmp)
        target = root / "asset.bin"
        attempts: list[float] = []
        sleeps: list[float] = []

        original_urlopen = bootstrap.urllib.request.urlopen
        original_sleep = bootstrap.time.sleep
        try:
            def flaky_urlopen(request, timeout=0):
                attempts.append(float(timeout))
                if len(attempts) < 3:
                    raise urllib.error.HTTPError(
                        _request_url(request),
                        504,
                        "Gateway Timeout",
                        None,
                        None,
                    )
                return io.BytesIO(payload)

            bootstrap.urllib.request.urlopen = flaky_urlopen
            bootstrap.time.sleep = lambda seconds: sleeps.append(float(seconds))
            state = bootstrap.download_verified(
                url="https://example.invalid/provider.bin",
                target=target,
                expected_sha256=payload_sha,
                expected_size=len(payload),
                label="provider asset",
                allow_file_urls=False,
                refresh=False,
            )
        finally:
            bootstrap.urllib.request.urlopen = original_urlopen
            bootstrap.time.sleep = original_sleep

        assert state == "downloaded"
        assert target.read_bytes() == payload
        assert attempts == [120.0, 120.0, 120.0]
        assert sleeps == [1.0, 2.0]

        target_404 = root / "missing.bin"
        attempts_404 = 0
        sleeps_404: list[float] = []
        try:
            def missing_urlopen(request, timeout=0):
                nonlocal attempts_404
                attempts_404 += 1
                raise urllib.error.HTTPError(
                    _request_url(request),
                    404,
                    "Not Found",
                    None,
                    None,
                )

            bootstrap.urllib.request.urlopen = missing_urlopen
            bootstrap.time.sleep = lambda seconds: sleeps_404.append(float(seconds))
            try:
                bootstrap.download_verified(
                    url="https://example.invalid/missing.bin",
                    target=target_404,
                    expected_sha256=payload_sha,
                    expected_size=len(payload),
                    label="missing provider asset",
                    allow_file_urls=False,
                    refresh=False,
                )
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
            else:
                raise AssertionError("non-transient HTTP 404 was retried or accepted")
        finally:
            bootstrap.urllib.request.urlopen = original_urlopen
            bootstrap.time.sleep = original_sleep

        assert attempts_404 == 1
        assert sleeps_404 == []
        assert not target_404.exists()
        assert not target_404.with_name(target_404.name + ".part").exists()

        target_bad = root / "bad.bin"
        attempts_bad = 0
        sleeps_bad: list[float] = []
        bad_payload = b"wrong-bytes"
        try:
            def bad_urlopen(request, timeout=0):
                nonlocal attempts_bad
                attempts_bad += 1
                return io.BytesIO(bad_payload)

            bootstrap.urllib.request.urlopen = bad_urlopen
            bootstrap.time.sleep = lambda seconds: sleeps_bad.append(float(seconds))
            try:
                bootstrap.download_verified(
                    url="https://example.invalid/bad.bin",
                    target=target_bad,
                    expected_sha256=payload_sha,
                    expected_size=len(bad_payload),
                    label="bad provider asset",
                    allow_file_urls=False,
                    refresh=False,
                )
            except ValueError as exc:
                assert "sha256 mismatch" in str(exc)
            else:
                raise AssertionError("integrity mismatch was retried or accepted")
        finally:
            bootstrap.urllib.request.urlopen = original_urlopen
            bootstrap.time.sleep = original_sleep

        assert attempts_bad == 1
        assert sleeps_bad == []
        assert not target_bad.exists()
        assert not target_bad.with_name(target_bad.name + ".part").exists()

    print("speech-like verified download retry: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
