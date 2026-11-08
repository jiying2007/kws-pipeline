#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build source-only R1 Binary64 capture executables for x86-64 and ARMv7-A.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash research/r1_stream_c/build_cross_arch.sh /absolute/new/output" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT="$1"
if [[ "$OUT" != /* || -e "$OUT" || -L "$OUT" ]]; then
  echo "output must be an absolute, fresh, non-symlink path" >&2
  exit 2
fi

PYTHON="\${PYTHON:-python3}"
HOST_CC="\${HOST_CC:-gcc}"
ARM_CC="\${ARM_CC:-arm-linux-gnueabihf-gcc}"
READELF="\${READELF:-readelf}"
QEMU_ARM="\${QEMU_ARM:-qemu-arm-static}"
for tool in "$PYTHON" "$HOST_CC" "$ARM_CC" "$READELF" "$QEMU_ARM"; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "required build tool is unavailable: $tool" >&2
    exit 2
  }
done

TMP="$(mktemp -d "\${TMPDIR:-/tmp}/r1-reference64-build.XXXXXXXX")"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT/host" "$OUT/arm" "$OUT/executor"

"$PYTHON" -I -S -B "$ROOT/research/r1_stream_c/numerics.py" \
  --repo "$ROOT" --mode reference64 --output "$TMP/reference64"
SRC="$TMP/reference64/source"
if [[ -n "$(find "$TMP/reference64/weights" -type f -print -quit)" ]]; then
  echo "source-only build unexpectedly contains weights" >&2
  exit 2
fi

cp "$ROOT/research/r1_stream_c/cross_arch_capture.c" "$OUT/cross_arch_capture.c"
SOURCES=(
  "$SRC/r1d_stream.c"
  "$SRC/r1d_pcm.c"
  "$SRC/reference64_fbank.c"
  "$SRC/r1dfsmn.c"
  "$SRC/sha256.c"
)
COMMON=(
  -std=c11 -O2 -Wall -Wextra -Werror -pedantic
  -fno-fast-math -ffp-contract=off -fexcess-precision=standard
  "-I$SRC"
)
ARM_FLAGS=(-mcpu=cortex-a32 -mfpu=neon-vfpv4 -mfloat-abi=hard)
ARM_BIN="$OUT/arm/r1d_capture_armhf_static"
HOST_BIN="$OUT/host/r1d_capture_x86_64"

"$ARM_CC" "\${COMMON[@]}" "\${ARM_FLAGS[@]}" -static \
  "$OUT/cross_arch_capture.c" "\${SOURCES[@]}" -lm -o "$ARM_BIN" \
  >"$OUT/arm/compile.stdout" 2>"$OUT/arm/compile.stderr"
"$HOST_CC" "\${COMMON[@]}" \
  "$OUT/cross_arch_capture.c" "\${SOURCES[@]}" -lm -o "$HOST_BIN" \
  >"$OUT/host/compile.stdout" 2>"$OUT/host/compile.stderr"

"$READELF" -h -A "$ARM_BIN" >"$OUT/arm/ELF.txt"
"$READELF" -h -A "$HOST_BIN" >"$OUT/host/ELF.txt"
cp "$(command -v "$QEMU_ARM")" "$OUT/executor/qemu-arm-static"
chmod 0755 "$OUT/executor/qemu-arm-static"
if [[ -f /usr/share/doc/qemu-user-static/copyright ]]; then
  cp /usr/share/doc/qemu-user-static/copyright "$OUT/executor/COPYRIGHT"
fi
if command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W qemu-user-static >/dev/null 2>&1; then
  dpkg-query -W -f='\${Package}\n\${Version}\n\${Source}\n' qemu-user-static \
    >"$OUT/executor/PACKAGE.txt"
fi
cat >"$OUT/executor/README.txt" <<'EOF'
Unmodified qemu-user-static user-mode emulator, kept separate from the
Apache-licensed R1 reference source. See COPYRIGHT and PACKAGE.txt when
present. The build did not load private model weights or private audio.
EOF

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unknown')"
"$PYTHON" -I -S -B - "$ROOT" "$OUT" "$SOURCE_COMMIT" "$ARM_CC" "$HOST_CC" <<'PY'
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

root, output, source_commit, arm_cc, host_cc = sys.argv[1:]
root = Path(root)
out = Path(output)

def require(ok, message):
    if not ok:
        raise SystemExit(message)

arm = out / "arm/r1d_capture_armhf_static"
host = out / "host/r1d_capture_x86_64"
raw_arm = arm.read_bytes()
raw_host = host.read_bytes()
require(raw_arm[:6] == b"\x7fELF\x01\x01", "ARM ELF class/endian mismatch")
require(struct.unpack_from("<H", raw_arm, 18)[0] == 40, "ARM ELF machine mismatch")
flags = struct.unpack_from("<I", raw_arm, 36)[0]
require(flags & 0xff000000 == 0x05000000, "ARM EABI5 required")
require(flags & 0x400 != 0 and flags & 0x200 == 0, "ARM hard-float ABI required")
require(raw_host[:6] == b"\x7fELF\x02\x01", "host ELF class/endian mismatch")
require(struct.unpack_from("<H", raw_host, 18)[0] == 62, "host x86-64 ELF required")

files = {}
for path in sorted(p for p in out.rglob("*") if p.is_file()):
    raw = path.read_bytes()
    files[path.relative_to(out).as_posix()] = {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
manifest = {
    "schema": "kws-r1-reference64-cross-arch-build-v1",
    "source_commit": source_commit,
    "target": "x86-64 and Cortex-A32/ARMv7-A NEON-VFPv4 hard-float",
    "arm_compiler": subprocess.check_output([arm_cc, "--version"], timeout=5).decode().splitlines()[0],
    "host_compiler": subprocess.check_output([host_cc, "--version"], timeout=5).decode().splitlines()[0],
    "private_audio_loaded": False,
    "model_weights_loaded": False,
    "model_forwards": 0,
    "training_updates": 0,
    "tts_calls": 0,
    "asr_calls": 0,
    "physical_ssc305": False,
    "files": files,
}
(out / "BUILD.json").write_text(
    json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "status": "BUILD_PASS",
    "source_commit": source_commit,
    "arm_bytes": arm.stat().st_size,
    "host_bytes": host.stat().st_size,
    "file_count": len(files),
    "private_audio_loaded": False,
    "model_weights_loaded": False,
}, sort_keys=True))
PY
