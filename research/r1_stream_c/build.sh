#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p bin
cc="${CC:-gcc}"
"$cc" -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -fexcess-precision=standard -fPIC -shared -Isource \
 source/r1_stream.c source/r1_pcm.c source/canonical_fbank.c source/r1fsmn.c source/sha256.c -lm -o bin/libr1_stream.so
"$cc" -std=c11 -O2 -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -fexcess-precision=standard -Isource \
 source/r1_pcm_cli.c source/r1_stream.c source/r1_pcm.c source/canonical_fbank.c source/r1fsmn.c source/sha256.c -lm -o bin/r1_pcm
