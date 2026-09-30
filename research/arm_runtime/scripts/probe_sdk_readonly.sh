#!/bin/sh
# Usage: sh probe_sdk_readonly.sh /absolute/trusted/SDK/bin/target-gcc [target-ELF]
# No compilation, downloading, installation, model loading, or target execution.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo 'Usage: probe_sdk_readonly.sh /absolute/trusted/SDK/bin/target-gcc [target-ELF]' >&2
  exit 2
fi
cc=$1
case "$cc" in /*) ;; *) echo 'Compiler must be an explicit absolute path' >&2; exit 2;; esac
test -x "$cc"
printf '%s\n' '=== Actual SDK compiler ==='
"$cc" --version
"$cc" -dumpmachine
"$cc" -print-sysroot
printf '%s\n' '=== Compiler-resolved runtime paths ==='
"$cc" -print-file-name=libc.so
"$cc" -print-file-name=libstdc++.so.6
"$cc" -print-file-name=libgcc_s.so.1
if [ "$#" -eq 2 ]; then
  target=$2
  test -f "$target"
  reader=$("$cc" -print-prog-name=readelf)
  printf '%s\n' '=== Existing target ELF; read only ==='
  "$reader" -h -A -l -d -V "$target"
fi
printf '%s\n' 'Record SDK/BSP release name and sysroot archive/source separately. No SSC305 acceptance claimed.'
