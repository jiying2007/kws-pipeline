#!/bin/sh
# Run on the actual target, if available. Reads only; stdout is the report.
# Does not load this runtime, run audio/model code, install, or change settings.
# Do not publish returned device metadata without reviewing it.
set -u
printf '%s\n' '=== Kernel and CPU ABI evidence ==='
uname -a
if [ -r /proc/cpuinfo ]; then
  grep -E '^(Processor|model name|CPU architecture|CPU implementer|CPU variant|CPU part|CPU revision|Features)[[:space:]]*:' /proc/cpuinfo || :
fi
printf '%s\n' '=== libc declaration (if getconf exists) ==='
if command -v getconf >/dev/null 2>&1; then getconf GNU_LIBC_VERSION 2>&1 || :; fi
printf '%s\n' '=== Dynamic-loader and libc filenames ==='
for item in /lib/ld* /lib/libc* /lib/libuClibc* /lib/libstdc++* /usr/lib/libstdc++* /lib/arm-linux-gnueabihf/ld* /lib/arm-linux-gnueabihf/libc* /usr/lib/arm-linux-gnueabihf/libstdc++*; do
  if [ -e "$item" ] || [ -L "$item" ]; then ls -ld "$item"; fi
done
printf '%s\n' '=== Existing target executable ELF (if readelf exists) ==='
if command -v readelf >/dev/null 2>&1; then
  for item in /bin/busybox /bin/sh; do
    if [ -f "$item" ]; then readelf -h -A -l "$item" 2>&1 || :; break; fi
  done
else
  printf '%s\n' 'readelf unavailable: inspect a copied target /bin/busybox using the SDK readelf on the host'
fi
printf '%s\n' 'This output is inventory only. It does not qualify runtime compatibility or performance.'
