#!/usr/bin/env bash
# Install the active core role; historical TTS/ASR recipes retain their locks.
set -euo pipefail

venv_path="${1:-.venv}"
python_bin="${KWS_RESEARCH_PYTHON:-python3}"
"$python_bin" tools/research_runtime.py check
"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" tools/research_runtime.py check
requirements_dir="$(mktemp -d)"
trap 'rm -rf -- "$requirements_dir"' EXIT

for index in pypi pytorch_cpu; do
  "$venv_path/bin/python" tools/research_runtime.py requirements \
    --role core --index "$index" > "$requirements_dir/$index.txt"
  index_url="$(jq -er --arg index "$index" '.indexes[$index]' configs/runtime/research-runtime.json)"
  "$venv_path/bin/python" -m pip install --disable-pip-version-check --no-input \
    --index-url "$index_url" --requirement "$requirements_dir/$index.txt"
done
"$venv_path/bin/python" -m pip check
"$venv_path/bin/python" tools/research_runtime.py check --role core --packages
