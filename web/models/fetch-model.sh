#!/bin/sh
# Fetch a LiteRT-LM web model (.litertlm bundle) for the bench.
#
# Default is Gemma 4 E2B (the smallest web-compatible model; the -gpu
# variant is what the Web API expects on WebGPU).
# Other web-compatible models:
#   https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
#   https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm
#
# Usage: ./fetch-model.sh [file]
set -eu

cd "$(dirname "$0")"

FILE="${1:-gemma-4-E2B-it-gpu.litertlm}"
REPO="litert-community/gemma-4-E2B-it-litert-lm"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

echo "downloading ${URL}"
curl -fL --progress-bar -o "$FILE" "$URL"

# (re)generate the list the page reads
printf '[\n' > list.json
ls -1 *.litertlm 2>/dev/null | while read -r f; do
  printf '  {"file": "%s"},\n' "$f"
done | sed '$ s/},$/}/' >> list.json
printf ']\n' >> list.json

echo "done: $(du -h "$FILE" | cut -f1)"
