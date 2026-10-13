#!/bin/sh
# Fetch a small GGUF model for the WebGPU bench.
# Usage: ./fetch-model.sh [file]   (default: qwen2.5-0.5b-q4_k_m.gguf)
set -eu

cd "$(dirname "$0")"

FILE="${1:-qwen2.5-0.5b-q4_k_m.gguf}"
REPO="ggml-org/Qwen2.5-0.5B-Instruct-GGUF"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

echo "downloading ${URL}"
curl -fL --progress-bar -o "$FILE" "$URL"

# (re)generate the list the page reads
printf '[\n' > list.json
ls -1 *.gguf 2>/dev/null | while read -r f; do
  printf '  {"file": "%s"},\n' "$f"
done | sed '$ s/},$/}/' >> list.json
printf ']\n' >> list.json

echo "done: $(ls -lh "$FILE" | awk '{print $5}')"
