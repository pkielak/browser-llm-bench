# llm-bench

Llama-bench-style performance benchmarks for small LLMs on **WebGPU**.
One static page: WebGPU compatibility check + raw GPU throughput probe,
then a model benchmark (llama.cpp-web: wasm + WebGPU). Results land in a
small dashboard.

```
web/ (static page, llama.js)  ──►  results/*.json  ──►  dashboard (:3003)
```

## Layout

| Path | What |
|---|---|
| `web/` | benchmark page (pure static, open it in any WebGPU browser) |
| `web/models/` | GGUF models (gitignored; `fetch-model.sh`) |
| `server/` | tiny node server for the dashboard (docker, :3003) |
| `dashboard/` | results table + paste-JSON submit box |
| `results/` | benchmark JSON files |

## Metrics (llama-bench convention)

- **pp** — prompt processing: tok/s over a fixed 256-token prompt
- **tg** — token generation: tok/s over 128 generated tokens
- **ttft** — ms from submit to first generated token
- 3 iterations; best + median reported
- Plus a **WebGPU compat/probe** block: adapter info, feature/limit
  snapshot, and raw compute throughput (FMA/s) from a fixed WGSL kernel —
  so devices can be compared in the dashboard even without a model run.

Result file shape (`results/<date>-<device>.json`):

```json
{
  "backend": "webgpu",
  "compat": {
    "webgpu": true,
    "adapter": "NVIDIA GeForce …",
    "features": { "shader-f16": true },
    "probe": { "fma_per_s_best": 1.2e12, "fma_per_s_median": 1.1e12 }
  },
  "engine": "llama.cpp-web (llama.js 0.8.9)",
  "device": "NVIDIA GeForce …",
  "model": "qwen2.5-0.5b-q4_k_m.gguf",
  "params": { "pp": 256, "tg": 128, "iters": 3 },
  "pp":  { "best": 123.4, "median": 120.1 },
  "tg":  { "best": 24.5,  "median": 24.2 },
  "ttft_ms": 310,
  "date": "2026-07-09T12:00:00Z"
}
```

## Running the benchmark

```bash
# 1. fetch a small GGUF (gitignored)
( cd web/models && ./fetch-model.sh )

# 2. serve web/ (any static server)
python3 -m http.server 8080 --directory web
# → http://localhost:8080
#   1 · Check WebGPU   → adapter info + compute probe
#   2 · Run model bench → pp / tg / ttft
#   copy the result JSON
```

Notes:
- WebGPU requires a secure context — `localhost` works, plain HTTP on a LAN
  IP does not (use HTTPS or localhost).
- Headless/CI browsers can't produce meaningful numbers (no real GPU);
  this is a device page, run it on the actual hardware you're benchmarking.

## Dashboard

```bash
# from the infra repo (service is in docker-compose.yml)
docker compose up -d llm-bench
# → http://localhost:3003 — table of results/ + submit box
```

## Adding a result

Copy the JSON from the bench page, paste it into the dashboard submit box
with a filename like `2026-07-09-rtx4090.json`, save. The server writes it
to `results/`; `git add results/ && git commit && git push` to share.
