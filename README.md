# llm-bench

Llama-bench-style performance benchmarks for small LLMs on **WebGPU**,
running on **LiteRT** (`@litert-lm/core`, LiteRT-LM WASM + WebGPU).
One static page: WebGPU compatibility check + raw GPU throughput probe,
then a model benchmark. Results land in a small dashboard.

```
web/ (static page, LiteRT-LM WASM)  ──►  results/*.json  ──►  dashboard (:3003)
```

## Layout

| Path | What |
|---|---|
| `web/` | benchmark page (pure static, open it in any WebGPU browser) |
| `web/models/` | `.litertlm` models (gitignored; `fetch-model.sh`) |
| `server/` | tiny node server for the dashboard (docker, :3003) |
| `dashboard/` | results table + paste-JSON submit box |
| `results/` | benchmark JSON files |

## Metrics (llama-bench convention)

- **pp** — prefill: tok/s, from the engine's own counter
  (`conversation.getBenchmarkInfo().lastPrefillTokensPerSecond`)
- **tg** — token generation: tok/s over a fixed 128-token generation,
  wall-clock over the streamed chunks
- **ttft** — ms from submit to first streamed token
- 3 iterations (fresh conversation each → clean KV cache + counters);
  best + median reported
- Plus a **WebGPU compat/probe** block: adapter info, feature/limit
  snapshot, and raw compute throughput (FMA/s) from a fixed WGSL kernel —
  so devices can be compared in the dashboard even without a model run.

Result file shape (`results/<date>-<device>.json`):

```json
{
  "backend": "litert",
  "compat": {
    "webgpu": true,
    "adapter": "NVIDIA GeForce …",
    "features": { "shader-f16": true },
    "probe": { "fma_per_s_best": 1.2e12, "fma_per_s_median": 1.1e12 }
  },
  "engine": "LiteRT-LM (@litert-lm/core, WASM + WebGPU)",
  "device": "NVIDIA GeForce …",
  "model": "gemma-4-E2B-it-gpu.litertlm",
  "params": { "tg": 128, "iters": 3, "prompt_chars": 768 },
  "pp":  { "best": 412.0, "median": 401.3 },
  "tg":  { "best": 18.9,  "median": 18.4 },
  "ttft_ms": 210,
  "date": "2026-07-09T12:00:00Z"
}
```

## Web-compatible models

The LiteRT-LM JS API supports (from `litert-community/*` on Hugging Face):

| Model | File |
|---|---|
| Gemma 4 E2B (default, smallest) | `gemma-4-E2B-it-gpu.litertlm` |
| Gemma 4 E4B | `gemma-4-E4B-it-gpu.litertlm` |
| Gemma 4 12B | `gemma-4-12B-it-gpu.litertlm` |
| Gemma 4 26B A4B / 31B | `gemma-4-26B-A4B-it-gpu.litertlm` / `gemma-4-31B-it-gpu.litertlm` |

The `-gpu` variants are what the Web API expects on WebGPU.

## Running the benchmark

```bash
# 1. fetch a model (gitignored; E2B is a few GB)
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
- Engine init downloads/loads the whole model; expect a while for E2B+.
- This is a device page — run it on the actual hardware you're benchmarking
  (no headless/CI numbers; a software rasterizer measures nothing).

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
