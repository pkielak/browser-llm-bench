# llm-bench

Llama-bench-style performance benchmarks for small LLMs on **WebGPU**,
running on **LiteRT** (LiteRT-LM, `@litert-lm/core`, WASM + WebGPU).

A small Svelte 5 + Vite static app. The whole model **download + run
mechanism is borrowed from [Pancratius](../Pancratius)**
(`src/lib/llm/`): the Hugging Face model catalogue, download-once with
progress into the browser's Cache Storage, WebGPU/WASM auto-detection, and
the LiteRT engine running in a Web Worker. On top of that sits a thin
benchmark harness (`src/lib/bench.ts`).

## Metrics (llama-bench convention)

The engine build does not stream per token (the whole generation arrives in
one batch), so the metrics are derived from **marginal run differences** —
two runs per iteration, each from a fresh conversation (`reset()` → clean
KV cache):

- run A: 1 token → tA ≈ prefill + 1 decode step
- run B: 128 tokens → tB ≈ prefill + 128 decode steps
- **tg** = 1000 / ((tB − tA) / 127) — tok/s, prefill cancels out
- **ttft** = tA − one decode step — pure prefill, ms
- **pp** = prompt tokens / prefill — tok/s (prompt tokens from
  `conversation.getTokenCount()` when exposed, else chars/4 estimate)

If the loaded runtime exposes its own counters (`getBenchmarkInfo()`), they
are preferred over the wall-clock derivations. 3 iterations; best + median
reported.

The page shows a copyable result JSON:

```json
{
  "backend": "webgpu",
  "model": "Gemma 4 E2B (web)",
  "params": { "tg": 128, "iters": 3, "prompt_chars": 768 },
  "iterations": [ { "index": 1, "pp": 401.3, "tg": 18.4, "ttftMs": 210, "tokens": 128 } ],
  "pp": { "best": 412.0, "median": 401.3 },
  "tg": { "best": 18.9, "median": 18.4 },
  "ttft_ms": 210,
  "date": "2026-07-09T12:00:00Z"
}
```

## Layout

```
src/
  App.svelte            the whole UI (model picker, status, bench, results)
  lib/
    bench.ts            the benchmark harness (pp / tg / ttft, best+median)
    llm/                borrowed from Pancratius (web target):
      types.ts          the LlmRuntime / LlmStatus / LlmState seam
      models.ts         HF `litert-community` model catalogue (sizes included)
      webgpu.ts         WebGPU detection → backend selection
      network.ts        connectivity check before downloads
      model-cache.ts    download-once + Cache Storage persistence
      runtime.ts        the web LiteRT-LM bridge (worker protocol host side)
      runtime-web-worker.ts  the Engine in a Web Worker (pinned @litert-lm/core)
      manager.ts        the reactive load/generate state machine
```

## Getting started

```sh
npm install
npm run dev        # local dev server
npm run build      # production static build → build/
npm run check      # svelte-check
```

Then: pick a model (it downloads once — E2B is ~2 GB — and is kept in Cache
Storage — it starts automatically), wait for "Model ready", hit **Run
benchmark**.

**Serve the build over HTTP — do not open `build/index.html` directly.**
Module scripts + the Web Worker do not load from `file://`, so the page
hangs on a blank tab:

```sh
npm run build && npm run preview   # serves build/ on http://localhost:4173
```

Notes:
- Downloads are **parallel byte-range slices** (8 concurrent `Range`
  requests — a single connection to the HF CDN is the usual bottleneck),
  with a single-stream fallback if the server does not support ranges.
- The page needs a **secure context** for Cache Storage + WebGPU:
  `localhost` works; plain HTTP on a LAN IP degrades (no cache persistence,
  no WebGPU → WASM).
- The LiteRT engine + WASM are loaded dynamically from a **version-pinned**
  `@litert-lm/core` build (see `LITERT_VERSION` in `runtime-web-worker.ts`)
  — the JS module and its `wasm/` dir must stay the same version.
- This is a device page: run it on the hardware you're measuring.
