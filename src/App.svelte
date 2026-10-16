<script lang="ts">
	import { MODELS, formatBytes, getModel } from './lib/llm/models';
	import { llm, selectModel, setAllowCpuFallback } from './lib/llm/manager';
	import type { LlmState } from './lib/llm/types';
	import { runBench, type BenchIteration, type BenchResult } from './lib/bench';
	import { probeWebGpu, summarizeProbe, type WebGpuProbeResult } from './lib/llm/webgpu-probe';

	// llama-bench convention: fixed prompt for prefill, fixed 128-token
	// generation, 3 iterations.
	const BENCH_PROMPT = 'The quick brown fox jumps over the lazy dog. '.repeat(16);
	const TG = 128;
	const ITERS = 3;

	// Mirror the store into a reactive local so the component updates.
	let modelState: LlmState = $state({ modelId: null, activeModel: null, backend: 'wasm', status: { state: 'idle' } });
	$effect(() => llm.subscribe((v) => (modelState = v)));

	// Auto-start: begin downloading the selected (default) model on load,
	// so the user does not have to re-select it to trigger the download.
	// The one-shot flag is essential: `selectModel` does not flip the status
	// synchronously (the async loadModel does), so without it this effect
	// would re-trigger itself on every store update → effect_update_depth_exceeded.
	let autoStarted = $state(false);
	$effect(() => {
		if (!autoStarted && modelState.status.state === 'idle' && modelState.modelId) {
			autoStarted = true;
			selectModel(modelState.modelId);
		}
	});
	const model = $derived(modelState.activeModel ?? getModel(modelState.modelId ?? undefined));

	let allowCpu = $state(false);
	let benching = $state(false);
	let iterations: BenchIteration[] = $state([]);
	let result: BenchResult | null = $state(null);
	let benchError: string | null = $state(null);

	const progressPct = $derived(
		modelState.status.state === 'downloading' ? Math.round(modelState.status.progress * 100) : 0
	);

	function pick(id: string) {
		result = null;
		iterations = [];
		benchError = null;
		selectModel(id);
	}

	async function bench() {
		if (benching) return;
		benching = true;
		iterations = [];
		result = null;
		benchError = null;
		try {
			result = await runBench(
				{ prompt: BENCH_PROMPT, tg: TG, iters: ITERS },
				(it) => (iterations = [...iterations, it])
			);
		} catch (err) {
			benchError = err instanceof Error ? err.message : String(err);
		} finally {
			benching = false;
		}
	}

	function copyJson() {
		if (result) void navigator.clipboard.writeText(JSON.stringify(result, null, 2));
	}

	const fmt = (v: number | null | undefined): string =>
		v == null ? '—' : String(Math.round(v * 10) / 10);

	// Live WebGPU compatibility probe — runs on load (a few hundred ms),
	// well before any download, and gates + educates.
	let probe = $state<WebGpuProbeResult | null>(null);
	let probing = $state(true);
	async function recheck() {
		probing = true;
		probe = null;
		probe = await probeWebGpu();
		probing = false;
	}
	// No reactive reads inside → runs exactly once on mount.
	$effect(() => {
		void recheck();
	});
</script>

<main>
	<h1>llm-bench</h1>
	<p class="sub">
		Llama-bench-style performance benchmark · LiteRT-LM on
		<strong>{modelState.backend === 'webgpu' ? 'WebGPU' : 'WASM/CPU'}</strong>
		<span class="hint">(auto-detected — WebGPU when available, WASM fallback)</span>
	</p>

	<section>
		<h2>WebGPU compatibility</h2>
		{#if probing || !probe}
			<p class="hint">Checking this browser…</p>
		{:else}
			<p class={probe.verdict === 'ok' ? 'ok' : 'err'}>{summarizeProbe(probe)}</p>
			<ul class="checks">
				<li class={probe.secure ? 'ok' : 'err'}>{probe.secure ? '✓' : '✗'} Secure context (https:// or localhost)</li>
				<li class={probe.apiPresent ? 'ok' : 'err'}>{probe.apiPresent ? '✓' : '✗'} WebGPU API (<code>navigator.gpu</code>)</li>
				<li class={probe.adapter ? 'ok' : 'err'}>{probe.adapter ? '✓' : '✗'} GPU adapter</li>
				<li class={probe.device ? 'ok' : 'err'}>{probe.device ? '✓' : '✗'} GPU device</li>
				<li class={probe.compute ? 'ok' : 'err'}>
					{probe.compute ? '✓' : '✗'} Compute verification (test shader){probe.computeDetail ? ` — ${probe.computeDetail}` : ''}
				</li>
			</ul>
			{#if probe.description}
				<p class="hint">Adapter: {probe.description}{probe.vendor ? ` · vendor ${probe.vendor}` : ''}</p>
			{/if}
			{#if probe.maxBufferSize}
				<p class="hint">Max GPU buffer: {formatBytes(probe.maxBufferSize)}</p>
			{/if}
			{#if probe.features.length > 0}
				<p class="hint">
					Features: {probe.features.map((f) => `${f.name} ${f.supported ? '✓' : '✗'}`).join(' · ')}
			</p>
			{/if}
			<button onclick={recheck}>Re-check</button>
		{/if}

		<label class="cpufallback">
			<input
				type="checkbox"
				checked={allowCpu}
				onchange={(e) => {
					allowCpu = e.currentTarget.checked;
					setAllowCpuFallback(allowCpu);
					// Probe failed and CPU is now allowed → retry immediately.
					if (allowCpu && modelState.status.state === 'error' && model) pick(model.id);
				}}
			/>
			<span>
				Allow WASM/CPU fallback — run on the CPU when WebGPU is unavailable
				<span class="hint">(much slower; results are labeled <code>backend: “wasm”</code> — useful to see the WebGPU speedup. The CPU path needs ~3× the model file size in RAM — a 2 GB model will OOM-kill a phone tab; pick a small model)</span>
			</span>
		</label>

		<details>
			<summary>What is WebGPU — and why does this benchmark need it?</summary>
			<p>
				<b>WebGPU</b> is the modern browser API for graphics and <b>compute</b> —
				the successor to WebGL, with a low-overhead interface to the
				platform's native backend (Vulkan on Windows/Linux/Android, Metal on
				macOS/iOS, D3D12 on Windows).
			</p>
			<p>
				<b>Why it matters for LLMs in the browser:</b> inference is almost
				entirely matrix multiplications and attention — exactly the kind of
				massively parallel work compute shaders run on the GPU. The
				LiteRT-LM engine compiles the model's kernels to WebGPU compute, so
				prefill and token generation run on your graphics hardware instead of
				a single CPU thread (WASM). The model and all data stay on the device —
				no server round-trips.
			</p>
			<h3>Pros</h3>
			<ul>
				<li>GPU parallelism → dramatically faster prefill/generation than CPU/WASM</li>
				<li>One portable API over Vulkan/Metal/D3D12 — no per-GPU driver code in the page</li>
				<li>Fully local: private, offline-capable, no per-token cost</li>
			</ul>
			<h3>Cons / caveats</h3>
			<ul>
				<li>Younger than WebGL — support depends on browser <i>and</i> version</li>
				<li>Requires a <b>secure context</b> (https:// or localhost) — over plain http, <code>navigator.gpu</code> is undefined</li>
				<li>Adapter quality varies: VMs, remote desktops, and some drivers expose no adapter (or a slow software one)</li>
				<li>Worst case: the adapter <i>and</i> device exist but <b>compute is broken</b> — the classic Android-with-“unsafe-WebGPU”-flag failure, where the model loads and then produces zero tokens. The probe above catches this with a test shader <i>before</i> the download</li>
				<li>Per-device feature/limit differences (buffer sizes, f16 support) affect what the engine can do</li>
				<li>GPU code is much harder to debug than CPU code</li>
				<li>The WASM/CPU fallback works but is far slower — the benchmark refuses it <i>by default</i> (the numbers would mislead); the checkbox above enables it explicitly, and those results are labeled <code>backend: “wasm”</code></li>
			</ul>
			<h3>Browser support (2025)</h3>
			<ul>
				<li><b>Chrome / Edge</b> — Windows, macOS, Linux since Chrome 113 (2023). On <b>Android</b> it often requires the “Enable unsafe WebGPU” flag, and quality depends on the GPU driver (Mali / Adreno) — the compute test above tells you if yours works</li>
				<li><b>Firefox</b> — 141+ (Windows first, late 2024)</li>
				<li><b>Safari</b> — 26+ (macOS / iOS, 2025)</li>
			</ul>
		</details>
	</section>

	<section>
		<h2>Model</h2>
		<div class="row">
			<select class="grow" value={model?.id ?? ''} onchange={(e) => pick(e.currentTarget.value)}>
				{#each MODELS as m}
					<option value={m.id}>{m.name} — {formatBytes(m.sizeBytes)}</option>
				{/each}
			</select>
		</div>
		{#if model}
			<p class="hint">{model.description}</p>
		{/if}

		<div class="status">
			{#if modelState.status.state === 'idle'}
				<p>Model not loaded yet.</p>
				<button onclick={() => model && pick(model.id)}>Download model</button>
			{:else if modelState.status.state === 'downloading'}
				<p>Downloading model… {progressPct}%</p>
				<progress value={progressPct} max={100}></progress>
			{:else if modelState.status.state === 'loading'}
				<p>Loading model into the {modelState.backend} engine…</p>
			{:else if modelState.status.state === 'ready'}
				<p class="ok">Model ready.</p>
			{:else}
				<p class="err">{modelState.status.message}</p>
				<button onclick={() => model && pick(model.id)}>Retry</button>
			{/if}
		</div>
	</section>

	<section>
		<h2>Benchmark</h2>
		<p class="hint">
			{BENCH_PROMPT.length} chars prompt · tg over {TG} tokens · {ITERS} iterations
			· best + median — the engine does not stream per token, so tg is
			measured as (t<sub>{TG}</sub> − t<sub>1</sub>) / ({TG} − 1) and ttft as the
			pure-prefill part of the 1-token run
		</p>
		<button class="primary" disabled={benching || modelState.status.state !== 'ready'} onclick={bench}>
			{benching ? 'Running…' : 'Run benchmark'}
		</button>

		{#if modelState.backend === 'wasm' && modelState.status.state === 'ready'}
			<p class="err">⚠ Running on WASM/CPU — the numbers below are CPU performance, not WebGPU. (If the tab crashes on load, that is an out-of-memory kill — use a smaller model.)</p>
		{/if}

		{#if benching && iterations.length > 0}
			<p class="hint">iteration {iterations.length}/{ITERS} done…</p>
		{/if}
		{#if benchError}
			<p class="err">{benchError}</p>
		{/if}

		{#if iterations.length > 0}
			<div class="tablewrap">
			<table>
				<thead>
					<tr>
						<th></th><th>pp (tok/s)</th><th>tg (tok/s)</th><th>ttft (ms)</th>
						<th>t₁ (ms)</th><th>t₁₂₈ (ms)</th>
						<th>pp eng</th><th>tg eng</th><th>ctx tok</th>
					</tr>
					</thead>
				<tbody>
					{#each iterations as it}
						<tr>
							<td>run {it.index}</td>
							<td>{fmt(it.pp)}</td>
							<td>{fmt(it.tg)}</td>
							<td>{it.ttftMs}</td>
							<td>{it.tA_ms}</td>
							<td>{it.tB_ms}</td>
							<td>{fmt(it.pp_engine)}</td>
							<td>{fmt(it.tg_engine)}</td>
							<td>{it.tokens_in_context ?? '—'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			</div>
		{/if}

		{#if result}
			{#if result.stopped}
				<div class="warn">
					<b>Stopped early:</b> {result.stopped} — showing
					{result.iterations.length} of {result.params.iters} iterations; the
					aggregates use the completed runs.
					{#if result.engine_log && result.engine_log.length > 0}
						<pre>{result.engine_log.join('\n')}</pre>
					{/if}
				</div>
			{/if}
			<table class="summary">
				<thead>
					<tr><th></th><th>best</th><th>median</th></tr>
				</thead>
				<tbody>
					<tr><td>pp (tok/s)</td><td>{fmt(result.pp.best)}</td><td>{fmt(result.pp.median)}</td></tr>
					<tr><td>tg (tok/s)</td><td>{fmt(result.tg.best)}</td><td>{fmt(result.tg.median)}</td></tr>
					<tr><td>ttft (ms)</td><td colspan={2}>{result.ttft_ms}</td></tr>
				</tbody>
			</table>

			<h3>Result JSON <button onclick={copyJson}>copy</button></h3>
			<pre>{JSON.stringify(result, null, 2)}</pre>
		{/if}
	</section>

	<footer>
		<p class="hint">
			<b>bench v4 — marginal timing + chat templates</b> (tg from t₁₂₈ − t₁; wall-clock is
			preferred, the engine's own counters are shown as reference — "eng"
			columns). If you do not see the eng columns you are running a stale
			build — rebuild and hard-refresh. Models download once and are kept in
			the browser's Cache
			Storage. The inference engine runs in a Web Worker (LiteRT-LM WASM +
			WebGPU, loaded dynamically from the pinned <code>@litert-lm/core</code>
			build).
		</p>
	</footer>
</main>

<style>
	main {
		font-family: system-ui, sans-serif;
		max-width: 720px;
		margin: 2rem auto;
		padding: 0 1rem;
	}
	/* the 9-column iteration table scrolls horizontally on narrow screens */
	.tablewrap { overflow-x: auto; }
	h1 { font-size: 1.4rem; margin-bottom: 0; }
	.sub { color: #555; margin-top: 0.25rem; }
	.hint { color: #777; font-size: 0.9rem; }
	.warn {
		border-left: 4px solid #c0392b;
		background: #fdecea;
		padding: 0.5rem 0.75rem;
		margin: 0.75rem 0;
		font-size: 0.9rem;
	}
	.warn pre {
		font-size: 0.78rem;
		overflow-x: auto;
		white-space: pre-wrap;
		margin: 0.5rem 0 0;
	}
	h2 { font-size: 1.1rem; margin-top: 1.5rem; }
	h3 { font-size: 0.95rem; margin: 1rem 0 0.25rem; }
	.row { display: flex; gap: 0.5rem; }
	.grow { flex: 1; }
	select, button {
		font: inherit;
		padding: 0.4rem 0.8rem;
	}
	button.primary {
		font-size: 1rem;
		padding: 0.5rem 1.2rem;
		cursor: pointer;
	}
	button:disabled { opacity: 0.5; cursor: default; }
	.status { margin: 0.5rem 0; }
	.ok { color: #0a0; font-weight: bold; }
	.err { color: #a00; }
	progress { width: 100%; margin-top: 0.4rem; }
	.checks { list-style: none; padding-left: 0.25rem; }
	.cpufallback {
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		margin: 0.75rem 0;
		cursor: pointer;
	}
	.cpufallback input { width: 1.1rem; height: 1.1rem; flex: none; }
	details { margin: 0.75rem 0; }
	details summary { cursor: pointer; font-weight: 600; }
	details p, details li { max-width: 65ch; }
	table { border-collapse: collapse; margin: 0.75rem 0; }
	td, th { border: 1px solid #ccc; padding: 0.35rem 0.7rem; text-align: right; }
	td:first-child, th:first-child { text-align: left; }
	pre {
		background: #f4f4f4;
		padding: 1rem;
		overflow-x: auto;
		white-space: pre-wrap;
		word-break: break-all;
	}
	footer { margin-top: 2rem; border-top: 1px solid #ddd; padding-top: 0.75rem; }

	/* ── mobile ─────────────────────────────────────────────── */
	@media (max-width: 640px) {
		main { margin: 0.75rem auto; padding: 0 0.75rem; }
		h1 { font-size: 1.2rem; }
		h2 { font-size: 1rem; margin-top: 1.25rem; }
		select, button { font-size: 1rem; }
		button { min-height: 44px; } /* comfortable tap target */
		button.primary { width: 100%; padding: 0.65rem 1rem; }
		table { font-size: 0.85rem; }
		td, th { padding: 0.3rem 0.5rem; }
		pre { font-size: 0.78rem; padding: 0.6rem; }
		.hint { font-size: 0.85rem; }
	}
</style>
