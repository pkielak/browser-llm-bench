<script lang="ts">
	import { MODELS, formatBytes, getModel } from './lib/llm/models';
	import { llm, selectModel } from './lib/llm/manager';
	import type { LlmState } from './lib/llm/types';
	import { runBench, type BenchIteration, type BenchResult } from './lib/bench';

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
</script>

<main>
	<h1>llm-bench</h1>
	<p class="sub">
		Llama-bench-style performance benchmark · LiteRT-LM on
		<strong>{modelState.backend === 'webgpu' ? 'WebGPU' : 'WASM/CPU'}</strong>
		<span class="hint">(auto-detected — WebGPU when available, WASM fallback)</span>
	</p>

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

		{#if benching && iterations.length > 0}
			<p class="hint">iteration {iterations.length}/{ITERS} done…</p>
		{/if}
		{#if benchError}
			<p class="err">{benchError}</p>
		{/if}

		{#if iterations.length > 0}
			<table>
				<thead>
					<tr><th></th><th>pp (tok/s)</th><th>tg (tok/s)</th><th>ttft (ms)</th></tr>
				</thead>
				<tbody>
					{#each iterations as it}
						<tr>
							<td>run {it.index}</td>
							<td>{fmt(it.pp)}</td>
							<td>{fmt(it.tg)}</td>
							<td>{it.ttftMs}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}

		{#if result}
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
			Models download once and are kept in the browser's Cache Storage. The
			inference engine runs in a Web Worker (LiteRT-LM WASM + WebGPU, loaded
			dynamically from the pinned <code>@litert-lm/core</code> build).
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
	h1 { font-size: 1.4rem; margin-bottom: 0; }
	.sub { color: #555; margin-top: 0.25rem; }
	.hint { color: #777; font-size: 0.9rem; }
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
</style>
