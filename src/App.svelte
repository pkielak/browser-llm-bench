<script lang="ts">
	import { MODELS, formatBytes, getModel } from './lib/llm/models';
	import { llm, selectModel, setAllowCpuFallback, setForceWebGpu } from './lib/llm/manager';
	import type { LlmModel, LlmState } from './lib/llm/types';
	import { runBench, type BenchIteration, type BenchResult } from './lib/bench';
	import {
		checkChromeAiEligibility,
		openChromeAiSession,
		closeChromeAiSession,
		runChromeAiBench,
		type ChromeAiEligibility,
		type ChromeAiIteration,
		type ChromeAiResult
	} from './lib/chrome-ai';
	import { probeWebGpu, summarizeProbe, type WebGpuProbeResult } from './lib/llm/webgpu-probe';
	import Accordion from './lib/Accordion.svelte';

	// llama-bench convention: a fixed-length prompt for prefill (pp512 by
	// default), a fixed 128-token generation, 3 iterations.
	const TG = 128;
	const ITERS = 3;
	const PROMPT_PRESETS = [256, 512, 1024];

	// The CPU (WASM) path needs ~3× the file size in RAM (and the 32-bit
	// WASM heap is capped at 4 GB) — bigger models OOM on phones, so they
	// are greyed out while the CPU fallback is enabled.
	const WASM_SAFE_MAX_BYTES = 512 * 1024 * 1024;

	// Reference backend: Chrome's own on-device model (Prompt API). Not a
	// LiteRT model — a synthetic LlmModel so the picker and UI handle it
	// uniformly.
	const CHROME_AI_ID = 'chrome-ai';
	const CHROME_AI_MODEL: LlmModel = {
		id: CHROME_AI_ID,
		name: 'Chrome built-in AI (Gemini Nano)',
		description:
			'Reference backend — Chrome’s own on-device model. No model download, no WebGPU probe (Chrome owns the engine). Metrics are estimates: the API exposes no token counts or output-length control.',
		url: '',
		sizeBytes: 0
	};

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

	// Header status pill.
	const statusInfo = $derived.by(() => {
		if (chromeSelected) {
			if (chromeError) return { cls: 'error', text: 'chrome ai error' };
			if (chromeBusy) return { cls: 'busy', text: 'preparing chrome ai' };
			if (chromeSessionReady) return { cls: 'ready', text: 'model ready' };
			return { cls: 'idle', text: 'not loaded' };
		}
		const s = modelState.status.state;
		switch (s) {
			case 'idle':
				return { cls: 'idle', text: 'not loaded' };
			case 'downloading':
				return { cls: 'busy', text: `downloading ${Math.round(modelState.status.progress * 100)}%` };
			case 'loading':
				return { cls: 'busy', text: 'loading engine' };
			case 'ready':
				return { cls: 'ready', text: 'model ready' };
			default:
				return { cls: 'error', text: 'error' };
		}
	});

	let allowCpu = $state(false);
	let forceGpu = $state(false);
	let promptTarget = $state(512);
	let benching = $state(false);
	let iterations: (BenchIteration | ChromeAiIteration)[] = $state([]);
	let result: (BenchResult | ChromeAiResult) | null = $state(null);

	// Chrome built-in AI state (local — it never touches the LiteRT store).
	let chromeElig: ChromeAiEligibility | null = $state(null);
	let chromeChecking = $state(true);
	let chromeSelected = $state(false);
	let chromeSessionReady = $state(false);
	let chromeBusy: string | null = $state(null);
	let chromeError: string | null = $state(null);

	const activeModel = $derived(chromeSelected ? CHROME_AI_MODEL : model);
	const isChromeAi = $derived(activeModel?.id === CHROME_AI_ID);
	// Selectable when the API exists AND Chrome reports the model as usable
	// ('downloading' is fine — create() continues the download).
	const chromeAvailable = $derived.by(() => {
		const e = chromeElig;
		return !!e && e.present && e.availability !== 'unavailable';
	});
	const litertIterations = $derived(
		iterations.filter((i) => !('tgEst' in i)) as BenchIteration[]
	);
	const chromeIterations = $derived(iterations.filter((i) => 'tgEst' in i) as ChromeAiIteration[]);
	let benchError: string | null = $state(null);
	let copied = $state(false);

	const progressPct = $derived(
		modelState.status.state === 'downloading' ? Math.round(modelState.status.progress * 100) : 0
	);
	const benchPct = $derived(benching ? Math.round((iterations.length / ITERS) * 100) : 0);

	function pick(id: string) {
		result = null;
		iterations = [];
		benchError = null;
		if (id === CHROME_AI_ID) {
			void pickChromeAi();
			return;
		}
		// Leaving the reference backend: drop its session (frees memory).
		chromeSelected = false;
		chromeSessionReady = false;
		chromeBusy = null;
		chromeError = null;
		closeChromeAiSession();
		selectModel(id);
	}

	async function pickChromeAi() {
		chromeSelected = true;
		chromeSessionReady = false;
		chromeBusy = 'Creating session…';
		chromeError = null;
		closeChromeAiSession();
		try {
			// First use on this origin makes Chrome download its model —
			// progress arrives via the monitor when Chrome provides it.
			await openChromeAiSession(
				(p) => (chromeBusy = `Downloading Chrome’s model ${Math.round(p * 100)}%`)
			);
			chromeBusy = null;
			chromeSessionReady = true;
		} catch (err) {
			chromeBusy = null;
			chromeError = err instanceof Error ? err.message : String(err);
		}
	}

	async function bench() {
		if (benching) return;
		benching = true;
		iterations = [];
		result = null;
		benchError = null;
		try {
			if (isChromeAi) {
				result = await runChromeAiBench(promptTarget, ITERS, (it) => (iterations = [...iterations, it]));
			} else {
				result = await runBench(
					{ promptTokens: promptTarget, tg: TG, iters: ITERS },
					(it) => (iterations = [...iterations, it])
				);
			}
			document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		} catch (err) {
			benchError = err instanceof Error ? err.message : String(err);
		} finally {
			benching = false;
		}
	}

	function copyJson() {
		if (!result) return;
		void navigator.clipboard.writeText(JSON.stringify(result, null, 2));
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}

	const fmt = (v: number | null | undefined): string =>
		v == null ? '—' : String(Math.round(v * 10) / 10);
	const fmtMs = (v: number | null | undefined): string =>
		v == null ? '—' : v.toLocaleString();

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

	// Same for the Chrome built-in AI eligibility (cheap, no download).
	async function recheckChrome() {
		chromeChecking = true;
		chromeElig = null;
		chromeElig = await checkChromeAiEligibility();
		chromeChecking = false;
	}
	$effect(() => {
		void recheckChrome();
	});
</script>

<header>
	<div>
		<h1>llm-bench</h1>
		<p class="sub">
			Llama-bench-style performance benchmark ·
			{isChromeAi ? 'Chrome built-in AI (reference)' : `LiteRT-LM on ${modelState.backend === 'webgpu' ? 'WebGPU' : 'WASM/CPU'}`}
		</p>
	</div>
	<div class="pills">
		<span class="pill backend">{isChromeAi ? 'Chrome AI' : modelState.backend === 'webgpu' ? 'WebGPU' : 'WASM/CPU'}</span>
		<span class="pill {statusInfo.cls}">{statusInfo.text}</span>
	</div>
</header>

<main>
	<section class="card">
		<h2><span class="step">1</span> GPU compatibility</h2>
		{#if probing || !probe}
			<p class="hint">Checking this browser…</p>
		{:else}
			<p class={probe.verdict === 'ok' ? 'ok' : 'err'}>
				{probe.verdict === 'ok' ? '✓' : '✗'} {summarizeProbe(probe)}
			</p>
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
					if (allowCpu) {
						// The CPU path needs ~3× the file size in RAM — the big
						// models OOM, so switch to the largest safe one.
						const safe = MODELS.filter((m) => m.sizeBytes <= WASM_SAFE_MAX_BYTES);
						if (!chromeSelected && model && model.sizeBytes > WASM_SAFE_MAX_BYTES && safe.length > 0) {
							pick(safe.sort((a, b) => b.sizeBytes - a.sizeBytes)[0].id);
						} else if (modelState.status.state === 'error' && model) pick(model.id);
					}
				}}
			/>
			<span>
				<b>Allow WASM/CPU fallback</b> — run on the CPU when WebGPU is unavailable
				<span class="hint"
					>(much slower; results are labeled <code>backend: “wasm”</code>. The CPU path
					needs ~3× the model file size in RAM — big models will OOM, so they are
					disabled below)</span>
			</span>
		</label>

		<label class="cpufallback">
			<input
				type="checkbox"
				checked={forceGpu}
				onchange={(e) => {
					forceGpu = e.currentTarget.checked;
					setForceWebGpu(forceGpu);
					// The probe gate blocked the load and consent is now given → retry.
					if (forceGpu && modelState.status.state === 'error' && model) pick(model.id);
				}}
			/>
			<span>
				<b>Force WebGPU anyway</b> — proceed even if the probe reports problems
				<span class="hint"
					>(explicit consent: the engine may load and then produce zero tokens — the classic
					broken-driver failure. Useful if you believe the probe is wrong, e.g. a false
					negative on an unusual GPU)</span>
			</span>
		</label>

		<Accordion title="What is WebGPU — and why does this benchmark need it?">
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
		</Accordion>
	</section>

	<section class="card">
		<h2><span class="step">2</span> Model</h2>
		<select class="grow" value={activeModel?.id ?? ''} onchange={(e) => pick(e.currentTarget.value)}>
			<option
				value={CHROME_AI_ID}
				disabled={!chromeAvailable}
			>{CHROME_AI_MODEL.name} · reference{
				chromeChecking
					? ' (checking…)' 
					: chromeElig?.availability === 'available'
						? ' · available'
						: chromeElig?.availability === 'downloading'
							? ' · model downloading'
							: ' (not available in this browser)'
				}</option>
			{#each MODELS as m}
				<option
					value={m.id}
					disabled={allowCpu && m.sizeBytes > WASM_SAFE_MAX_BYTES}
				>{m.name}{m.wasmOnly ? ' · CPU only' : ''} — {formatBytes(m.sizeBytes)}{allowCpu && m.sizeBytes > WASM_SAFE_MAX_BYTES ? ' (too big for CPU)' : ''}</option>
			{/each}
		</select>
		{#if activeModel}
			<p class="hint">{activeModel.description}</p>
		{/if}
		{#if chromeSelected && chromeBusy}
			<p class="hint">{chromeBusy}</p>
		{/if}
		{#if chromeError}
			<div class="alert err">{chromeError}</div>
		{/if}

		{#if chromeChecking || !chromeElig}
			<p class="hint">Checking Chrome built-in AI eligibility…</p>
		{:else}
			<ul class="checks">
				<li class={chromeElig.present ? 'ok' : 'err'}>
					{chromeElig.present ? '✓' : '✗'} Prompt API (<code>window.ai.language.model</code>)
				</li>
				<li
					class={
						chromeElig.availability === 'available'
							? 'ok'
							: chromeElig.availability === 'downloading'
								? 'ok'
								: 'err'
					}
				>
					{chromeElig.availability === 'downloading' ? '↓' : chromeElig.availability === 'available' ? '✓' : '✗'}
					Model availability: <code>{chromeElig.availability ?? 'n/a'}</code>
				</li>
			</ul>
			{#if chromeElig.present && chromeElig.availability === 'unavailable'}
				<p class="hint">
					Chrome needs 22 GB free space and a GPU with &gt;4 GB VRAM (or 16 GB
					RAM + 4 cores) — details at <code>chrome://on-device-internals</code>.
					Desktop Chrome only; not on Android/iOS.
				</p>
			{/if}
			<button onclick={() => void recheckChrome()}>Re-check</button>
		{/if}

		<Accordion title="Chrome built-in AI — what is it?">
			<p>
				<b>Chrome ships its own on-device LLM.</b> Since Chrome 148 (desktop),
				the browser includes <b>Gemini Nano</b> — a small generative model that
				runs entirely on your machine — and exposes it to web pages through
				the <b>Prompt API</b> (<code>window.ai.language.model</code>).
			</p>
			<ul>
				<li><b>Model is opaque</b> — Chrome chooses the version and updates it with browser releases; there is nothing to pin. Its current size is visible at <code>chrome://on-device-internals</code></li>
				<li><b>Downloaded by Chrome, not by us</b> — on first use per origin Chrome fetches the model (the progress bar above is Chrome's download); afterwards it runs fully offline and <b>no data leaves your device</b></li>
				<li><b>Hardware requirements</b> — Windows 10/11, macOS 13+, Linux, or Chromebook Plus; 22 GB free space; a GPU with &gt;4 GB VRAM <i>or</i> 16 GB RAM + 4 cores. <b>Not available on Android/iOS</b></li>
				<li><b>API surface</b> — <code>LanguageModel.create()</code> → a session with <code>prompt()</code> and <code>promptStreaming()</code>; no output-length control, no token counts, sampling parameters are origin-trial-gated</li>
			</ul>
			<p>
				<b>Why it is a reference backend here:</b> the missing token counts and
				output-length control make its metrics <b>estimates</b> (tg = chars/4 over
				wall time, ttft = time to first chunk), and the model isn't comparable
				with our pinned LiteRT builds. It answers a different question: what
				does the OS-vendored on-device model do on <i>this exact hardware</i> —
				a free baseline to put the LiteRT numbers in context.
		</p>
		</Accordion>

		<div class="status">
			{#if modelState.status.state === 'idle'}
				<p>Model not loaded yet.</p>
				<button class="primary" onclick={() => model && pick(model.id)}>Download model</button>
			{:else if modelState.status.state === 'downloading'}
				<p>Downloading model… {progressPct}%</p>
				<progress value={progressPct} max={100}></progress>
			{:else if modelState.status.state === 'loading'}
				<p class="busy">Loading model into the {modelState.backend} engine…</p>
			{:else if modelState.status.state === 'ready'}
				<p class="ok">✓ Model ready.</p>
			{:else}
				<p class="err">{modelState.status.message}</p>
				<button onclick={() => model && pick(model.id)}>Retry</button>
			{/if}
		</div>
	</section>

	<section class="card">
		<h2><span class="step">3</span> Benchmark</h2>
		<div class="row">
			<label for="promptlen">Prompt length</label>
			<select id="promptlen" value={promptTarget} onchange={(e) => (promptTarget = Number(e.currentTarget.value))}>
				{#each PROMPT_PRESETS as p}
					<option value={p}>~{p} tokens (pp{p})</option>
				{/each}
			</select>
		</div>
		{#if isChromeAi}
			<p class="hint">
				{ITERS} iterations · best + median — the Prompt API has no output-length
				control, so a “~100 words” nudge keeps the runs comparable, and all
				metrics are <b>estimates</b> (no token counts from the API)
			</p>
		{:else}
			<p class="hint">
				tg over {TG} tokens · {ITERS} iterations · best + median — the engine does not
				stream per token, so tg is measured as (t<sub>{TG}</sub> − t<sub>1</sub>) / ({TG} − 1)
				and ttft as the pure-prefill part of the 1-token run
			</p>
		{/if}

		<Accordion title="How the numbers are measured">
			<p>
				<b>Marginal timing.</b> This engine build does not stream per token — the
				whole generation arrives in one batch at the end — so each iteration
				makes two runs, each from a fresh conversation (clean KV cache):
			</p>
			<ul>
				<li>t<sub>1</sub> = prefill + 1 decode step</li>
				<li>t<sub>128</sub> = prefill + 128 decode steps</li>
				<li>1 decode step = (t<sub>128</sub> − t<sub>1</sub>) / 127 — the prefill cancels out → <b>tg</b> = 1000 / step</li>
				<li>prefill = t<sub>1</sub> − 1 step → <b>ttft</b> (ms) and <b>pp</b> = prompt tokens / prefill</li>
			</ul>
			<p>
				<b>Wall-clock is primary.</b> The engine's internal counters (the
				"eng" columns) are recorded for reference only — in the pinned build
				they are populated but self-inconsistent between runs.
			</p>
			<p>
				<b>Prompt.</b> A natural English passage synthesized to the selected
				token target (~4 chars/token), wrapped in the model's chat template.
				llama-bench itself uses random vocabulary token IDs (exact counts);
				the LiteRT API takes text only, so we approximate and record the
				measured count in the result.
			</p>
			<p>
				<b>Aggregation.</b> 3 iterations, best + median reported. If the engine
				runs out of memory on a later iteration (the WASM heap is not released
				between conversations), the completed iterations are still reported.
			</p>
			<p>
				<b>Chrome built-in AI (reference).</b> The optional entry benchmarks
				Chrome’s own on-device Gemini Nano via the Prompt API. Its metrics
				are estimates — no token counts, no output-length control, no
				prefill/decode split: tg is chars/4 over wall time, ttft is time to
				first chunk. It contextualizes the LiteRT numbers on the same
				hardware; it is not meant to compete with them.
			</p>
		</Accordion>

		<button
			class="primary run"
			disabled={benching || (isChromeAi ? !chromeSessionReady : modelState.status.state !== 'ready')}
			onclick={bench}
		>
			{benching ? `Running… iteration ${Math.min(iterations.length + 1, ITERS)}/${ITERS}` : 'Run benchmark'}
		</button>
		{#if benching && iterations.length > 0}
			<progress class="benchprogress" value={benchPct} max={100}></progress>
		{/if}

		{#if !isChromeAi && modelState.backend === 'wasm' && modelState.status.state === 'ready'}
			<div class="alert warn">
				⚠ Running on <b>WASM/CPU</b> — the numbers are CPU performance, not WebGPU.
				(If the tab crashes on load, that is an out-of-memory kill — use a smaller model.)
			</div>
		{/if}
		{#if !isChromeAi && forceGpu && probe && probe.verdict !== 'ok' && modelState.status.state === 'ready'}
			<div class="alert warn">
				⚠ <b>WebGPU forced</b> despite probe problems ({summarizeProbe(probe)}).
				Zero tokens are possible — these numbers are at your consent.
			</div>
			{/if}
		{#if benchError}
			<div class="alert err">{benchError}</div>
		{/if}

		{#if isChromeAi && chromeIterations.length > 0}
			<div class="tablewrap">
				<table>
					<thead>
						<tr>
							<th></th><th>ttft (ms)</th><th>total (ms)</th><th>chars</th>
							<th>tokens (est)</th><th>tok/s (est)</th>
						</tr>
					</thead>
					<tbody>
						{#each chromeIterations as it}
							<tr>
								<td>run {it.index}</td>
								<td>{fmtMs(it.ttftMs)}</td>
								<td>{fmtMs(it.totalMs)}</td>
								<td>{it.textChars}</td>
								<td>{it.tokensEst}</td>
								<td>{fmt(it.tgEst)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		{#if !isChromeAi && litertIterations.length > 0}
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
						{#each litertIterations as it}
							<tr>
								<td>run {it.index}</td>
								<td>{fmt(it.pp)}</td>
								<td>{fmt(it.tg)}</td>
								<td>{fmtMs(it.ttftMs)}</td>
								<td>{fmtMs(it.tA_ms)}</td>
								<td>{fmtMs(it.tB_ms)}</td>
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
			<div id="results">
				{#if result.backend === 'chrome-builtin'}
					<div class="alert warn">
						⚠ <b>Reference backend</b> — {result.notes}
					</div>
					<div class="stats">
						<div class="stat">
							<div class="num">{fmt(result.tg_est.best)}</div>
							<div class="label">tg · tok/s (est, best)</div>
							<div class="sub2">median {fmt(result.tg_est.median)}</div>
						</div>
						<div class="stat">
							<div class="num">{fmtMs(result.ttft_ms.best)}</div>
							<div class="label">ttft · ms (first chunk)</div>
							<div class="sub2">median {fmtMs(result.ttft_ms.median)}</div>
						</div>
					</div>
					<div class="tablewrap">
						<table>
							<thead>
								<tr>
									<th></th><th>ttft (ms)</th><th>total (ms)</th><th>chars</th>
									<th>tokens (est)</th><th>tok/s (est)</th>
								</tr>
							</thead>
							<tbody>
								{#each result.iterations as it}
									<tr>
										<td>run {it.index}</td>
										<td>{fmtMs(it.ttftMs)}</td>
										<td>{fmtMs(it.totalMs)}</td>
										<td>{it.textChars}</td>
										<td>{it.tokensEst}</td>
										<td>{fmt(it.tgEst)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else}

				{#if result.stopped}
					<div class="alert warn">
						<b>Stopped early:</b> {result.stopped} — showing
						{result.iterations.length} of {result.params.iters} iterations; the
						aggregates use the completed runs.
						{#if result.engine_log && result.engine_log.length > 0}
							<pre>{result.engine_log.join('\n')}</pre>
						{/if}
					</div>
				{/if}

				<div class="stats">
					<div class="stat">
						<div class="num">{fmt(result.pp.best)}</div>
						<div class="label">pp{result.params.prompt_tokens} · tok/s (best)</div>
						<div class="sub2">median {fmt(result.pp.median)}</div>
					</div>
					<div class="stat">
						<div class="num">{fmt(result.tg.best)}</div>
						<div class="label">tg · tok/s (best)</div>
						<div class="sub2">median {fmt(result.tg.median)}</div>
					</div>
					<div class="stat">
						<div class="num">{fmtMs(result.ttft_ms)}</div>
						<div class="label">ttft · ms</div>
						<div class="sub2">time to first token</div>
					</div>
				</div>

				{/if}

				<Accordion title="Result JSON">
					<div class="json-head">
						<button class="copy" onclick={copyJson}>{copied ? 'Copied ✓' : 'Copy'}</button>
					</div>
					<pre>{JSON.stringify(result, null, 2)}</pre>
				</Accordion>
			</div>
		{/if}
	</section>

	<footer>
		<p class="hint">
			<b>bench v7</b> — llama-bench-style prompt lengths (pp256/pp512/pp1024),
			Chrome built-in AI (Gemini Nano) reference backend,
			marginal timing (tg from t₁₂₈ − t₁), chat templates, partial results on
			engine OOM. Wall-clock is preferred; the engine's own counters are shown
			as reference ("eng" columns). Models download once and are kept in the
			browser's Cache Storage. The inference engine runs in a Web Worker
			(LiteRT-LM WASM + WebGPU, pinned <code>@litert-lm/core</code> build).
		</p>
	</footer>
</main>

<style>
	:root {
		--bg: #f5f6f8;
		--card: #ffffff;
		--card2: #fafbfc;
		--text: #1a1d21;
		--muted: #6b7280;
		--border: #e5e7eb;
		--accent: #4f46e5;
		--accent-soft: #eef2ff;
		--ok: #16a34a;
		--ok-bg: #f0fdf4;
		--warn: #b45309;
		--warn-bg: #fffbeb;
		--err: #dc2626;
		--err-bg: #fef2f2;
		--code-bg: #f3f4f6;
		--rowhover: #f9fafb;
	}
	@media (prefers-color-scheme: dark) {
		:root {
			--bg: #0f1115;
			--card: #171a21;
			--card2: #1c2029;
			--text: #e5e7eb;
			--muted: #9ca3af;
			--border: #272c37;
			--accent: #818cf8;
			--accent-soft: #1e1b4b;
			--ok: #4ade80;
			--ok-bg: #052e16;
			--warn: #fbbf24;
			--warn-bg: #451a03;
			--err: #f87171;
			--err-bg: #450a0a;
			--code-bg: #0d0f13;
			--rowhover: #1c2029;
		}
	}

	* { box-sizing: border-box; }
	:global(body) {
		background: var(--bg);
		color: var(--text);
		font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
		margin: 0;
	}
	main, header {
		max-width: 760px;
		margin: 0 auto;
		padding: 0 1rem;
	}

	/* ── header ─────────────────────────────────────────────── */
	header {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
		gap: 1rem;
		margin-top: 1.5rem;
		flex-wrap: wrap;
	}
	h1 { font-size: 1.5rem; margin: 0; letter-spacing: -0.02em; }
	.sub { color: var(--muted); margin: 0.2rem 0 0; font-size: 0.92rem; }
	.pills { display: flex; gap: 0.4rem; flex-wrap: wrap; }
	.pill {
		font-size: 0.78rem;
		font-weight: 600;
		padding: 0.3rem 0.7rem;
		border-radius: 999px;
		border: 1px solid var(--border);
		background: var(--card);
		white-space: nowrap;
	}
	.pill.backend { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
	.pill.ready { background: var(--ok-bg); border-color: var(--ok); color: var(--ok); }
	.pill.busy { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
	.pill.error { background: var(--err-bg); border-color: var(--err); color: var(--err); }
	.pill.idle { color: var(--muted); }

	/* ── cards & sections ─────────────────────────────────────── */
	.card {
		background: var(--card);
		border: 1px solid var(--border);
		border-radius: 14px;
		padding: 1.1rem 1.25rem 1.25rem;
		margin: 1rem 0;
		box-shadow: 0 1px 2px rgb(0 0 0 / 4%);
	}
	h2 {
		font-size: 1.05rem;
		margin: 0 0 0.75rem;
		display: flex;
		align-items: center;
		gap: 0.55rem;
		letter-spacing: -0.01em;
	}
	.step {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.5rem;
		height: 1.5rem;
		flex: none;
		border-radius: 50%;
		background: var(--accent);
		color: #fff;
		font-size: 0.85rem;
		font-weight: 700;
	}
	h3 { font-size: 0.95rem; margin: 1rem 0 0.25rem; }
	.hint { color: var(--muted); font-size: 0.9rem; }
	code {
		background: var(--code-bg);
		border-radius: 4px;
		padding: 0.1rem 0.35rem;
		font-size: 0.88em;
	}

	/* ── buttons & inputs ─────────────────────────────────────── */
	button {
		font: inherit;
		font-weight: 600;
		color: var(--text);
		padding: 0.45rem 0.9rem;
		border: 1px solid var(--border);
		border-radius: 9px;
		background: var(--card);
		cursor: pointer;
		transition: filter 120ms, background 120ms;
	}
	button:hover { filter: brightness(1.04); }
	button:disabled { opacity: 0.55; cursor: default; }
	button.primary {
		background: var(--accent);
		border-color: var(--accent);
		color: #fff;
	}
	button.run { width: 100%; padding: 0.7rem 1rem; font-size: 1.02rem; margin-top: 0.5rem; }
	select {
		font: inherit;
		width: 100%;
		padding: 0.5rem 0.8rem;
		border: 1px solid var(--border);
		border-radius: 9px;
		background: var(--card);
		color: var(--text);
	}
	progress {
		width: 100%;
		height: 0.55rem;
		margin-top: 0.4rem;
		accent-color: var(--accent);
	}
	progress.benchprogress { margin: 0.6rem 0 0; }

	.row {
		display: flex;
		gap: 0.6rem;
		align-items: center;
		margin: 0.25rem 0 0.5rem;
	}
	.row label { font-weight: 600; font-size: 0.92rem; }
	.row select { width: auto; }

	/* ── status & alerts ──────────────────────────────────────── */
	.status { margin: 0.6rem 0 0; }
	.ok { color: var(--ok); font-weight: 600; }
	.err { color: var(--err); }
	.busy { color: var(--accent); font-weight: 600; }
	.alert {
		border-radius: 10px;
		padding: 0.6rem 0.9rem;
		margin: 0.75rem 0;
		font-size: 0.92rem;
		border: 1px solid;
	}
	.alert.err { background: var(--err-bg); border-color: var(--err); color: var(--err); }
	.alert.warn { background: var(--warn-bg); border-color: var(--warn); color: var(--warn); }
	.alert pre {
		margin: 0.5rem 0 0;
		font-size: 0.78rem;
		white-space: pre-wrap;
		overflow-x: auto;
	}

	/* ── compatibility checks ─────────────────────────────────── */
	.checks {
		list-style: none;
		margin: 0.5rem 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 0.3rem 1.25rem;
		font-size: 0.92rem;
	}
	.cpufallback {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
		margin: 0.9rem 0 0.25rem;
		cursor: pointer;
		padding: 0.6rem 0.75rem;
		border: 1px dashed var(--border);
		border-radius: 10px;
	}
	.cpufallback:hover { border-color: var(--accent); }
	.cpufallback input { width: 1.15rem; height: 1.15rem; flex: none; accent-color: var(--accent); }
	.cpufallback .hint { display: block; margin-top: 0.15rem; }

	/* ── tables ───────────────────────────────────────────────── */
	.tablewrap { overflow-x: auto; margin: 0.75rem 0; }
	table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
	th {
		color: var(--muted);
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 0.45rem 0.6rem;
		border-bottom: 2px solid var(--border);
		text-align: right;
		white-space: nowrap;
	}
	td {
		padding: 0.5rem 0.6rem;
		text-align: right;
		border-bottom: 1px solid var(--border);
		white-space: nowrap;
	}
	th:first-child, td:first-child { text-align: left; }
	tbody tr:hover { background: var(--rowhover); }

	/* ── results ──────────────────────────────────────────────── */
	#results { scroll-margin-top: 1rem; }
	.stats {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 0.7rem;
		margin: 0.9rem 0;
	}
	.stat {
		border: 1px solid var(--border);
		border-radius: 12px;
		padding: 0.8rem 0.9rem;
		background: var(--card2);
		text-align: center;
	}
	.stat .num {
		font-size: 1.7rem;
		font-weight: 750;
		letter-spacing: -0.03em;
		font-variant-numeric: tabular-nums;
		color: var(--accent);
	}
	.stat .label { font-size: 0.78rem; font-weight: 600; margin-top: 0.15rem; }
	.stat .sub2 { font-size: 0.75rem; color: var(--muted); margin-top: 0.1rem; }
	button.copy { font-size: 0.82rem; padding: 0.25rem 0.7rem; }
	.json-head { display: flex; justify-content: flex-end; margin-bottom: 0.5rem; }
	pre {
		background: var(--code-bg);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 0.9rem 1rem;
		overflow-x: auto;
		white-space: pre-wrap;
		word-break: break-all;
		font-size: 0.82rem;
	}

	footer { margin: 1.5rem 0 2.5rem; }
	footer .hint { border-top: 1px solid var(--border); padding-top: 0.75rem; }

	/* ── mobile ───────────────────────────────────────────────── */
	@media (max-width: 640px) {
		header { margin-top: 0.9rem; }
		main { padding: 0 0.75rem; }
		h1 { font-size: 1.25rem; }
		.card { padding: 0.9rem; border-radius: 12px; }
		button { min-height: 44px; } /* comfortable tap target */
		.stats { gap: 0.5rem; }
		.stat .num { font-size: 1.3rem; }
		table { font-size: 0.85rem; }
		td, th { padding: 0.4rem 0.5rem; }
		pre { font-size: 0.76rem; padding: 0.6rem; }
		.hint { font-size: 0.85rem; }
	}
</style>
