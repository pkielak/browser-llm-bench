import { get } from 'svelte/store';
import { generate, getBenchmark, llm, resetConversation } from './llm/manager';

/**
 * The benchmark harness — llama-bench convention on top of the borrowed
 * LiteRT runtime.
 *
 * This engine build does not stream per token (the whole generation arrives
 * in one batch at the end), so every metric is derived from **marginal run
 * differences**, two runs per iteration:
 *
 * - run A: 1 token     → tA ≈ prefill + 1 decode step
 * - run B: `tg` tokens → tB ≈ prefill + `tg` decode steps
 * - decode step = (tB − tA) / (tg − 1)   (the prefill cancels out)
 * - **tg**   = 1000 / decode step (tok/s)
 * - **ttft** = tA − decode step (pure prefill, ms)
 * - **pp**   = prompt tokens / prefill (tok/s)
 *
 * When the loaded runtime exposes its own counters (`getBenchmarkInfo`,
 * `getTokenCount`) they are preferred over the wall-clock derivations.
 * Each run starts from a `reset()` — fresh conversation, clean KV cache,
 * fresh engine counters. Best + median over `iters` runs.
 */

export interface BenchParams {
	/** The fixed prompt for the prefill measurement. */
	prompt: string;
	/** How many tokens each generation produces (the tg length). */
	tg: number;
	/** How many iterations to run. */
	iters: number;
}

export interface BenchIteration {
	index: number;
	/** Prefill tok/s. */
	pp: number | null;
	/** Generation tok/s. */
	tg: number;
	/** Time to first token (pure prefill), ms. */
	ttftMs: number;
	/** Wall time of the 1-token run, ms (marginal-timing method). */
	tA_ms: number;
	/** Wall time of the tg-token run, ms (marginal-timing method). */
	tB_ms: number;
	/** Streamed text chunks observed in the tg run (≈ generated tokens). */
	tokens: number;
}

/** The params recorded in a result (the prompt itself is not stored). */
export interface BenchResultParams {
	tg: number;
	iters: number;
	prompt_chars: number;
}

export interface BenchResult {
	backend: 'webgpu' | 'wasm';
	model: string;
	params: BenchResultParams;
	iterations: BenchIteration[];
	pp: { best: number | null; median: number | null };
	tg: { best: number; median: number };
	ttft_ms: number;
	date: string;
}

function median(values: number[]): number | null {
	if (values.length === 0) return null;
	const s = [...values].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Run the benchmark against the currently selected (ready) model.
 * `onIteration` (when given) is called after each iteration with its result.
 */
export async function runBench(
	params: BenchParams,
	onIteration?: (iteration: BenchIteration) => void
): Promise<BenchResult> {
	const state = get(llm);
	const model = state.activeModel;
	if (!model) throw new Error('No on-device model is selected.');
	if (state.status.state !== 'ready') {
		throw new Error('The model is not ready yet. Load it first.');
	}

	const iterations: BenchIteration[] = [];
	for (let i = 1; i <= params.iters; i++) {
		// Two marginal runs (see the module doc): 1 token, then `tg` tokens,
		// each from a fresh conversation (clean KV cache + counters).
		let tokens = 0;

		await resetConversation();
		const tA0 = performance.now();
		await generate(params.prompt, { onText: () => {}, onEnd: () => {} }, { maxOutputTokens: 1 });
		const tA = performance.now() - tA0;

		await resetConversation();
		const tB0 = performance.now();
		await generate(
			params.prompt,
			{
				onText: () => {
					tokens++;
				},
				onEnd: () => {}
			},
			{ maxOutputTokens: params.tg }
		);
		const tB = performance.now() - tB0;
		if (tokens === 0) throw new Error('The model produced no tokens.');

		const bench = await getBenchmark();

		// Marginal decode step (ms/token) and pure prefill time (ms).
		const perTokenMs = Math.max(0.001, (tB - tA) / (params.tg - 1));
		const prefillMs = Math.max(0.001, tA - perTokenMs);
		const wallTg = 1000 / perTokenMs;

		// Prompt length in tokens: the context after the generation minus the
		// generated tokens. When the engine does not expose a usable count
		// (absent or 0), estimate from the character count (~4 chars/token for
		// Gemma on English).
		const ctx = bench?.tokensInContext;
		const promptTokens =
			ctx != null && Number.isFinite(ctx) && ctx > 0
				? Math.max(1, ctx - params.tg)
				: Math.max(1, Math.round(params.prompt.length / 4));
		const wallPp = promptTokens / (prefillMs / 1000);

		// The engine counters exist in this build but read 0 (unpopulated) —
		// only a positive, finite engine value beats the wall-clock derivation.
		const pick = (engine: number | null | undefined, wall: number): number =>
			engine != null && Number.isFinite(engine) && engine > 0 ? engine : wall;

		const iteration: BenchIteration = {
			index: i,
			pp: pick(bench?.prefillTokensPerSecond, wallPp),
			tg: pick(bench?.decodeTokensPerSecond, wallTg),
			ttftMs: Math.round(prefillMs),
			tA_ms: Math.round(tA),
			tB_ms: Math.round(tB),
			tokens
		};
		iterations.push(iteration);
		onIteration?.(iteration);
	}

	const ppValues = iterations.map((it) => it.pp).filter((v): v is number => v !== null);
	return {
		backend: state.backend === 'wasm' ? 'wasm' : 'webgpu',
		model: model.name,
		params: { tg: params.tg, iters: params.iters, prompt_chars: params.prompt.length },
		iterations,
		pp: { best: ppValues.length ? Math.max(...ppValues) : null, median: median(ppValues) },
		tg: { best: Math.max(...iterations.map((it) => it.tg)), median: median(iterations.map((it) => it.tg)) ?? 0 },
		ttft_ms: Math.round(median(iterations.map((it) => it.ttftMs)) ?? 0),
		date: new Date().toISOString()
	};
}
