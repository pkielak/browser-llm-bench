import { get } from 'svelte/store';
import { generate, getBenchmark, llm, resetConversation } from './llm/manager';

/**
 * The benchmark harness — llama-bench convention on top of the borrowed
 * LiteRT runtime:
 *
 * - **pp** (prefill) — tok/s over a fixed prompt, from the engine's own
 *   counter (`getBenchmarkInfo().lastPrefillTokensPerSecond`) when the loaded
 *   runtime exposes it.
 * - **tg** (token generation) — tok/s over a fixed `tg`-token generation;
 *   engine counter when available, wall-clock over streamed chunks otherwise.
 * - **ttft** — ms from submit to the first streamed token.
 *
 * Each iteration starts from a `reset()` — fresh conversation, clean KV
 * cache, fresh engine counters. Best + median over `iters` runs.
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
	/** Prefill tok/s (null when the engine exposes no counter). */
	pp: number | null;
	/** Generation tok/s. */
	tg: number;
	/** Time to first token, ms. */
	ttftMs: number;
	/** Streamed text chunks observed (≈ generated tokens). */
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
		// Fresh conversation each iteration: clean KV cache + counters.
		await resetConversation();

		let first = 0;
		let tokens = 0;
		const t0 = performance.now();
		await generate(
			params.prompt,
			{
				onText: () => {
					tokens++;
					if (!first) first = performance.now();
				},
				onEnd: () => {}
			},
			{ maxOutputTokens: params.tg }
		);
		const tEnd = performance.now();
		const bench = await getBenchmark();
		if (!first) throw new Error('The model produced no tokens.');

		// Wall-clock measurements. Streaming chunks are not 1:1 with tokens
		// (the engine batches them), so speed is computed from the *requested*
		// generation length over the wall time, not from the chunk count.
		const genMs = tEnd - first;
		const wallTg = genMs > 0 ? params.tg / (genMs / 1000) : 0;

		// Prompt length in tokens: the context after the generation minus the
		// generated tokens. When the engine does not expose the count, estimate
		// from the character count (~4 chars/token for Gemma on English) —
		// better a labelled-ish estimate than no number at all.
		let wallPp: number | null = null;
		const ttftMs = first - t0;
		if (ttftMs > 0) {
			const promptTokens =
				bench?.tokensInContext != null
					? Math.max(1, bench.tokensInContext - params.tg)
					: Math.max(1, Math.round(params.prompt.length / 4));
			wallPp = promptTokens / (ttftMs / 1000);
		}

		const iteration: BenchIteration = {
			index: i,
			pp: bench?.prefillTokensPerSecond ?? wallPp,
			tg: bench?.decodeTokensPerSecond ?? wallTg,
			ttftMs: Math.round(first - t0),
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
