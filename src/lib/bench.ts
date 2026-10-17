import { get } from 'svelte/store';
import { generate, getBenchmark, getEngineLog, llm, resetConversation } from './llm/manager';

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
 * The **wall-clock derivations are the primary numbers**. The engine's own
 * counters (`getBenchmarkInfo`, `getTokenCount`) are recorded as separate
 * reference fields (`pp_engine`, `tg_engine`, `tokens_in_context`): in the
 * current build the prefill counter is populated but self-inconsistent
 * (it measures the engine's internal tokenization, not the prompt), so it
 * must not silently replace the wall-clock value.
 *
 * Each run starts from a `reset()` — fresh conversation, clean KV cache,
 * fresh engine counters. Best + median over `iters` runs.
 */

export interface BenchParams {
	/** Target prompt length in tokens — llama-bench's pp512 convention.
	 *  The engine API takes text (not token IDs), so the prompt is
	 *  synthesized to approximately this length. */
	promptTokens: number;
	/** How many tokens each generation produces (the tg length). */
	tg: number;
	/** How many iterations to run. */
	iters: number;
}

// The prompt is a natural (non-repetitive-within-a-cycle) English passage,
// repeated to the target length. llama-bench itself uses random vocabulary
// tokens — exact counts, immune to model behavior; we can't inject token
// IDs through the LiteRT text API, so ~4 chars/token is the working
// approximation (the engine's context counter, when available, is the
// exact count and what pp is computed from).
const BASE_TEXT =
	'The sun rose over the quiet harbor as the first fishing boats headed out to sea. ' +
	'Seagulls circled above the masts, their cries carried by the morning wind along the coast. ' +
	'Fishermen checked their nets and engines before setting off, knowing that the day\'s catch ' +
	'would depend on weather and skill as much as luck. By midmorning the harbor was empty ' +
	'except for a few small pleasure boats drifting lazily in the calm water, while the lighthouse ' +
	'keeper climbed his tower to log the first entries of a new day\'s weather book.';

/** Synthesize a prompt of approximately `targetTokens` tokens. */
export function buildBenchPrompt(targetTokens: number): string {
	const targetChars = targetTokens * 4;
	let out = BASE_TEXT;
	while (out.length < targetChars) out += ' ' + BASE_TEXT;
	return out;
}

export interface BenchIteration {
	index: number;
	/** Prefill tok/s (wall-clock: prompt tokens / prefill time). */
	pp: number;
	/** Generation tok/s (wall-clock: marginal decode step). */
	tg: number;
	/** Time to first token (pure prefill), ms. */
	ttftMs: number;
	/** Wall time of the 1-token run, ms (marginal-timing method). */
	tA_ms: number;
	/** Wall time of the tg-token run, ms (marginal-timing method). */
	tB_ms: number;
	/** Streamed text chunks observed in the tg run (≈ generated tokens). */
	tokens: number;
	/** The engine's own prefill counter, tok/s (reference; null when unpopulated). */
	pp_engine: number | null;
	/** The engine's own decode counter, tok/s (reference; null when unpopulated). */
	tg_engine: number | null;
	/** Tokens in the conversation context after the tg run (prompt + output). */
	tokens_in_context: number | null;
}

/** A positive, finite engine counter value, else null (0 means unpopulated). */
function positive(v: number | null | undefined): number | null {
	return v != null && Number.isFinite(v) && v > 0 ? v : null;
}

/** The params recorded in a result (the prompt itself is not stored). */
export interface BenchResultParams {
	tg: number;
	iters: number;
	/** Target prompt length in tokens (llama-bench-style, e.g. 512 = pp512). */
	prompt_tokens: number;
	prompt_chars: number;
}

export interface BenchResult {
	backend: 'webgpu' | 'wasm';
	model: string;
	params: BenchResultParams;
	iterations: BenchIteration[];
	pp: { best: number; median: number };
	tg: { best: number; median: number };
	ttft_ms: number;
	date: string;
	/** Set when the bench stopped before `iters` (e.g. the engine exhausted
	 *  the WASM heap on a later iteration). The completed iterations are
	 *  still valid and are what the aggregates are computed from. */
	stopped?: string;
	/** The engine's console output at the moment of stopping. */
	engine_log?: string[];
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

	// Synthesize the prompt to the target length, then apply the chat
	// template: a raw prompt makes instruct models (e.g. Qwen3) emit an
	// immediate EOS → zero text tokens, and the engine does not apply
	// templates itself.
	const rawPrompt = buildBenchPrompt(params.promptTokens);
	const prompt = model.promptTemplate
		? model.promptTemplate.split('{prompt}').join(rawPrompt)
		: rawPrompt;

	const iterations: BenchIteration[] = [];
	let stopped: string | undefined;
	for (let i = 1; i <= params.iters; i++) {
		try {
		{
		// Two marginal runs (see the module doc): a short run, then `tg`
		// tokens, each from a fresh conversation (clean KV cache + counters).
		// The short run asks for 1 token; some engine/model combos treat a
		// cap of 1 as "generate nothing" (empty stream in a few ms), so it
		// falls back to 2 — the marginal formula adapts to `capA`.
		let tokens = 0;
		let capA = 1;
		let tokensA = 0;
		let tA = 0;
		await resetConversation();
		{
			const t0 = performance.now();
			await generate(
				prompt,
				{ onText: () => { tokensA++; }, onEnd: () => {} },
				{ maxOutputTokens: 1 }
			);
			tA = performance.now() - t0;
		}
		if (tokensA === 0) {
			await resetConversation();
			const t0 = performance.now();
			await generate(prompt, { onText: () => {}, onEnd: () => {} }, { maxOutputTokens: 2 });
			tA = performance.now() - t0;
			capA = 2;
		}

		await resetConversation();
		const tB0 = performance.now();
		await generate(
			prompt,
			{
				onText: () => {
					tokens++;
				},
				onEnd: () => {}
			},
			{ maxOutputTokens: params.tg }
		);
		const tB = performance.now() - tB0;
		if (tokens === 0) {
			const cause =
				state.backend === 'webgpu'
					? 'This usually means WebGPU is broken on this device (e.g. Android '
						+ 'with flag-forced “unsafe WebGPU”, or a Mali/Adreno Vulkan driver). '
						+ 'Check the compatibility section (compute verification), chrome://gpu, '
						+ 'or try desktop Chrome.'
					: 'You are on the WASM/CPU backend, so this is NOT a WebGPU problem — the '
						+ 'engine itself returned nothing. It can be memory pressure (the CPU '
						+ 'path needs ~3× the file size in RAM) — try the tiny model.';
			const log = getEngineLog().slice(-4).join(' | ');
			throw new Error(
				'The model produced no tokens — the engine ran, but the generation '
				+ `returned nothing. ${cause}` +
				(log ? ` Engine console: ${log}` : ' Check the browser console (worker) for engine errors.')
			);
		}

		const bench = await getBenchmark();

		// Marginal decode step (ms/token) and pure prefill time (ms) —
		// `capA` tokens were generated in run A (1, or 2 on the fallback).
		const perTokenMs = Math.max(0.001, (tB - tA) / (params.tg - capA));
		const prefillMs = Math.max(0.001, tA - capA * perTokenMs);
		const wallTg = 1000 / perTokenMs;

		// Prompt length in tokens: the context after the generation minus the
		// generated tokens. When the engine does not expose a usable count
		// (absent or 0), estimate from the character count (~4 chars/token for
		// Gemma on English).
		const ctx = bench?.tokensInContext;
		const promptTokens =
			ctx != null && Number.isFinite(ctx) && ctx > 0
				? Math.max(1, ctx - params.tg)
				: Math.max(1, Math.round(prompt.length / 4));
		const wallPp = promptTokens / (prefillMs / 1000);

		const iteration: BenchIteration = {
			index: i,
			pp: wallPp,
			tg: wallTg,
			ttftMs: Math.round(prefillMs),
			tA_ms: Math.round(tA),
			tB_ms: Math.round(tB),
			tokens,
			pp_engine: positive(bench?.prefillTokensPerSecond),
			tg_engine: positive(bench?.decodeTokensPerSecond),
			tokens_in_context: positive(bench?.tokensInContext)
		};
			iterations.push(iteration);
			onIteration?.(iteration);
		}
		} catch (err) {
			// A later iteration failing (typically the engine exhausting the
			// WASM heap — per-conversation memory is not released on reset)
			// does not invalidate the completed iterations: report them.
			if (iterations.length === 0) throw err;
			stopped = `${err instanceof Error ? err.message : String(err)} — iteration ${i} of ${params.iters}`;
			break;
		}
	}

	return {
		backend: state.backend === 'wasm' ? 'wasm' : 'webgpu',
		model: model.name,
		params: {
			tg: params.tg,
			iters: params.iters,
			prompt_tokens: Math.round(prompt.length / 4),
			prompt_chars: prompt.length
		},
		iterations,
		pp: { best: Math.max(...iterations.map((it) => it.pp)), median: median(iterations.map((it) => it.pp)) ?? 0 },
		tg: { best: Math.max(...iterations.map((it) => it.tg)), median: median(iterations.map((it) => it.tg)) ?? 0 },
		ttft_ms: Math.round(median(iterations.map((it) => it.ttftMs)) ?? 0),
		date: new Date().toISOString(),
		...(stopped
			? { stopped, engine_log: getEngineLog().slice(-6) }
			: {})
	};
}
