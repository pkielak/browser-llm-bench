/**
 * Chrome built-in AI (Gemini Nano) — a *reference* backend.
 *
 * Chrome ships its own on-device model (Gemini Nano) and exposes it through
 * the Prompt API (`window.ai.language.model`), stable in Chrome 148+.
 * Unlike the LiteRT path it is NOT a benchmark-grade backend:
 *
 * - the model is opaque (Chrome's choice, updated with the browser — not
 *   pinnable, no version/size control);
 * - there is no output-length control (no `maxOutputTokens`) and no token
 *   counts — so tg is an **estimate** (chars/4) and pp is **not derivable**;
 * - streaming exists (`promptStreaming`), so time-to-first-chunk is a
 *   reasonable ttft *approximation* (a chunk is not a token).
 *
 * It is still worth having: it answers "what does the OS-vendored
 * on-device model do on this exact hardware?" and puts the LiteRT numbers
 * in context for free. Availability: desktop Chrome on Windows 10/11,
 * macOS 13+, Linux, Chromebook Plus — not on Android/iOS; needs 22 GB free
 * space and a GPU with >4 GB VRAM or 16 GB RAM + 4 cores. The model is
 * downloaded by Chrome on first use (progress is observable via the
 * `monitor` option).
 */

import { buildBenchPrompt } from './bench';

// Structural typings for the slice of the Prompt API we use (Chrome 148+;
// the full set is in the @types/dom-chromium-ai package).
interface AiMonitor {
	addEventListener(type: string, cb: (e: { loaded: number }) => void): void;
}
interface AiSession {
	promptStreaming(prompt: string, options?: { signal?: AbortSignal }): ReadableStream<string>;
	reset?(): Promise<void>;
	contextUsage?: number;
	contextWindow?: number;
}
interface AiLanguageModel {
	availability(options?: {
		expectedInputs: { type: string; languages: string[] }[];
		expectedOutputs: { type: string; languages: string[] }[];
	}): Promise<string>;
	create(options?: {
		monitor?: (m: AiMonitor) => void;
		signal?: AbortSignal;
	}): Promise<AiSession>;
}

let session: AiSession | null = null;

function languageModel(): AiLanguageModel | null {
	const lm = (window as unknown as { ai?: { language?: { model?: { LanguageModel?: AiLanguageModel } } } })
		?.ai?.language?.model?.LanguageModel;
	return typeof lm?.create === 'function' ? lm : null;
}

/** Cheap presence check (no async, no side effects). */
export function chromeAiPresent(): boolean {
	return languageModel() !== null;
}

export interface ChromeAiEligibility {
	/** The Prompt API exists in this browser at all. */
	present: boolean;
	/** Chrome's own verdict: 'available' | 'downloading' | 'unavailable'
	 *  (null when the API is absent). */
	availability: string | null;
}

/**
 * Ask Chrome whether its built-in model can serve text→text prompts here.
 * Cheap and side-effect-free — unlike `create()`, it does NOT start a
 * download.
 */
export async function checkChromeAiEligibility(): Promise<ChromeAiEligibility> {
	const lm = languageModel();
	if (!lm) return { present: false, availability: null };
	try {
		const availability = await lm.availability({
			expectedInputs: [{ type: 'text', languages: ['en'] }],
			expectedOutputs: [{ type: 'text', languages: ['en'] }]
		});
		return { present: true, availability };
	} catch {
		return { present: true, availability: 'unavailable' };
	}
}

export function hasChromeAiSession(): boolean {
	return session !== null;
}

export function closeChromeAiSession(): void {
	session = null;
}

/**
 * Create (or reuse) the session. The FIRST call on an origin makes Chrome
 * download its model — that can take a while, and progress is reported via
 * `onProgress` (0..1) when Chrome provides it.
 */
export async function openChromeAiSession(onProgress?: (p: number) => void): Promise<void> {
	if (session) return;
	const lm = languageModel();
	if (!lm) throw new Error('The Chrome Prompt API is not available in this browser.');

	const availability = await lm.availability({
		expectedInputs: [{ type: 'text', languages: ['en'] }],
		expectedOutputs: [{ type: 'text', languages: ['en'] }]
	});
	if (availability === 'unavailable') {
		throw new Error(
			'Chrome reports its built-in model as unavailable. Check chrome://on-device-internals — ' +
				'it needs 22 GB free space and a GPU with >4 GB VRAM (or 16 GB RAM + 4 cores).'
		);
	}
	session = await lm.create({
		monitor: (m) => {
			m.addEventListener('downloadprogress', (e) => onProgress?.(e.loaded));
		}
	});
}

export interface ChromeAiIteration {
	index: number;
	/** Time to the first streamed chunk (≈ ttft — a chunk is not a token). */
	ttftMs: number;
	/** Wall time of the full response, ms. */
	totalMs: number;
	textChars: number;
	/** Estimated generated tokens (chars / 4). */
	tokensEst: number;
	/** Estimated tok/s (tokensEst / total). */
	tgEst: number;
}

export interface ChromeAiResult {
	backend: 'chrome-builtin';
	model: string;
	params: {
		prompt_tokens: number;
		iters: number;
		response_nudge: string;
	};
	iterations: ChromeAiIteration[];
	tg_est: { best: number; median: number };
	ttft_ms: { best: number; median: number };
	estimated: true;
	notes: string;
}

// The API has no output-length control, so a response-length nudge keeps
// the iterations comparable (the prompt itself is the fixed bench prompt).
const RESPONSE_NUDGE = '\n\nSummarize the passage above in about 100 words.';

function bestMedian(values: number[]): { best: number; median: number } {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return {
		best: Math.max(...values),
		median: sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
	};
}

export async function runChromeAiBench(
	promptTokens: number,
	iters: number,
	onIteration: (it: ChromeAiIteration) => void
): Promise<ChromeAiResult> {
	if (!session) throw new Error('No Chrome built-in AI session — select the model first.');
	const prompt = buildBenchPrompt(promptTokens) + RESPONSE_NUDGE;

	const iterations: ChromeAiIteration[] = [];
	for (let i = 0; i < iters; i++) {
		// Fresh context each iteration — the session otherwise accumulates
		// history and later iterations get longer inputs.
		if (session.reset) await session.reset();

		const t0 = performance.now();
		let ttft: number | null = null;
		let text = '';
		const stream = session.promptStreaming(prompt);
		const reader = stream.getReader();
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			if (ttft === null) ttft = performance.now() - t0;
			text += value;
		}
		const total = performance.now() - t0;
		const tokensEst = Math.max(1, Math.round(text.length / 4));

		const it: ChromeAiIteration = {
			index: i + 1,
			ttftMs: Math.round(ttft ?? total),
			totalMs: Math.round(total),
			textChars: text.length,
			tokensEst,
			tgEst: tokensEst / (total / 1000)
		};
		iterations.push(it);
		onIteration(it);
	}

	return {
		backend: 'chrome-builtin',
		model: 'chrome-builtin-gemini-nano',
		params: { prompt_tokens: promptTokens, iters, response_nudge: RESPONSE_NUDGE.trim() },
		iterations,
		tg_est: bestMedian(iterations.map((i) => i.tgEst)),
		ttft_ms: bestMedian(iterations.map((i) => i.ttftMs)),
		estimated: true,
		notes:
			'Estimated metrics: the Chrome Prompt API exposes no token counts, no ' +
			'output-length control, and no prefill/decode split — tg is chars/4 over ' +
			'wall time, ttft is time-to-first-chunk. The model is Chrome\u2019s own ' +
			'Gemini Nano (version tracked by the browser, not pinnable).'
	};
}
