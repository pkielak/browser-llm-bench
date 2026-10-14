/**
 * Local on-device inference (LiteRT-LM) — shared types.
 *
 * Borrowed from Pancratius (`src/lib/llm/types.ts`), trimmed to the web
 * target and extended with the benchmark surface (`reset` / `benchmark`).
 *
 * These types form the stable seam between the LiteRT-LM runtime (which is
 * loaded dynamically at runtime and may change under us) and the rest of the
 * app. Keeping this surface small is what gives the UI its "error boundaries":
 * everything the UI cares about is expressed here, and any runtime failure is
 * funnelled into a `LlmStatus` the UI can render.
 */

/**
 * Which engine the model will run on.
 *
 * - `webgpu` — LiteRT-LM's web module uses WebGPU when available (fast).
 * - `wasm` — transparent fallback to WASM/CPU everywhere else (slower, but
 *   works on any modern browser).
 */
export type LlmBackend = 'webgpu' | 'wasm' | 'unsupported';

/**
 * The lifecycle of the active model, from the UI's point of view.
 *
 * It is a discriminated union so the UI can render exactly one status at a
 * time without null-checks. Every failure collapses into `{ state: 'error' }`
 * with a human-readable `message` — the single error boundary the UI renders.
 */
export type LlmStatus =
	| { state: 'idle' }
	| { state: 'downloading'; progress: number } // 0..1
	| { state: 'loading' }
	| { state: 'ready' }
	| { state: 'error'; message: string };

/**
 * A selectable on-device model. `url` is the remote location of the model
 * file (the web-optimized `-web.litertlm` build); it is downloaded once and
 * then served from the browser cache (see `model-cache.ts`) so repeat loads
 * are instant.
 */
export interface LlmModel {
	id: string;
	name: string;
	/** Short human description (shown in the picker). */
	description: string;
	/** Remote URL of the model file (web build, used by the browser runtime). */
	url: string;
	/** Exact size of the file, in bytes (shown in the picker, used for progress/validation). */
	sizeBytes: number;
}

/**
 * The whole reactive state of the local inference backend, as the UI sees it.
 * Exposed as a Svelte store (see `manager.ts`).
 */
export interface LlmState {
	/** Id of the selected model (see `models.ts`), or null if none. */
	modelId: string | null;
	/** The selected model object, or null. */
	activeModel: LlmModel | null;
	/** Which engine the model will run on in this environment. */
	backend: LlmBackend;
	/** Lifecycle status of the active model. */
	status: LlmStatus;
}

/** Callbacks a caller receives while a reply is generated (streaming). */
export interface LlmGenerateCallbacks {
	/** Called with each generated text chunk as it arrives. */
	onText: (text: string) => void;
	/** Called once when generation finishes. */
	onEnd: () => void;
}

/** Per-generate options. */
export interface LlmGenerateOptions {
	/** Cap how many tokens a single generation may produce. */
	maxOutputTokens?: number;
}

/**
 * The engine's own speed counters from the most recent generation (see
 * `conversation.getBenchmarkInfo()` in the LiteRT-LM JS API). Either field
 * is `null` when the loaded runtime version does not expose them.
 */
export interface LlmBenchmarkInfo {
	prefillTokensPerSecond: number | null;
	decodeTokensPerSecond: number | null;
	/** Total tokens in the conversation context after the generation (prompt + output). */
	tokensInContext: number | null;
}

/**
 * The minimal surface of a loaded on-device model the app depends on.
 *
 * The seam between the concrete runtime implementation (the web LiteRT-LM
 * bridge — see `runtime.ts`) and the rest of the app.
 */
export interface LlmRuntime {
	/** Generate a reply to `prompt`, streaming chunks, and return the full text. */
	generate(prompt: string, callbacks: LlmGenerateCallbacks, options?: LlmGenerateOptions): Promise<string>;
	/** Drop the active conversation (fresh KV cache); the next generate() starts a new one. */
	reset(): Promise<void>;
	/** The engine's speed counters from the most recent generate() (null if unavailable). */
	benchmark(): Promise<LlmBenchmarkInfo | null>;
	/** Release the model and free its memory. */
	close(): Promise<void>;
}
