/**
 * Web (browser) LiteRT-LM runtime — **worker** side.
 *
 * Borrowed from Pancratius (`src/lib/llm/runtime-web-worker.ts`), extended
 * for benchmarking:
 * - `generate` accepts a per-call `maxOutputTokens`,
 * - a `reset` message drops the active conversation (fresh KV cache + fresh
 *   engine speed counters),
 * - `done` carries the engine's own `getBenchmarkInfo()` counters
 *   (prefill / decode tok/s) when the loaded version exposes them.
 *
 * This file *is* the Web Worker script that `createWebLlmRuntime`
 * (in `runtime.ts`) spawns. The point of the worker: LiteRT-LM runs its
 * inference (WASM, and even the WebGPU orchestration) on whichever thread
 * created the `Engine`, so the `Engine` lives here, in a dedicated worker,
 * and the main thread only shuttles small messages.
 *
 * The worker speaks a tiny request/response protocol (see the message types
 * below). `createWebLlmRuntime` is the only other place that knows about it.
 */

// The LiteRT-LM module is loaded dynamically (not bundled). A URL `import()`
// works in a classic worker on every modern browser, so no module-worker
// features are required here.
//
// The build is **version-pinned**: the JS module and the WASM binaries it
// loads must come from the same `@litert-lm/core` version (the package ships
// a matching `wasm/` directory). The worker needs to know that directory
// explicitly — see the `Module.locateFile` setup in `handleCreate` below.
const LITERT_VERSION = '0.18.0';
const LITERT_MODULE_URL = `https://cdn.jsdelivr.net/npm/@litert-lm/core@${LITERT_VERSION}/+esm`;

// --- Shapes of the external LiteRT-LM module (kept loose on purpose) ---

interface LiteRtMessageItem {
	type: string;
	text?: string;
}
interface LiteRtStreamChunk {
	content?: LiteRtMessageItem[];
}
/** The engine's per-conversation speed counters (v0.18+). */
interface LiteRtBenchmarkInfo {
	lastPrefillTokensPerSecond?: number;
	lastDecodeTokensPerSecond?: number;
}
interface LiteRtConversation {
	sendMessageStreaming(prompt: string): AsyncIterable<LiteRtStreamChunk>;
	getBenchmarkInfo?(): Promise<LiteRtBenchmarkInfo>;
	/** Current token count of the conversation context (prompt + output). */
	getTokenCount?(): Promise<number>;
}
/**
 * The per-conversation `sessionConfig`. Only the fields we rely on are named;
 * the rest are passed through. `maxOutputTokens` caps how many tokens a single
 * generation may produce.
 */
interface LiteRtSessionConfig {
	maxOutputTokens?: number;
	[k: string]: unknown;
}
interface LiteRtConversationConfig {
	sessionConfig?: LiteRtSessionConfig;
	[k: string]: unknown;
}
interface LiteRtEngine {
	createConversation(config?: LiteRtConversationConfig): Promise<LiteRtConversation>;
	delete(): Promise<void>;
}
interface LiteRtBackend {
	CPU: number;
	GPU: number;
	GPU_ARTISAN: number;
}
interface LiteRtModule {
	Engine?: {
		create(settings: {
			model: string | Blob;
			backend?: number;
			/**
			 * `maxNumTokens` sizes the engine's context window (the KV cache),
			 * in tokens — the input *and* output combined. This is the real
			 * knob for the window; there is no `context_window` option.
			 */
			mainExecutorSettings?: { maxNumTokens?: number; [k: string]: unknown };
			[k: string]: unknown;
		}): Promise<LiteRtEngine>;
	};
	Backend?: LiteRtBackend;
	/** Exposes the CDN directory that ships this build's `wasm/` files. */
	LiteRtLm?: { DEFAULT_WASM_PATH?: string };
}

/** A message the main thread sends to the worker. */
type InMessage =
	| { kind: 'create'; model: ArrayBuffer; backend: number }
	| { kind: 'generate'; id: number; prompt: string; maxOutputTokens?: number }
	| { kind: 'reset'; id: number }
	| { kind: 'close' };

/** The engine speed counters attached to a `done` message (nulls when absent). */
export interface BenchCounters {
	prefill: number | null;
	decode: number | null;
	/** Total tokens in the conversation context after the generation (prompt + output). */
	tokensInContext: number | null;
}

function describeCause(err: unknown): string {
	if (err instanceof Error && err.message) return err.message;
	return String(err);
}

/** Map our high-level backend label onto LiteRT-LM's `Backend` enum. */
function toLiteRtBackend(backend: number, mod: LiteRtModule): number {
	const B = mod.Backend;
	// 3 === CPU/WASM, 2 === GPU_ARTISAN (the streaming GPU backend). The main
	// thread already resolved our label to a number before posting `create`.
	if (backend === 3) return B?.CPU ?? 3;
	return B?.GPU_ARTISAN ?? backend;
}

// The context window, in tokens. This is the budget for the input *and* the
// generated output **combined** (LiteRT-LM sizes the KV cache to this total),
// set on the engine via `mainExecutorSettings.maxNumTokens`.
const CONTEXT_WINDOW_TOKENS = 8192;

// The default maximum number of tokens a single generation may produce.
// Benchmark runs pass their own cap (see `handleGenerate`).
const MAX_OUTPUT_TOKENS = 2048;

let engine: LiteRtEngine | null = null;
// One conversation per engine: keeps multi-turn context alive across
// `generate()` calls until `reset` drops it or the worker is closed.
let conversation: LiteRtConversation | null = null;
let busy = false;

async function handleCreate(msg: Extract<InMessage, { kind: 'create' }>): Promise<void> {
	let mod: LiteRtModule;
	try {
		// @ts-ignore — dynamic import of a runtime URL (not a bundled module).
		mod = (await import(/* @vite-ignore */ LITERT_MODULE_URL)) as LiteRtModule;
	} catch (err) {
		throw new Error(
			`Could not load the LiteRT-LM runtime from ${LITERT_MODULE_URL}. ` +
				'Check your internet connection, or self-host the runtime and set LITERT_MODULE_URL.',
			{ cause: err }
		);
	}

	const Engine = mod.Engine;
	if (typeof Engine?.create !== 'function') {
		throw new Error('The LiteRT-LM module did not expose a usable Engine — check the module URL.');
	}

	const backend = toLiteRtBackend(msg.backend, mod);

	// --- Point the Emscripten glue at the CDN's `wasm/` directory -----------
	// The LiteRT-LM WASM glue is Emscripten output, and Emscripten locates the
	// `.wasm` binary *relative to the script that defines it*: on a page it
	// uses `document.currentScript`, but inside a Web Worker it falls back to
	// `self.location.href` — which is **our worker script's** URL, not the
	// jsdelivr URL the glue was loaded from. Left alone, `Engine.create` would
	// fetch `<our origin>/<worker dir>/litertlm_wasm_internal.wasm`, our
	// static host would answer with the index.html fallback, and
	// `WebAssembly.instantiate` would fail with
	// "expected magic word 00 61 73 6d, found 3c 21 64 6f" (`<!do…`).
	//
	// The documented override is Emscripten's `Module.locateFile` hook —
	// setting it on the worker global *before* `Engine.create` routes the
	// `.wasm` fetch to the CDN directory that ships with the loaded build.
	const wasmBase =
		(mod.LiteRtLm?.DEFAULT_WASM_PATH ??
			`https://cdn.jsdelivr.net/npm/@litert-lm/core@${LITERT_VERSION}/wasm`) + '/';
	(self as unknown as { Module?: { locateFile: (file: string) => string } }).Module = {
		locateFile: (file: string) => new URL(file, wasmBase).href
	};

	try {
		// The model bytes arrive as a transferable ArrayBuffer (the main thread
		// converted the cached Blob); wrap it in a Blob for the engine.
		const blob = new Blob([msg.model]);
		engine = await Engine.create({
			model: blob,
			backend,
			mainExecutorSettings: { maxNumTokens: CONTEXT_WINDOW_TOKENS }
		});
	} catch (err) {
		throw new Error(`The on-device model failed to start. ${describeCause(err)}`, { cause: err });
	}
}

async function handleGenerate(msg: Extract<InMessage, { kind: 'generate' }>): Promise<void> {
	if (!engine) throw new Error('The on-device model is not loaded yet.');
	// `sessionConfig.maxOutputTokens` caps a single generation (in tokens).
	// The context window itself is set on the engine at create time (see
	// `handleCreate`). Benchmark runs pass their own cap.
	conversation ??= await engine.createConversation({
		sessionConfig: { maxOutputTokens: msg.maxOutputTokens ?? MAX_OUTPUT_TOKENS }
	});
	const stream = conversation.sendMessageStreaming(msg.prompt);
	for await (const chunk of stream) {
		for (const item of chunk.content ?? []) {
			if (item.type === 'text' && item.text) {
				(self as unknown as Worker).postMessage({ kind: 'text', id: msg.id, text: item.text });
			}
		}
	}

	// The stream is over — report `done` **immediately**. The engine's own
	// speed counters (v0.18+ `getBenchmarkInfo`) and the context token count
	// are fetched *afterwards, detached and timeout-guarded*: on some builds
	// those calls never settle, and blocking `done` on them would hang the
	// whole generation (and hold the `busy` flag against the next `reset`).
	// When the counters do arrive they are sent as a separate `bench`
	// message; the bench always has its wall-clock numbers either way.
	(self as unknown as Worker).postMessage({ kind: 'done', id: msg.id });

	void (async () => {
		const bench: BenchCounters = { prefill: null, decode: null, tokensInContext: null };
		const within = (ms: number) => new Promise<undefined>((r) => setTimeout(() => r(undefined), ms));
		try {
			const info = (await Promise.race([conversation.getBenchmarkInfo?.(), within(2000)])) as
				| LiteRtBenchmarkInfo
				| undefined;
			bench.prefill = info?.lastPrefillTokensPerSecond ?? null;
			bench.decode = info?.lastDecodeTokensPerSecond ?? null;
		} catch {
			// Counters unavailable.
		}
		try {
			bench.tokensInContext =
				((await Promise.race([conversation.getTokenCount?.(), within(2000)])) as number | undefined) ?? null;
		} catch {
			// Not exposed.
		}
		if (bench.prefill != null || bench.decode != null || bench.tokensInContext != null) {
			(self as unknown as Worker).postMessage({ kind: 'bench', bench });
		}
	})();
}

/** Drop the active conversation: the next generate() starts a fresh one
 * (clean KV cache and fresh engine speed counters). */
function handleReset(): void {
	conversation = null;
}

async function handleClose(): Promise<void> {
	if (engine) {
		const old = engine;
		engine = null;
		conversation = null;
		try {
			await old.delete();
		} catch {
			// Closing is best-effort; ignore failures.
		}
	}
}

self.onmessage = async (event: MessageEvent<InMessage>) => {
	const msg = event.data;
	if (busy) return; // generation is single-flight; ignore stray messages
	busy = true;
	try {
		switch (msg.kind) {
			case 'create':
				await handleCreate(msg);
				(self as unknown as Worker).postMessage({ kind: 'ready' });
				break;
			case 'generate':
				await handleGenerate(msg);
				break;
			case 'reset':
				handleReset();
				(self as unknown as Worker).postMessage({ kind: 'reset-done', id: msg.id });
				break;
			case 'close':
				await handleClose();
				(self as unknown as Worker).postMessage({ kind: 'closed' });
				break;
		}
	} catch (err) {
		// Surface the failure back to the main thread; a `generate`/`reset`
		// failure is tagged with its request id so the matching promise rejects.
		const id = msg.kind === 'generate' || msg.kind === 'reset' ? msg.id : null;
		(self as unknown as Worker).postMessage({
			kind: 'error',
			id,
			message: err instanceof Error ? err.message : String(err)
		});
	} finally {
		busy = false;
	}
};
