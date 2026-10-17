import type {
	LlmBackend,
	LlmBenchmarkInfo,
	LlmGenerateCallbacks,
	LlmGenerateOptions,
	LlmModel,
	LlmRuntime,
	LlmStatus
} from './types';

/**
 * The **web** (browser) LiteRT-LM bridge.
 *
 * Web target (no Capacitor dispatch), extended with the benchmark surface
 * (`reset` / `benchmark`).
 *
 * The LiteRT-LM web module (`@litert-lm/core`) is loaded **dynamically at
 * runtime** inside a Web Worker (see `runtime-web-worker.ts`), not bundled
 * by Vite. This keeps the static build independent of the (large, web-only)
 * runtime, and means any load failure is a runtime error the manager can
 * surface as a friendly UI message rather than a broken build.
 */

export interface CreateRuntimeOptions {
	/** The model catalogue entry. */
	model: LlmModel;
	/** The model bytes, already downloaded + cached (see `model-cache.ts`). */
	blob?: Blob;
	/** Which engine to prefer — WebGPU (GPU) when available, else CPU/WASM. */
	backend: LlmBackend;
	/**
	 * Optional status reporter so a runtime can surface fine-grained lifecycle
	 * updates while it loads. The manager passes a callback that writes into
	 * the reactive store.
	 */
	onStatus?: (status: LlmStatus) => void;
}

/** Map our high-level backend label onto the numeric backend we post to the worker. */
function toWorkerBackend(backend: LlmBackend): number {
	// 3 === CPU/WASM, 2 === GPU_ARTISAN (the streaming GPU backend). The worker
	// re-maps this onto the loaded module's `Backend` enum (see the worker).
	return backend === 'wasm' ? 3 : 2;
}

/**
 * Create the `LlmRuntime` by spawning the LiteRT-LM engine inside a Web
 * Worker and wrapping it behind the small `LlmRuntime` interface.
 *
 * Every failure is converted to a plain `Error` with a human-readable message,
 * so the manager can present it in the UI without leaking stack traces.
 */
export async function createWebLlmRuntime(options: CreateRuntimeOptions): Promise<LlmRuntime> {
	const model = options.blob;
	if (!model) {
		throw new Error('The web runtime needs a downloaded model blob — the model was not fetched.');
	}
	if (typeof Worker === 'undefined') {
		// No Web Worker support (very old browser / some web views): there is no
		// way to keep inference off the main thread, so say so clearly.
		throw new Error('This browser does not support Web Workers, which the local model needs.');
	}

	// The worker is a *classic* (non-module) worker, not a module worker.
	// The LiteRT-LM runtime loads its WASM with `importScripts()`, which only
	// exists in classic workers — a module worker throws
	// "Module scripts don't support importScripts()" and the model never
	// starts. Our own code only relies on a dynamic `import()` of an external
	// URL, which works in classic workers on every modern browser, so we lose
	// nothing by dropping `type: 'module'`.
	const worker = new Worker(new URL('./runtime-web-worker.ts', import.meta.url));

	type BenchCounters = { prefill: number | null; decode: number | null; tokensInContext: number | null };
	type OutMessage =
		| { kind: 'ready' }
		| { kind: 'closed' }
		| { kind: 'text'; id: number; text: string }
		| { kind: 'done'; id: number; console?: string[] }
		| { kind: 'bench'; bench: BenchCounters }
		| { kind: 'reset-done'; id: number }
		| { kind: 'error'; id: number | null; message: string };

	const toBenchInfo = (b: BenchCounters): LlmBenchmarkInfo => ({
		prefillTokensPerSecond: b.prefill,
		decodeTokensPerSecond: b.decode,
		tokensInContext: b.tokensInContext
	});

	interface Pending {
		resolve: () => void;
		reject: (err: Error) => void;
	}
	let nextId = 1;
	const pending = new Map<number, Pending>();
	let lifecycle: Pending | null = null;
	let closed = false;
	// The engine counters from the most recent generate(). They arrive in a
	// separate `bench` message *after* `done` (the worker fetches them
	// detached, timeout-guarded — some builds never settle them).
	let lastBench: BenchCounters | null = null;
	let benchWaiters: Array<(b: BenchCounters) => void> = [];
	// The worker's captured engine console output from the most recent
	// generate() — the only trace of a silent zero-token generation.
	let engineLog: string[] = [];
	// The `generate()` currently streaming (at most one at a time).
	let activeCallbacks: { onText: (t: string) => void } | null = null;

	function requestLifecycle(): Promise<void> {
		return new Promise<void>((resolve, reject) => {
			lifecycle = { resolve, reject };
		});
	}
	function settle(id: number, err: Error | null): void {
		const p = pending.get(id);
		if (!p) return;
		pending.delete(id);
		if (err) p.reject(err);
		else p.resolve();
	}

	worker.onmessage = (event: MessageEvent<OutMessage>) => {
		const msg = event.data;
		switch (msg.kind) {
			case 'text':
				activeCallbacks?.onText(msg.text);
				break;
			case 'done':
				engineLog = msg.console ?? [];
				settle(msg.id, null);
				break;
			case 'bench':
				lastBench = msg.bench;
				const waiters = benchWaiters;
				benchWaiters = [];
				waiters.forEach((w) => w(msg.bench));
				break;
			case 'reset-done':
				settle(msg.id, null);
				break;
			case 'error':
				if (msg.id !== null) settle(msg.id, new Error(msg.message));
				else lifecycle?.reject(new Error(msg.message));
				break;
			case 'ready':
				lifecycle?.resolve();
				break;
			case 'closed':
				lifecycle?.resolve();
				break;
		}
	};
	worker.onerror = (event) => {
		// An uncaught worker error (e.g. the module failed to load) rejects the
		// in-flight lifecycle request and any pending generation.
		const message = event.message || 'The on-device model failed to start in its worker.';
		for (const [id] of pending) settle(id, new Error(message));
		lifecycle?.reject(new Error(message));
	};

	// Load the engine in the worker. The model bytes are converted to an
	// ArrayBuffer and *transferred* (not copied) into the worker.
	const ready = requestLifecycle();
	const modelBytes = await model.arrayBuffer();
	worker.postMessage(
		{ kind: 'create', model: modelBytes, backend: toWorkerBackend(options.backend) },
		[modelBytes]
	);
	try {
		await ready;
	} catch (err) {
		worker.terminate();
		throw err instanceof Error ? err : new Error(String(err));
	}

	return {
		async generate(prompt, callbacks, options?: LlmGenerateOptions) {
			if (closed) throw new Error('The on-device model has been closed.');
			const id = nextId++;
			let full = '';
			// Wrap `onText` so we can return the concatenated reply (the worker
			// only streams chunks and never sees the whole result).
			activeCallbacks = {
				onText: (t) => {
					full += t;
					callbacks.onText(t);
				}
			};
			const done = new Promise<void>((resolve, reject) => {
				pending.set(id, { resolve, reject });
			});
			try {
				worker.postMessage({ kind: 'generate', id, prompt, maxOutputTokens: options?.maxOutputTokens });
				await done;
				return full;
			} finally {
				activeCallbacks = null;
				callbacks.onEnd();
			}
		},
		async reset() {
			if (closed) return;
			const id = nextId++;
			const done = new Promise<void>((resolve, reject) => {
				pending.set(id, { resolve, reject });
			});
			worker.postMessage({ kind: 'reset', id });
			await done;
		},
		async benchmark(): Promise<LlmBenchmarkInfo | null> {
			if (lastBench) return toBenchInfo(lastBench);
			// The counters may still be in flight (they land after `done`).
			// Give them up to 2 s; the bench falls back to wall-clock numbers.
			const b = await new Promise<BenchCounters | null>((resolve) => {
				const timer = setTimeout(() => resolve(lastBench), 2000);
				benchWaiters.push(() => {
					clearTimeout(timer);
					resolve(lastBench);
				});
			});
			return b ? toBenchInfo(b) : null;
		},
		getLog(): string[] {
			return engineLog;
		},
		async close() {
			if (closed) return;
			closed = true;
			const closedAck = requestLifecycle();
			worker.postMessage({ kind: 'close' });
			try {
				await closedAck;
			} finally {
				worker.terminate();
			}
		}
	};
}

export type { LlmGenerateCallbacks, LlmRuntime } from './types';
