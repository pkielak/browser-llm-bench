import { get, writable } from 'svelte/store';
import { defaultModelId, getModel } from './models';
import { detectBackend } from './webgpu';
import { probeWebGpu, summarizeProbe } from './webgpu-probe';
import { downloadModel } from './model-cache';
import { createWebLlmRuntime, type LlmRuntime } from './runtime';
import { checkNetworkAccess, NO_NETWORK_MESSAGE } from './network';
import type {
	LlmBenchmarkInfo,
	LlmGenerateCallbacks,
	LlmGenerateOptions,
	LlmModel,
	LlmState
} from './types';

/**
 * The local inference backend, exposed to the UI as a single reactive store.
 *
 * Borrowed from Pancratius (`src/lib/llm/manager.ts`), trimmed to the web
 * target (no Capacitor path, no background notifications, no i18n).
 *
 * - **Model selection** lives here and is persisted in localStorage.
 * - **Model caching**: selecting a model downloads it once (via
 *   `downloadModel`) and caches the file in Cache Storage for future loads.
 * - **Error boundaries**: the whole load/generate lifecycle is a small state
 *   machine; *every* failure is funnelled into
 *   `{ status: { state: 'error', message } }` — a friendly, recoverable UI
 *   state (never a crash / unhandled rejection).
 * - **WebGPU + non-WebGPU**: `detectBackend()` reports the engine; LiteRT
 *   runs on WebGPU when present and transparently falls back to WASM.
 */

const SELECTED_MODEL_KEY = 'llm-bench.model';

function initialModelId(): string {
	try {
		const saved = localStorage.getItem(SELECTED_MODEL_KEY);
		if (saved && getModel(saved)) return saved;
	} catch {
		// ignore
	}
	return defaultModelId();
}

function initialBackend(): LlmState['backend'] {
	try {
		return detectBackend();
	} catch {
		return 'wasm';
	}
}

const initialModelIdValue = initialModelId();

/** The single reactive source of truth for the local inference backend. */
export const llm = writable<LlmState>({
	modelId: initialModelIdValue || null,
	activeModel: getModel(initialModelIdValue) ?? null,
	backend: initialBackend(),
	status: { state: 'idle' }
});

// --- internal, non-reactive handles (module singleton) ---
let runtime: LlmRuntime | null = null;
/** Bumped on every load attempt; lets a newer load cancel an older one. */
let loadingToken = 0;
/** When true, a failed WebGPU probe falls back to the WASM/CPU engine
 *  instead of refusing (explicit user opt-in — the UI checkbox). */
let allowCpuFallback = false;

/** Opt in to the WASM/CPU fallback when WebGPU is unavailable/broken. */
export function setAllowCpuFallback(v: boolean): void {
	allowCpuFallback = v;
}

function setState(patch: Partial<LlmState>): void {
	llm.update((s) => ({ ...s, ...patch }));
}

function setStatus(status: LlmState['status']): void {
	setState({ status });
}

function humanizeError(err: unknown): string {
	if (err instanceof Error && err.message) return err.message;
	return String(err);
}

/**
 * Select a model (persisting the choice) and load + cache it. Loading is
 * fire-and-forget; progress and any error are observable via the `llm` store.
 */
export function selectModel(id: string): void {
	const model = getModel(id);
	if (!model) return;
	setState({ modelId: model.id, activeModel: model });
	try {
		localStorage.setItem(SELECTED_MODEL_KEY, model.id);
	} catch {
		// ignore
	}
	void loadModel(model);
}

/**
 * Load a model and initialize its runtime → ready. Cancels any in-flight load
 * and tears down the previous model. Never throws; failures land in the store
 * as an error.
 *
 * Web path: `downloadModel()` (fetch with progress → Cache Storage) →
 * `createWebLlmRuntime({ model, blob })`.
 */
async function loadModel(model: LlmModel): Promise<void> {
	const token = ++loadingToken;
	const isCurrent = () => token === loadingToken;

	// Tear down any previously loaded model (frees memory).
	if (runtime) {
		const old = runtime;
		runtime = null;
		await old.close();
	}

	try {
		if (!isCurrent()) return;

		// Full WebGPU capability check BEFORE the download: secure context →
		// API → adapter → device → a minimal compute shader. The last one is
		// decisive — adapters can exist (even with device creation) while
		// compute is broken (flag-forced Android WebGPU, weak Vulkan
		// drivers), in which case the engine loads and then produces zero
		// tokens. This is a WebGPU benchmark: a multi-GB download that lands
		// on a broken/CPU path is wasted bandwidth, so fail fast.
		const probe = await probeWebGpu();
		if (probe.verdict !== 'ok') {
			if (!allowCpuFallback) {
				setStatus({
					state: 'error',
					message:
						summarizeProbe(probe) +
						' Nothing was downloaded. (Tick “Allow WASM/CPU fallback” to run on the CPU instead.)'
			});
				return;
			}
			// Explicit opt-in: proceed on the WASM/CPU engine.
			setState({ backend: 'wasm' });
		} else {
			setState({ backend: 'webgpu' });
		}

		// Downloading a model needs network access — check up front so a
		// missing connection surfaces a clear, actionable message.
		const { online } = await checkNetworkAccess();
		if (!online) {
			setStatus({ state: 'error', message: NO_NETWORK_MESSAGE });
			return;
		}

		setStatus({ state: 'downloading', progress: 0 });
		const blob = await downloadModel(model, (progress) => {
			if (!isCurrent()) return;
			setStatus({ state: 'downloading', progress });
		});
		if (!isCurrent()) return;

		setStatus({ state: 'loading' });
		const next = await createWebLlmRuntime({ model, blob, backend: get(llm).backend });
		if (!isCurrent()) {
			await next.close();
			return;
		}

		runtime = next;
		setStatus({ state: 'ready' });
	} catch (err) {
		if (!isCurrent()) return;
		setStatus({ state: 'error', message: humanizeError(err) });
	}
}

/**
 * Generate a reply, streaming chunks via `callbacks.onText`. Ensures the
 * selected model is loaded (triggering a download/cache + init if needed)
 * before generating. Throws a friendly `Error` if the model cannot be made
 * ready or generation fails — the caller is the error boundary.
 */
export async function generate(
	prompt: string,
	callbacks: LlmGenerateCallbacks,
	options?: LlmGenerateOptions
): Promise<string> {
	const state = get(llm);
	const model = state.activeModel ?? getModel(state.modelId ?? undefined);
	if (!model) throw new Error('No on-device model is selected.');

	if (state.status.state !== 'ready' || !runtime) {
		await loadModel(model);
		const after = get(llm);
		if (after.status.state !== 'ready' || !runtime) {
			throw new Error(
				after.status.state === 'error'
					? after.status.message
					: 'The model is not ready yet. Please try again.'
			);
		}
	}

	return runtime.generate(prompt, callbacks, options);
}

/** Drop the active conversation (fresh KV cache + fresh engine counters). */
export async function resetConversation(): Promise<void> {
	await runtime?.reset();
}

/** The engine's speed counters from the most recent generate(). */
export function getBenchmark(): Promise<LlmBenchmarkInfo | null> {
	return runtime ? runtime.benchmark() : Promise.resolve(null);
}

/** The engine's captured console output from the most recent generate(). */
export function getEngineLog(): string[] {
	return runtime?.getLog() ?? [];
}

/** Release the loaded model and free its memory. */
export async function releaseModel(): Promise<void> {
	loadingToken++;
	if (runtime) {
		const old = runtime;
		runtime = null;
		await old.close();
	}
	setStatus({ state: 'idle' });
}
