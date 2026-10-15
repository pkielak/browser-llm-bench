import type { LlmBackend } from './types';

/**
 * WebGPU detection. (Borrowed from Pancratius, `src/lib/llm/webgpu.ts`.)
 *
 * We only need to know whether a WebGPU device is exposed; LiteRT-LM picks
 * the engine automatically (WebGPU if present, WASM otherwise).
 *
 * `navigator.gpu` is not (yet) in the standard TS DOM lib, so it is read
 * defensively. A `nav` argument is accepted so this can be unit-tested.
 */
export function hasWebGpu(nav: Navigator = globalThis.navigator): boolean {
	return !!(nav as unknown as { gpu?: unknown })?.gpu;
}

/** Which backend this browser will use: WebGPU if available, else WASM. */
export function detectBackend(nav: Navigator = globalThis.navigator): LlmBackend {
	return hasWebGpu(nav) ? 'webgpu' : 'wasm';
}

/**
 * Real capability check: actually request a WebGPU adapter (not just the
 * presence of `navigator.gpu`). Resolves `true` only when a device is
 * obtainable — `false` when the API is missing (e.g. a non-secure context:
 * plain `http://` origins), no adapter exists, or the request fails.
 * Fast (hundreds of ms) — run it BEFORE downloading a multi-GB model.
 */
export async function webGpuAvailable(nav: Navigator = globalThis.navigator): Promise<boolean> {
	const gpu = (nav as unknown as { gpu?: { requestAdapter?: () => Promise<unknown> } })?.gpu;
	if (!gpu?.requestAdapter) return false;
	try {
		return (await gpu.requestAdapter()) != null;
	} catch {
		return false;
	}
}
