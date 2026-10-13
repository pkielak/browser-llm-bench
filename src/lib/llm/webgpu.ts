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
