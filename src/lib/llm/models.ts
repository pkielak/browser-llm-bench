import type { LlmModel } from './types';

/**
 * Catalogue of on-device models the user can pick.
 *
 * These are the web-optimized `.litertlm` builds published on Hugging
 * Face by `litert-community`. Hugging Face serves them with permissive CORS
 * headers, so a `fetch` from any origin works.
 *
 * Add entries here to offer more models — the picker, cache and backend all
 * key off this list.
 */
export const MODELS: LlmModel[] = [
	{
		id: 'gemma-4-e2b',
		name: 'Gemma 4 E2B (web)',
		description: 'Small and fast — good default. Text-only build, ~2 GB.',
		url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm',
		sizeBytes: 2_008_432_640
	},
	{
		id: 'gemma-4-e4b',
		name: 'Gemma 4 E4B (web)',
		description: 'Larger and higher quality — needs more memory; a GPU helps a lot. ~3 GB.',
		url: 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.litertlm',
		sizeBytes: 2_969_059_328
	},
	// ── small models: the WASM/CPU path needs roughly 3× the file size in
	// RAM (blob + WASM weight heap + KV cache), so a 2 GB model OOM-kills a
	// phone tab. These are the sizes where the CPU path actually works. ──
	{
		id: 'smollm2-360m',
		name: 'SmolLM2 360M (small)',
		description: 'Small — CPU (WASM) only: this file format does not load on the WebGPU path. ~0.37 GB.',
		url: 'https://huggingface.co/litert-community/SmolLM2-360M-Instruct/resolve/main/SmolLM2_360M_instruct.litertlm',
		sizeBytes: 373_719_040,
		promptTemplate: '<|begin_of_text|><|user|>\n{prompt}\n<|assistant|>\n',
		wasmOnly: true
	},
	{
		id: 'smollm2-135m',
		name: 'SmolLM2 135M (tiny)',
		description: 'Tiny — CPU (WASM) only: this file format does not load on the WebGPU path. ~0.14 GB.',
		url: 'https://huggingface.co/litert-community/SmolLM2-135M-Instruct/resolve/main/SmolLM2_135M_Instruct.litertlm',
		sizeBytes: 142_819_328,
		promptTemplate: '<|begin_of_text|><|user|>\n{prompt}\n<|assistant|>\n',
		wasmOnly: true
	},
];

/** Look up a model by id, or `undefined` if it is not in the catalogue. */
export function getModel(id: string | null | undefined): LlmModel | undefined {
	if (!id) return undefined;
	return MODELS.find((m) => m.id === id);
}

/** The model selected by default (the first in the list). */
export function defaultModelId(): string {
	return MODELS[0]?.id ?? '';
}

/** Human-readable size, e.g. "1.6 GB". */
export function formatBytes(bytes: number): string {
	if (!Number.isFinite(bytes) || bytes <= 0) return '—';
	const units = ['B', 'KB', 'MB', 'GB', 'TB'];
	let value = bytes;
	let i = 0;
	while (value >= 1024 && i < units.length - 1) {
		value /= 1024;
		i++;
	}
	const rounded = value < 10 && i > 0 ? Math.round(value * 10) / 10 : Math.round(value);
	return `${rounded} ${units[i]}`;
}
