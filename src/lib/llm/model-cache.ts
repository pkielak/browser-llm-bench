import type { LlmModel } from './types';
import { NO_NETWORK_MESSAGE } from './network';

/**
 * Model file caching.
 *
 * (Borrowed from Pancratius, `src/lib/llm/model-cache.ts` — same mechanism,
 * own cache name.)
 *
 * On-device model files are large (hundreds of MB to GB). We download each
 * model file **once** and store it in the browser's Cache Storage
 * (`caches`), which persists across sessions. Subsequent loads are served
 * straight from the cache — no re-download.
 *
 * Cache Storage is the right tool here because it is:
 * - persistent (survives app restarts / days),
 * - keyed by URL (so each model has a stable cache entry), and
 * - available in any secure-context browser.
 *
 * If Cache Storage is unavailable (very old browser, non-secure context), we
 * degrade gracefully to in-memory blobs: the model still works, it just gets
 * re-downloaded next time.
 */
export const MODEL_CACHE_NAME = 'llm-bench-models-v1';

/** Progress callback: called with a 0..1 fraction as the file downloads. */
export type DownloadProgress = (fraction: number) => void;

function cacheAvailable(): boolean {
	return typeof caches !== 'undefined' && caches != null;
}

/** Read a previously downloaded model from Cache Storage, if present. */
export async function readCachedModel(url: string): Promise<Blob | null> {
	if (!cacheAvailable()) return null;
	try {
		const cache = await caches.open(MODEL_CACHE_NAME);
		const hit = await cache.match(url);
		if (hit) {
			const buffer = await hit.arrayBuffer();
			return new Blob([buffer], { type: 'application/octet-stream' });
		}
	} catch {
		// Treat a cache read failure as a miss and download fresh below.
	}
	return null;
}

/**
 * Get a model file as a Blob, downloading it (with progress) and storing it
 * in Cache Storage on first use. Cached models are returned immediately.
 */
export async function downloadModel(model: LlmModel, onProgress?: DownloadProgress): Promise<Blob> {
	const cached = await readCachedModel(model.url);
	if (cached) {
		onProgress?.(1);
		return cached;
	}

	let response: Response;
	try {
		response = await fetch(model.url, { mode: 'cors' });
	} catch (err) {
		// A rejected fetch (network unreachable / offline) is a connectivity
		// problem, not a server error — say so clearly.
		throw new Error(NO_NETWORK_MESSAGE, { cause: err });
	}
	if (!response.ok) {
		throw new Error(`Downloading the model failed (HTTP ${response.status}). Check your connection and try again.`);
	}

	const total = Number(response.headers.get('content-length') ?? 0);

	const reader = response.body?.getReader();
	if (!reader) {
		// No readable stream (rare); just buffer the whole body.
		const blob = new Blob([await response.arrayBuffer()], { type: 'application/octet-stream' });
		onProgress?.(1);
		return storeBlob(model, blob);
	}

	const chunks: BlobPart[] = [];
	let received = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		if (value) {
			chunks.push(value);
			received += value.length;
			if (total > 0) onProgress?.(received / total);
		}
	}
	onProgress?.(1);
	return storeBlob(model, new Blob(chunks, { type: 'application/octet-stream' }));
}

/** Best-effort store of a model blob in Cache Storage (never throws). */
async function storeBlob(model: LlmModel, blob: Blob): Promise<Blob> {
	if (!cacheAvailable()) return blob;
	try {
		const cache = await caches.open(MODEL_CACHE_NAME);
		await cache.put(model.url, new Response(blob));
	} catch {
		// Cache unavailable/full: keep the in-memory blob, lose persistence.
	}
	return blob;
}
