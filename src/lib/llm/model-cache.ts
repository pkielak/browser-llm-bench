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

/**
 * How many parallel range slices to download a big file with. One HTTP
 * connection to the HF CDN is often the bottleneck; N concurrent `Range`
 * requests use N connections (browsers allow ~6 per host, the rest queue —
 * 8 is a good balance). Peak memory is ~2x the file size either way.
 */
const RANGE_PARTS = 8;
/** Files below this size are not worth splitting. */
const RANGE_MIN_BYTES = 32 * 1024 * 1024;

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

	// First pass: learn the size + whether the server supports byte ranges
	// (HF's CDN does). The body is cancelled — the real download follows.
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
	const acceptRanges = (response.headers.get('accept-ranges') ?? '')
		.toLowerCase()
		.includes('bytes');
	await response.body?.cancel().catch(() => {});

	// Big file + range support → parallel slice download (much faster than a
	// single connection). Any failure falls back to the single stream.
	if (total >= RANGE_MIN_BYTES && acceptRanges) {
		try {
			return await downloadRanged(model, total, onProgress);
		} catch {
			// Fall through to the single-stream download below.
		}
	}

	const single = await fetch(model.url, { mode: 'cors' });
	if (!single.ok) {
		throw new Error(`Downloading the model failed (HTTP ${single.status}). Check your connection and try again.`);
	}
	return downloadStreamed(model, single, total, onProgress);
}

/** Single-connection download of an open response, with progress. */
async function downloadStreamed(
	model: LlmModel,
	response: Response,
	total: number,
	onProgress?: DownloadProgress
): Promise<Blob> {
	const reader = response.body?.getReader();
	if (!reader) {
		// No readable stream (rare); just buffer the whole body.
		const blob = new Blob([await response.arrayBuffer()], { type: 'application/octet-stream' });
		await onProgress?.(1);
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

/**
 * Download the file as `RANGE_PARTS` parallel byte-range slices and
 * assemble them. If a server ignores `Range` and answers a slice request
 * with the *whole* file (200 instead of 206), that body is used as-is.
 */
async function downloadRanged(
	model: LlmModel,
	total: number,
	onProgress?: DownloadProgress
): Promise<Blob> {
	const chunkSize = Math.ceil(total / RANGE_PARTS);
	let received = 0;
	const results = await Promise.all(
		Array.from({ length: RANGE_PARTS }, (_, i) => {
			const start = i * chunkSize;
			const end = i === RANGE_PARTS - 1 ? total - 1 : Math.min(start + chunkSize - 1, total - 1);
			return fetch(model.url, { mode: 'cors', headers: { Range: `bytes=${start}-${end}` } }).then(
				async (res) => {
					if (res.status === 200) {
						// The server ignored the Range header — this is the full file.
						const buffer = new Uint8Array(await res.arrayBuffer());
						return { buffer, full: true } as const;
					}
					if (res.status !== 206) {
						throw new Error(`HTTP ${res.status} for a byte-range request`);
					}
					// Stream the slice with live progress — a whole-slice
					// `arrayBuffer()` would report nothing until the entire
					// ~250 MB slice had landed (the bar would sit at 0%).
					const reader = res.body?.getReader();
					const chunks: Uint8Array[] = [];
					let partBytes = 0;
					if (reader) {
						for (;;) {
							const { done, value } = await reader.read();
							if (done) break;
							if (value) {
								chunks.push(value);
								partBytes += value.length;
								received += value.length;
								onProgress?.(received / total);
							}
						}
					} else {
						const buf = new Uint8Array(await res.arrayBuffer());
						chunks.push(buf);
						partBytes = buf.length;
						received += buf.length;
					}
					// Assemble this slice into one contiguous buffer.
					const buffer = new Uint8Array(partBytes);
					let offset = 0;
					for (const c of chunks) {
						buffer.set(c, offset);
						offset += c.length;
					}
					return { buffer, full: false } as const;
				}
			);
		})
	);
	const full = results.find((r) => r.full);
	onProgress?.(1);
	const blob = new Blob(
		full ? [full.buffer] : results.map((r) => r.buffer),
		{ type: 'application/octet-stream' }
	);
	return storeBlob(model, blob);
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
