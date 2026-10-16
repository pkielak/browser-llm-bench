/**
 * A deep, live WebGPU compatibility probe — runs in a few hundred ms, before
 * any model download. (The lightweight `webgpu.ts` only checks that
 * `navigator.gpu` exists; this one actually walks the whole chain:
 * secure context → API → adapter → device → features/limits.)
 *
 * The result powers the "WebGPU compatibility" section of the UI, which
 * both gates the benchmark (no point benchmarking a WASM fallback) and
 * educates about what WebGPU is and why it matters in browsers.
 *
 * Everything is defensive: the WebGPU surface is new and varies between
 * browsers — any missing member resolves to `null`/`false`, never a throw.
 */

export type WebGpuVerdict = 'ok' | 'insecure' | 'no-api' | 'no-adapter' | 'no-device';

export interface WebGpuProbeResult {
	verdict: WebGpuVerdict;
	/** `window.isSecureContext` — WebGPU requires https:// or localhost. */
	secure: boolean;
	/** `navigator.gpu` exists. */
	apiPresent: boolean;
	/** `requestAdapter()` returned an adapter. */
	adapter: boolean;
	/** `adapter.requestDevice()` returned a device. */
	device: boolean;
	/** Human-readable adapter description, when the browser exposes it. */
	description: string | null;
	vendor: string | null;
	architecture: string | null;
	/** Largest single GPU buffer the device allows, bytes. */
	maxBufferSize: number | null;
	/** Features worth knowing for compute/ML workloads. */
	features: { name: string; supported: boolean }[];
}

interface GpuAdapterLike {
	info?: unknown;
	limits?: { maxBufferSize?: number };
	requestDevice?: () => Promise<GpuDeviceLike>;
}
interface GpuDeviceLike {
	features?: { has?: (f: string) => boolean };
}
interface GpuAdapterInfoLike {
	description?: string;
	vendor?: string;
	architecture?: string;
}

/** Features that matter for compute/ML shaders. */
const FEATURE_NAMES = [
	'shader-f16',
	'float32-filtering',
	'barycentric-interpolation',
	'timestamp-query',
	'depth-clamping'
];

/**
 * Walk the full WebGPU chain and report each step. Never throws.
 */
export async function probeWebGpu(
	nav: Navigator = globalThis.navigator,
	win: Window = globalThis as unknown as Window
): Promise<WebGpuProbeResult> {
	const base: WebGpuProbeResult = {
		verdict: 'ok',
		secure: Boolean(win.isSecureContext),
		apiPresent: false,
		adapter: false,
		device: false,
		description: null,
		vendor: null,
		architecture: null,
		maxBufferSize: null,
		features: []
	};

	if (!base.secure) {
		base.verdict = 'insecure';
		return base;
	}

	const gpu = (nav as unknown as { gpu?: { requestAdapter?: () => Promise<GpuAdapterLike | null> } })
		?.gpu;
	if (!gpu?.requestAdapter) {
		base.verdict = 'no-api';
		return base;
	}
	base.apiPresent = true;

	let adapter: GpuAdapterLike | null = null;
	try {
		adapter = await gpu.requestAdapter();
	} catch {
		adapter = null;
	}
	if (!adapter) {
		base.verdict = 'no-adapter';
		return base;
	}
	base.adapter = true;

	// `adapter.info` is newer than the rest of the API — it may be a plain
	// object, a Promise, or absent. Handle all three.
	try {
		let info = adapter.info;
		if (info && typeof (info as Promise<unknown>).then === 'function') {
			info = await info;
		}
		const i = info as GpuAdapterInfoLike | undefined;
		base.description = i?.description ?? null;
		base.vendor = i?.vendor ?? null;
		base.architecture = i?.architecture ?? null;
	} catch {
		// info unavailable — fine
	}

	base.maxBufferSize = adapter.limits?.maxBufferSize ?? null;

	let device: GpuDeviceLike | null = null;
	try {
		device = (await adapter.requestDevice?.()) ?? null;
	} catch {
		device = null;
	}
	if (!device) {
		base.verdict = 'no-device';
		return base;
	}
	base.device = true;

	base.features = FEATURE_NAMES.map((name) => ({
		name,
		supported: device?.features?.has?.(name) ?? false
	}));

	return base;
}

/** One-line human summary of a probe result. */
export function summarizeProbe(p: WebGpuProbeResult): string {
	switch (p.verdict) {
		case 'ok':
			return p.description
				? `WebGPU ready — ${p.description}`
				: 'WebGPU ready (adapter + device available)';
		case 'insecure':
			return 'Not a secure context — WebGPU requires https:// (or localhost).';
		case 'no-api':
			return 'This browser does not expose WebGPU (navigator.gpu is missing).';
		case 'no-adapter':
			return 'WebGPU exists but no GPU adapter was found (driver, VM, or remote session).';
		case 'no-device':
			return 'An adapter exists but device creation failed.';
	}
}
