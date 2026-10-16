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

export type WebGpuVerdict =
	| 'ok'
	| 'insecure'
	| 'no-api'
	| 'no-adapter'
	| 'no-device'
	| 'no-compute';

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
	/** A minimal compute shader ran and read back the expected values.
	 *  Adapters can exist (even with device creation) while compute is
	 *  broken — the classic flag-forced-WebGPU-on-Android failure. */
	compute: boolean;
	computeDetail: string | null;
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
interface GpuMappedRangeLike {
	getArrayBuffer: () => ArrayBuffer;
}
interface GpuBufferLike {
	mapAsync?: (mode: number) => Promise<GpuMappedRangeLike>;
	unmap?: () => void;
}
interface GpuPassLike {
	setPipeline: (p: unknown) => void;
	setBindGroup: (i: number, g: unknown) => void;
	dispatchWorkgroups: (n: number) => void;
	end: () => void;
}
interface GpuEncoderLike {
	beginComputePass: () => GpuPassLike;
	copyBufferToBuffer: (a: unknown, aoff: number, b: unknown, boff: number, n: number) => void;
	finish: () => unknown;
}
interface GpuDeviceLike {
	features?: { has?: (f: string) => boolean };
	createBuffer?: (o: { size: number; usage: number }) => GpuBufferLike;
	createShaderModule?: (o: { code: string }) => unknown;
	createComputePipeline?: (o: {
		layout: string;
		compute: { module: unknown; entryPoint: string };
	}) => { getBindGroupLayout: (i: number) => unknown };
	createBindGroup?: (o: { layout: unknown; entries: { binding: number; resource: { buffer: unknown } }[] }) => unknown;
	createCommandEncoder?: () => GpuEncoderLike;
	queue?: { submit: (b: unknown[]) => Promise<void> };
}

/** Storage buffer that a compute shader writes into. */
const WGSL_COMPUTE_TEST = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  out[gid.x] = gid.x * 7u + 1u;
}
`;

/**
 * Run a minimal compute shader and verify the readback. Catches the
 * "adapter and device exist but GPU compute is broken" case — typical for
 * Android WebGPU force-enabled via the unsafe flag, and some Mali/Adreno
 * Vulkan drivers. Never throws.
 */
async function verifyCompute(device: GpuDeviceLike): Promise<{ ok: boolean; detail: string | null }> {
	const N = 64;
	const bytes = N * 4;
	try {
		// GPUBufferUsage: MAP_READ=1, COPY_DST=4, STORAGE=128
		const out = device.createBuffer?.({ size: bytes, usage: 128 | 4 });
		const module = device.createShaderModule?.({ code: WGSL_COMPUTE_TEST });
		const pipe = device.createComputePipeline?.({ layout: 'auto', compute: { module, entryPoint: 'main' } });
		const bind = device.createBindGroup?.({
			layout: pipe?.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: { buffer: out } }]
		});
		const enc = device.createCommandEncoder?.();
		if (!out || !module || !pipe || !bind || !enc) return { ok: false, detail: 'incomplete WebGPU API surface' };

		const pass = enc.beginComputePass();
		pass.setPipeline(pipe);
		pass.setBindGroup(0, bind);
		pass.dispatchWorkgroups(N);
		pass.end();
		await device.queue?.submit([enc.finish()]);

		const staging = device.createBuffer?.({ size: bytes, usage: 1 | 4 });
		if (!staging) return { ok: false, detail: 'staging buffer creation failed' };
		const enc2 = device.createCommandEncoder?.();
		if (!enc2) return { ok: false, detail: 'second encoder failed' };
		enc2.copyBufferToBuffer(out, 0, staging, 0, bytes);
		await device.queue?.submit([enc2.finish()]);

		const view = await staging.mapAsync?.(1); // GPUMapMode.READ
		if (!view) return { ok: false, detail: 'buffer map failed' };
		const arr = new Uint32Array(view.getArrayBuffer());
		staging.unmap?.();
		for (let i = 0; i < N; i++) {
			if (arr[i] !== i * 7 + 1) return { ok: false, detail: 'compute readback mismatch' };
		}
		return { ok: true, detail: null };
	} catch (err) {
		return { ok: false, detail: err instanceof Error ? err.message : String(err) };
	}
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
		compute: false,
		computeDetail: null,
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

	// The decisive test: actually RUN GPU compute. Adapter + device can
	// both succeed while the compute path is broken (flag-forced Android
	// WebGPU, weak Vulkan drivers) — in that case the LLM engine loads
	// fine and then produces zero tokens.
	const compute = await verifyCompute(device);
	base.compute = compute.ok;
	base.computeDetail = compute.detail;
	if (!compute.ok) base.verdict = 'no-compute';

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
		case 'no-compute':
			return (
				'WebGPU adapter + device exist, but the compute verification failed'
				+ (p.computeDetail ? ` (${p.computeDetail})` : '')
				+ ' — the GPU driver is likely broken (common on Android with flag-forced WebGPU).'
			);
	}
}
