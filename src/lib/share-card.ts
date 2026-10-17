// Renders a benchmark result as an OpenGraph-style share card
// (1200×630 PNG) entirely client-side — download it or copy it to the
// clipboard for pasting into chat.
import type { AnyResult } from './saved-runs';

const W = 1200;
const H = 630;

const C = {
	bg: '#12151c',
	bar: '#6366f1',
	title: '#9aa3b2',
	white: '#ffffff',
	muted: '#8b93a5',
	box: '#1c212c',
	boxline: '#2a3140',
	accent: '#818cf8'
};

function fitFont(
	g: CanvasRenderingContext2D,
	text: string,
	base: number,
	family: string,
	maxWidth: number
): string {
	for (let size = base; size >= 24; size -= 2) {
		const f = `700 ${size}px ${family}`;
		g.font = f;
		if (g.measureText(text).width <= maxWidth) return f;
	}
	return `700 24px ${family}`;
}

function roundedRect(
	g: CanvasRenderingContext2D,
	x: number, y: number, w: number, h: number, r: number
): void {
	g.beginPath();
	g.moveTo(x + r, y);
	g.arcTo(x + w, y, x + w, y + h, r);
	g.arcTo(x + w, y + h, x, y + h, r);
	g.arcTo(x, y + h, x, y, r);
	g.arcTo(x, y, x + w, y, r);
	g.closePath();
}

const fmt1 = (v: number): string => String(Math.round(v * 10) / 10);
const fmtMs = (v: number): string => Math.round(v).toLocaleString();

interface Stat {
	value: string;
	label: string;
}

function statsFor(r: AnyResult): Stat[] {
	if (r.backend === 'chrome-builtin') {
		return [
			{ value: fmt1(r.tg_est.best), label: 'tg · tok/s (est)' },
			{ value: fmtMs(r.ttft_ms.best), label: 'ttft · ms (first chunk)' },
			{ value: String(r.iterations.length), label: 'runs' }
		];
	}
	return [
		{ value: fmt1(r.pp.best), label: `pp${r.params.prompt_tokens} · tok/s` },
		{ value: fmt1(r.tg.best), label: 'tg · tok/s' },
		{ value: fmtMs(r.ttft_ms), label: 'ttft · ms' }
	];
}

function badgeFor(r: AnyResult): string {
	if (r.backend === 'chrome-builtin') return 'Chrome built-in AI · reference';
	if (r.backend === 'wasm') return 'WASM / CPU';
	return 'WebGPU · LiteRT';
}

export function renderShareCard(result: AnyResult): Promise<Blob> {
	const canvas = document.createElement('canvas');
	canvas.width = W;
	canvas.height = H;
	const g = canvas.getContext('2d');
	if (!g) return Promise.reject(new Error('canvas 2d unavailable'));

	const F = 'system-ui, -apple-system, "Segoe UI", sans-serif';

	// background + accent bar
	g.fillStyle = C.bg;
	g.fillRect(0, 0, W, H);
	g.fillStyle = C.bar;
	g.fillRect(0, 0, W, 8);

	// header
	g.textBaseline = 'alphabetic';
	g.fillStyle = C.title;
	g.font = `500 32px ${F}`;
	g.fillText('browser-llm-bench', 64, 88);
	g.textAlign = 'right';
	g.fillText('LLM benchmarks in the browser', W - 64, 88);
	g.textAlign = 'left';

	// model name (auto-shrink)
	g.fillStyle = C.white;
	g.font = fitFont(g, result.model, 62, F, W - 128);
	g.fillText(result.model, 64, 185);

	// backend badge
	g.font = `600 28px ${F}`;
	const badge = badgeFor(result);
	const bw = g.measureText(badge).width + 48;
	g.fillStyle = C.bar;
	roundedRect(g, 64, 215, bw, 52, 26);
	g.fill();
	g.fillStyle = C.white;
	g.fillText(badge, 88, 250);

	// stat boxes
	const stats = statsFor(result);
	const gap = 32;
	const bw2 = (W - 128 - gap * (stats.length - 1)) / stats.length;
	const by = 315;
	const bh = 190;
	stats.forEach((s, i) => {
		const x = 64 + i * (bw2 + gap);
		g.fillStyle = C.box;
		g.strokeStyle = C.boxline;
		g.lineWidth = 2;
		roundedRect(g, x, by, bw2, bh, 16);
		g.fill();
		g.stroke();
		g.fillStyle = C.white;
		g.font = fitFont(g, s.value, 76, F, bw2 - 40);
		g.fillText(s.value, x + 24, by + 100);
		g.fillStyle = C.muted;
		g.font = `500 26px ${F}`;
		g.fillText(s.label, x + 24, by + 150);
	});

	// footer: date + stop note
	g.fillStyle = C.muted;
	g.font = `400 26px ${F}`;
	const date = new Date().toLocaleDateString();
	const note =
		result.backend !== 'chrome-builtin' && result.stopped
			? ` · stopped early: ${result.stopped}`
			: '';
	g.fillText(`run of ${date}${note}`, 64, H - 44);
	g.textAlign = 'right';
	g.fillText(result.backend, W - 64, H - 44);
	g.textAlign = 'left';

	return new Promise((resolve, reject) => {
		canvas.toBlob(
			(b) => (b ? resolve(b) : reject(new Error('PNG encode failed'))),
			'image/png'
		);
	});
}
