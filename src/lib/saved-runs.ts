// Saving + sharing of benchmark results — no backend involved:
//   * persistence: localStorage (per browser, per origin)
//   * sharing:     the result JSON base64url-encoded into the URL hash
//     (#r=…) — the page restores it on load, so a shared link is a
//     self-contained result view.
import type { BenchResult } from './bench';
import type { ChromeAiResult } from './chrome-ai';

export type AnyResult = BenchResult | ChromeAiResult;

export interface SavedRun {
	id: string;
	saved_at: number;
	result: AnyResult;
}

const KEY = 'browser-llm-bench.runs';
const MAX_RUNS = 20;

export function loadSavedRuns(): SavedRun[] {
	try {
		const raw = localStorage.getItem(KEY);
		if (!raw) return [];
		const arr = JSON.parse(raw) as SavedRun[];
		return Array.isArray(arr) ? arr.filter((r) => r && r.result && r.result.backend) : [];
	} catch {
		return [];
	}
}

function persist(runs: SavedRun[]): void {
	let list = runs;
	for (;;) {
		try {
			localStorage.setItem(KEY, JSON.stringify(list));
			return;
		} catch {
			// quota exceeded — drop the oldest and retry
			if (list.length <= 1) return;
			list = list.slice(0, -1);
		}
	}
}

export function saveRun(result: AnyResult): SavedRun[] {
	const run: SavedRun = {
		id: crypto.randomUUID(),
		saved_at: Date.now(),
		result
	};
	const runs = [run, ...loadSavedRuns()].slice(0, MAX_RUNS);
	persist(runs);
	return runs;
}

export function deleteSavedRun(id: string): SavedRun[] {
	const runs = loadSavedRuns().filter((r) => r.id !== id);
	persist(runs);
	return runs;
}

// ── URL sharing ──────────────────────────────────────────────────────────

function b64urlEncode(s: string): string {
	const bytes = new TextEncoder().encode(s);
	let bin = '';
	for (const b of bytes) bin += String.fromCharCode(b);
	return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function b64urlDecode(s: string): string {
	s = s.replace(/-/g, '+').replace(/_/g, '/');
	while (s.length % 4) s += '=';
	const bin = atob(s);
	const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
	return new TextDecoder().decode(bytes);
}

/** A shareable, self-contained URL for a result. */
export function shareUrl(result: AnyResult): string {
	return location.origin + location.pathname + '#r=' + b64urlEncode(JSON.stringify(result));
}

/** Extract a shared result from a location hash (`#r=…`), if present. */
export function resultFromHash(hash: string): AnyResult | null {
	const m = /#r=([A-Za-z0-9_-]+)/.exec(hash);
	if (!m) return null;
	try {
		const r = JSON.parse(b64urlDecode(m[1])) as AnyResult;
		return r && typeof r === 'object' && typeof r.backend === 'string' && Array.isArray(r.iterations)
			? r
			: null;
	} catch {
		return null;
	}
}
