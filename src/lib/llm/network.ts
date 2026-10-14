/**
 * Network-access checks for model downloads.
 *
 * Downloading an on-device model file requires an active connection. This
 * module gives the manager a single, cheap way to answer "can we download
 * right now?" so the UI can show a clear, actionable message instead of a
 * bare "download failed".
 *
 * The check is deliberately best-effort and never throws: if we can't
 * determine connectivity we optimistically report `online: true` so we never
 * block a download that would otherwise have succeeded.
 */

/** Result of a network-access probe. */
export interface NetworkAccess {
	/** Whether the app currently has a usable network connection. */
	online: boolean;
}

/** The friendly, actionable message shown when there is no network access. */
export const NO_NETWORK_MESSAGE =
	'No network connection available. Make sure WiFi is turned on and you can reach ' +
	'the internet, then try downloading again.';

/** The domain the models are downloaded from — the probe target. */
const PROBE_URL = 'https://huggingface.co/robots.txt';
/** Give up on the probe after this long. */
const PROBE_TIMEOUT_MS = 5000;

/**
 * Check whether the browser can use the network to download files right now.
 * Never throws.
 *
 * `navigator.onLine` alone is not trustworthy — Chromium is known to leave
 * it stale-`false` after a connection change even though the network works —
 * so a `false` flag is only a fast-path out; otherwise a real reachability
 * probe is done against the download domain. `no-cors` means an opaque
 * response is enough: *resolving* proves the network is reachable.
 */
export async function checkNetworkAccess(): Promise<NetworkAccess> {
	try {
		if (typeof navigator !== 'undefined' && navigator.onLine === false) {
			return { online: false };
		}
	} catch {
		// Reading `navigator.onLine` failed → fall through to the probe.
	}
	try {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
		await fetch(PROBE_URL, { method: 'HEAD', mode: 'no-cors', signal: controller.signal });
		clearTimeout(timer);
		return { online: true };
	} catch {
		return { online: false };
	}
}
