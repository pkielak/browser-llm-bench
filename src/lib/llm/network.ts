/**
 * Network-access checks for model downloads.
 * (Borrowed from Pancratius, `src/lib/llm/network.ts`.)
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

/**
 * Check whether the browser can use the network to download files right now.
 * Never throws. An inability to read the flag is treated as "online".
 */
export async function checkNetworkAccess(): Promise<NetworkAccess> {
	try {
		if (typeof navigator !== 'undefined' && navigator.onLine === false) {
			return { online: false };
		}
	} catch {
		// Reading `navigator.onLine` failed → assume online (see module doc).
	}
	return { online: true };
}
