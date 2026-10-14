import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [svelte()],
	// Relative asset paths so the build can be served from any base path.
	base: './',
	build: {
		outDir: 'build'
	}
});
