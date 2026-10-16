<script lang="ts">
	/**
	 * A lag-free accordion. The native <details> element animates its
	 * content open in modern Chrome, which janks on long content blocks;
	 * this one toggles a plain {#if} — instant, no reflow animation.
	 */
	let { title, actions, children }: {
		title: string;
		actions?: () => any;
		children: () => any;
	} = $props();
	let open = $state(false);
</script>

<div class="acc">
	<div class="acc-head">
		<button
			class="acc-btn"
			aria-expanded={open}
			onclick={() => (open = !open)}
		>
			{title}
			<span class="chev">{open ? '−' : '+'}</span>
		</button>
		{@render actions?.()}
	</div>
	{#if open}
		<div class="acc-body">{@render children()}</div>
	{/if}
</div>

<style>
	.acc { margin: 0.9rem 0 0; }
	.acc-head { display: flex; align-items: center; gap: 0.5rem; }
	.acc-btn {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.6rem;
		width: 100%;
		font: inherit;
		font-weight: 600;
		color: var(--text);
		text-align: left;
		background: var(--card2);
		border: 1px solid var(--border);
		border-radius: 10px;
		padding: 0.55rem 0.9rem;
		cursor: pointer;
		transition: border-color 120ms;
	}
	.acc-btn:hover { border-color: var(--accent); }
	.chev {
		flex: none;
		font-size: 1.1rem;
		line-height: 1;
		color: var(--accent);
	}
	.acc-body {
		padding: 0.75rem 0.25rem 0.25rem;
		border-left: 2px solid var(--border);
		margin-left: 0.55rem;
		padding-left: 0.9rem;
	}
</style>
