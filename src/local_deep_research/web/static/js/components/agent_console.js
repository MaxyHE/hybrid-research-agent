/**
 * Evidence-first Agent Console presentation layer.
 *
 * The composer mirrors existing form fields. The live console only counts
 * events emitted by the running agent and sources returned by the canonical
 * research-resources API; it never fabricates completed calls from config.
 */
(function() {
    'use strict';

    const state = {
        researchId: null,
        counts: { collection: 0, web: 0, fetch: 0 },
        seenCalls: new Set(),
        sourcesLoaded: false
    };

    function element(id) {
        return document.getElementById(id);
    }

    function setText(id, value) {
        const target = element(id);
        if (target && value !== undefined && value !== null) {
            target.textContent = String(value);
        }
    }

    function initComposer() {
        const query = element('query');
        const packToggle = element('medical-source-pack-toggle');
        const rewriteToggle = element('agent-query-rewrite-enabled');
        const profile = element('agent_research_profile');
        if (!query || !packToggle || !rewriteToggle || !profile) return;

        const rewriteAvailable = !rewriteToggle.disabled;

        function updateQueryPreview() {
            const value = query.value.trim();
            setText('ldr-agent-console-original-query', value || 'Waiting for input');
            setText(
                'ldr-agent-console-normalized-query',
                rewriteToggle.checked
                    ? 'Resolved on Collection call'
                    : (value || 'Original query used when rewrite is off')
            );
        }

        function syncFromPack() {
            profile.value = packToggle.checked ? 'hybrid' : 'default';
            rewriteToggle.disabled = !packToggle.checked || !rewriteAvailable;
            if (!packToggle.checked) rewriteToggle.checked = false;
            if (packToggle.checked && rewriteAvailable) rewriteToggle.checked = true;
            profile.dispatchEvent(new Event('change', { bubbles: true }));
            updateQueryPreview();
        }

        function syncFromProfile() {
            const hybrid = profile.value === 'hybrid';
            if (!packToggle.disabled) packToggle.checked = hybrid;
            rewriteToggle.disabled = !hybrid || !rewriteAvailable;
            if (!hybrid) rewriteToggle.checked = false;
            updateQueryPreview();
        }

        query.addEventListener('input', updateQueryPreview);
        packToggle.addEventListener('change', syncFromPack);
        rewriteToggle.addEventListener('change', updateQueryPreview);
        profile.addEventListener('change', syncFromProfile);

        if (packToggle.checked) profile.value = 'hybrid';
        updateQueryPreview();
    }

    function classifyTool(toolName) {
        const name = String(toolName || '').toLowerCase();
        if (name === 'general_search_local_collection') return 'collection';
        if (name === 'general_search_public_web') return 'web';
        if (name === 'general_fetch_verified_candidate') return 'fetch';
        if (name.startsWith('search_collection_')) return 'collection';
        if (name === 'fetch_content') return 'fetch';
        if (name === 'web_search' || name.startsWith('search_')) return 'web';
        return null;
    }

    function renderCounts() {
        for (const role of ['collection', 'web', 'fetch']) {
            setText(`ldr-agent-console-${role}-count`, state.counts[role]);
            const row = document.querySelector(`[data-console-tool="${role}"]`);
            if (row) row.classList.toggle('ldr-is-called', state.counts[role] > 0);
        }
    }

    function renderBudget(used, limit) {
        const container = element('ldr-agent-console-live');
        const fallbackLimit = container ? Number(container.dataset.toolBudget) : 0;
        const resolvedLimit = Number(limit) || fallbackLimit;
        const resolvedUsed = Number(used) || 0;
        setText('ldr-agent-console-tool-used', resolvedUsed);
        if (resolvedLimit > 0) setText('ldr-agent-console-tool-limit', resolvedLimit);
        const fill = element('ldr-agent-console-budget-fill');
        if (fill) {
            const ratio = resolvedLimit > 0
                ? Math.min(100, (resolvedUsed / resolvedLimit) * 100)
                : 0;
            fill.style.width = `${ratio}%`;
        }
    }

    function recordEvent(data) {
        if (!data || data.phase !== 'tool_call') {
            if (data && data.query_rewrite) renderRewrite(data.query_rewrite);
            return;
        }

        const role = classifyTool(data.tool);
        if (!role) return;
        const signature = [
            data.iteration || '',
            data.tool || '',
            JSON.stringify(data.arguments || {})
        ].join('|');
        if (!state.seenCalls.has(signature)) {
            state.seenCalls.add(signature);
            state.counts[role] += 1;
            renderCounts();
        }

        const emittedUsed = Number(data.scheduled_tool_calls);
        const countedUsed = Object.values(state.counts).reduce((sum, value) => sum + value, 0);
        renderBudget(Number.isFinite(emittedUsed) ? emittedUsed : countedUsed, data.max_tool_calls);
    }

    function renderRewrite(rewrite) {
        if (!rewrite || typeof rewrite !== 'object') return;
        setText('ldr-agent-console-live-original', rewrite.original_query || 'Original query');
        if (rewrite.fallback) {
            setText('ldr-agent-console-live-normalized', 'Fallback to original query');
            return;
        }
        setText(
            'ldr-agent-console-live-normalized',
            rewrite.normalized_query || rewrite.original_query || 'Original query'
        );
    }

    function applyDetails(details) {
        if (!details || typeof details !== 'object') return;
        const metadata = details.metadata || {};
        const submission = metadata.submission || {};
        const hybrid = submission.agent_research_profile === 'hybrid';
        const general = submission.agent_research_profile === 'general';
        const generalRun = metadata.general_research || {};
        const generalSourceMode = generalRun.source_mode || 'web';
        const generalLabel = generalSourceMode === 'private_collection'
            ? 'General Research Agent V1 · Private collection evidence audit'
            : (generalSourceMode === 'web_plus_public_collection'
                ? 'General Research Agent V1 · Web + public collection evidence audit'
                : 'General Research Agent V1 · Evidence-first Web research');
        const generalEgress = generalSourceMode === 'private_collection'
            ? 'Private collection · local-only capability set'
            : (generalSourceMode === 'web_plus_public_collection'
                ? 'Capability-gated Web + public collection'
                : 'Capability-gated public Web');
        setText(
            'ldr-agent-console-profile',
            general
                ? (generalRun.publishable === false
                    ? 'General Research Agent V1 · Evidence gate incomplete'
                    : generalLabel)
                : (hybrid
                    ? 'Hybrid evidence · Medical Public Sources v1'
                    : 'Standard research profile')
        );
        if (details.query) setText('ldr-agent-console-live-original', details.query);
        if (!submission.agent_query_rewrite_enabled) {
            setText('ldr-agent-console-live-normalized', 'Rewrite off · original query');
        }
        setText(
            'ldr-agent-console-egress-scope',
            general
                ? generalEgress
                : (hybrid ? 'Collection + public Web' : 'Run snapshot')
        );
    }

    async function loadSources() {
        if (!state.researchId || state.sourcesLoaded) return;
        state.sourcesLoaded = true;
        try {
            const response = await fetch(`/research/api/resources/${encodeURIComponent(state.researchId)}`);
            if (!response.ok) return;
            const payload = await response.json();
            if (!payload || payload.status !== 'success' || !Array.isArray(payload.resources)) return;
            const resources = payload.resources;
            setText('ldr-agent-console-source-count', `${resources.length} cited`);

            let collectionCount = 0;
            let webCount = 0;
            for (const resource of resources) {
                const metadata = resource && resource.metadata ? resource.metadata : {};
                const original = metadata.original_data || {};
                const engine = String(original.source_engine || original.engine || '').toLowerCase();
                if (engine.startsWith('collection_')) collectionCount += 1;
                else webCount += 1;
            }

            const register = element('ldr-agent-console-source-register');
            if (!register) return;
            register.textContent = '';
            const groups = [
                ['Collection evidence', collectionCount],
                ['Web evidence', webCount]
            ];
            for (const [label, count] of groups) {
                if (!count) continue;
                const pill = document.createElement('span');
                pill.className = 'ldr-agent-console-source-pill';
                pill.textContent = `${label} · ${count}`;
                register.appendChild(pill);
            }
            if (!resources.length) register.textContent = 'No citation sources were saved.';
        } catch (_error) {
            // The progress page remains usable when source hydration is unavailable.
        }
    }

    function updateStatus(data) {
        if (!data || typeof data !== 'object') return;
        if (data.metadata) applyDetails({ metadata: data.metadata });
        if (data.query_rewrite) renderRewrite(data.query_rewrite);
        if (['completed', 'success'].includes(String(data.status).toLowerCase())) {
            loadSources();
        }
    }

    async function initProgress(researchId) {
        state.researchId = researchId;
        setText('ldr-agent-console-run-id', researchId ? researchId.slice(0, 8) : '—');
        renderCounts();
        renderBudget(0, null);
        if (!researchId) return;
        try {
            const response = await fetch(`/api/research/${encodeURIComponent(researchId)}`);
            if (!response.ok) return;
            applyDetails(await response.json());
        } catch (_error) {
            // The existing progress component owns error handling for the page.
        }
    }

    window.AgentConsole = {
        initProgress,
        loadSources,
        recordEvent,
        updateStatus
    };

    document.addEventListener('DOMContentLoaded', initComposer);
})();
