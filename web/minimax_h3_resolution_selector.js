import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "MiniMaxH3ResolutionSelector.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MiniMaxH3ResolutionSelector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;
            const aspectWidget = node.widgets?.find((w) => w.name === "aspect_ratio");
            const resolutionWidget = node.widgets?.find((w) => w.name === "resolution");
            if (!aspectWidget || !resolutionWidget) return;

            // The full original list (includes the two legacy compat values) --
            // always filter FROM this master copy, never from whatever's
            // currently displayed, so switching ratios repeatedly doesn't
            // progressively lose entries.
            const allResolutions = (resolutionWidget.options.values || []).slice();

            let presetTable = null; // { ratios, tiers, table } fetched from the server
            let currentTier = null; // which tier the current resolution value belongs to, if known

            function findTierFor(value) {
                if (!presetTable) return null;
                for (const ratioLabel of presetTable.ratios) {
                    const row = presetTable.table[ratioLabel];
                    for (const tier of presetTable.tiers) {
                        if (row && row[tier] === value) return tier;
                    }
                }
                return null;
            }

            // Only ever called from an actual aspect_ratio interaction (see
            // below) -- never automatically on workflow load, so loading an
            // older saved workflow never silently changes its resolution value.
            function applyFilter() {
                const ratioLabel = aspectWidget.value;
                const row = presetTable?.table?.[ratioLabel];
                if (!row) {
                    // "Custom / manual", or presets haven't loaded yet -- restore
                    // the full unfiltered list rather than leaving a stale one.
                    resolutionWidget.options.values = allResolutions;
                    return;
                }

                const filtered = presetTable.tiers.map((t) => row[t]).filter(Boolean);
                resolutionWidget.options.values = filtered;

                // Preserve whichever tier the previous value belonged to, so
                // switching ratios feels like a real linked dropdown instead of
                // resetting every time; fall back to H3's own default tier.
                const targetTier = (currentTier && row[currentTier]) ? currentTier
                    : (row["Large — H3 default"] ? "Large — H3 default" : presetTable.tiers[0]);
                resolutionWidget.value = row[targetTier];
                currentTier = targetTier;
                node.graph?.setDirtyCanvas(true, true);
            }

            const origAspectCallback = aspectWidget.callback;
            aspectWidget.callback = function (...args) {
                const ret = origAspectCallback?.apply(this, args);
                applyFilter();
                return ret;
            };

            // Track which tier the user picks manually too, so a later ratio
            // switch preserves it instead of only tracking ratio-driven picks.
            const origResolutionCallback = resolutionWidget.callback;
            resolutionWidget.callback = function (...args) {
                const ret = origResolutionCallback?.apply(this, args);
                currentTier = findTierFor(resolutionWidget.value);
                return ret;
            };

            fetch("/minimax_h3_resolution_selector/presets")
                .then((res) => {
                    if (!res.ok) throw new Error(`server returned ${res.status} ${res.statusText}`);
                    return res.json();
                })
                .then((data) => {
                    presetTable = data;
                    currentTier = findTierFor(resolutionWidget.value);
                    // Deliberately NOT calling applyFilter() here -- see the
                    // function's own comment. This just makes the table ready
                    // for the next actual aspect_ratio change.
                })
                .catch((err) => {
                    console.error("MiniMaxH3ResolutionSelector: couldn't fetch preset table:", err);
                });
        };
    },
});
