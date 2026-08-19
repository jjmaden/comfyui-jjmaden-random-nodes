import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "UniversalImageHub.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "UniversalImageHub") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            // `filename` is now a searchable COMBO widget (same UX as the checkpoint/
            // LoRA loaders -- click to open, type to filter), backed by a directory
            // scan done server-side in image_hub.py. That scan only happens when
            // ComfyUI rebuilds this node's schema (server start / page reload), so
            // this button re-runs it on demand mid-session, and folder_type changes
            // trigger it automatically below.
            async function refreshFileList(alertIfEmpty) {
                const filenameWidget = node.widgets?.find((w) => w.name === "filename");
                const folderWidget = node.widgets?.find((w) => w.name === "folder_type");
                const customPathWidget = node.widgets?.find((w) => w.name === "custom_path");
                if (!filenameWidget) return;

                const folderType = folderWidget?.value ?? "input";
                const customPath = customPathWidget?.value ?? "";
                const params = new URLSearchParams({ folder_type: folderType, custom_path: customPath });

                let files;
                try {
                    const res = await fetch(`/universal_image_hub/files?${params}`);
                    if (!res.ok) throw new Error(`server returned ${res.status} ${res.statusText}`);
                    ({ files } = await res.json());
                } catch (err) {
                    console.error("UniversalImageHub refresh failed:", err);
                    alert("Couldn't refresh the file list -- see the browser console for details.");
                    return;
                }

                if (!files || !files.length) {
                    files = ["<no image files found -- add files and click Refresh>"];
                    if (alertIfEmpty) alert(`No files found in the ${folderType} folder.`);
                }

                filenameWidget.options = filenameWidget.options || {};
                filenameWidget.options.values = files;
                if (!files.includes(filenameWidget.value)) {
                    filenameWidget.value = files[0];
                }
                node.graph?.setDirtyCanvas(true, true);
            }

            // Re-scan automatically whenever folder_type changes, so switching between
            // input/output/custom doesn't leave a stale list from whichever folder was
            // showing before.
            const folderWidget = this.widgets?.find((w) => w.name === "folder_type");
            if (folderWidget) {
                const origCallback = folderWidget.callback;
                folderWidget.callback = function (...args) {
                    const ret = origCallback?.apply(this, args);
                    refreshFileList(false);
                    return ret;
                };
            }

            this.addWidget("button", "🔄 Refresh File List", null, () => refreshFileList(true));

            this.addWidget("button", "Paste from Clipboard", null, async () => {
                let items;
                try {
                    items = await navigator.clipboard.read();
                } catch (err) {
                    console.error("UniversalImageHub paste failed:", err);
                    alert("Couldn't read the clipboard (needs HTTPS or localhost, plus " +
                          "clipboard permission). See the browser console for details.");
                    return;
                }

                for (const item of items) {
                    // Use whichever MIME type on this item actually matched,
                    // not item.types[0] -- a copied image often also carries
                    // e.g. text/html, which may be listed first.
                    const imageType = item.types.find((t) => t.startsWith("image/"));
                    if (!imageType) continue;

                    try {
                        const blob = await item.getType(imageType);
                        const ext = imageType.split("/")[1] || "png";
                        const formData = new FormData();
                        formData.append("image", blob, `pasted_${Date.now()}.${ext}`);
                        formData.append("type", "input");
                        formData.append("overwrite", "false");

                        const res = await fetch("/upload/image", { method: "POST", body: formData });
                        if (!res.ok) {
                            throw new Error(`server returned ${res.status} ${res.statusText}`);
                        }
                        const data = await res.json();
                        const savedName = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;

                        const filenameWidget = this.widgets?.find((w) => w.name === "filename");
                        const folderTypeWidget = this.widgets?.find((w) => w.name === "folder_type");
                        if (folderTypeWidget) folderTypeWidget.value = "input";  // that's where /upload/image just put it
                        if (filenameWidget) {
                            filenameWidget.options = filenameWidget.options || {};
                            const values = filenameWidget.options.values || [];
                            if (!values.includes(savedName)) {
                                filenameWidget.options.values = [savedName, ...values];
                            }
                            filenameWidget.value = savedName;
                        }
                        this.graph?.setDirtyCanvas(true, true);

                        alert(`Pasted as "${savedName}" and selected it in filename (folder_type: input).`);
                    } catch (err) {
                        console.error("UniversalImageHub paste failed:", err);
                        alert("Paste upload failed -- see the browser console for details.");
                    }
                    return; // only handle the first image found on the clipboard
                }

                alert("No image found on the clipboard.");
            });

            // The combo's initial choices (from INPUT_TYPES) were only current as of
            // the last full page load / object_info fetch -- do one live re-scan as
            // soon as the node is placed so it reflects whatever's on disk right now.
            refreshFileList(false);
        };
    }
});
