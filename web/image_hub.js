import { app } from "../../../scripts/app.js";

// Interactive crop rectangle on the thumbnail preview: drag to draw, drag
// inside the rectangle to move it, drag a corner to resize, click (without
// dragging) outside it to clear -- idea credited to noEmbryo's "Load Image
// (from path)" node in ComfyUI-noEmbryo
// (https://github.com/noembryo/ComfyUI-noEmbryo), which offers the same
// interaction set on its own preview. The canvas math and event handling
// below are our own implementation, not copied from that project's source.
const HANDLE_PX = 10;     // corner-handle hit-test radius, in canvas px
const CLICK_SLOP_PX = 3;  // max mouse movement to still count as a plain click

function computeContainRect(boxW, boxH, imgW, imgH) {
    // Mirrors CSS `object-fit: contain`: the {x,y,w,h} sub-rectangle (in
    // box-local canvas px) the image actually occupies once letterboxed.
    if (!imgW || !imgH || !boxW || !boxH) return { x: 0, y: 0, w: boxW, h: boxH };
    const boxAspect = boxW / boxH;
    const imgAspect = imgW / imgH;
    if (imgAspect > boxAspect) {
        const w = boxW;
        const h = boxW / imgAspect;
        return { x: 0, y: (boxH - h) / 2, w, h };
    }
    const h = boxH;
    const w = boxH * imgAspect;
    return { x: (boxW - w) / 2, y: 0, w, h };
}

function parseCropRect(str) {
    if (!str) return null;
    const parts = String(str).split(",").map(Number);
    if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) return null;
    const [x, y, w, h] = parts;
    if (w <= 0 || h <= 0) return null;
    return { x, y, w, h };
}

function formatCropRect(rect) {
    return `${rect.x},${rect.y},${rect.w},${rect.h}`;
}

function cornerPoints(px) {
    return [
        ["nw", px.x, px.y], ["ne", px.x + px.w, px.y],
        ["sw", px.x, px.y + px.h], ["se", px.x + px.w, px.y + px.h],
    ];
}

function hitCorner(px, mx, my) {
    for (const [name, hx, hy] of cornerPoints(px)) {
        if (Math.hypot(mx - hx, my - hy) <= HANDLE_PX) return name;
    }
    return null;
}

function insideRect(px, mx, my) {
    return mx >= px.x && mx <= px.x + px.w && my >= px.y && my <= px.y + px.h;
}

app.registerExtension({
    name: "UniversalImageHub.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "UniversalImageHub") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;

            // Thumbnail preview shown under the filename dropdown, backed by
            // /universal_image_hub/thumbnail (a small server-resized JPEG --
            // not used for the node's actual IMAGE/MASK output, just display
            // plus the crop-rectangle overlay below). Fetched (rather than
            // set directly as an <img src>) so the X-Image-Width/
            // X-Image-Height response headers -- the *original* file's
            // dimensions, not the downscaled thumbnail's -- can be read for
            // the caption and for the crop overlay's coordinate math.
            const previewImg = document.createElement("img");
            previewImg.style.cssText = "display:block; width:100%; height:160px; object-fit:contain; " +
                "background:#111; border-radius:4px; border:1px solid #333;";
            previewImg.alt = "preview";
            previewImg.onerror = () => { previewBox.style.display = "none"; };

            // Transparent canvas overlaid on previewImg (via previewBox's
            // relative positioning below) purely for the crop rectangle --
            // the image itself is still shown by previewImg, this just
            // captures drag events and draws the rectangle/handles.
            const cropCanvas = document.createElement("canvas");
            cropCanvas.style.cssText = "display:block; position:absolute; top:0; left:0; " +
                "width:100%; height:160px; cursor:crosshair;";

            const previewBox = document.createElement("div");
            previewBox.style.cssText = "position:relative; width:100%; height:160px; display:none;";
            previewBox.appendChild(previewImg);
            previewBox.appendChild(cropCanvas);

            const dimsLabel = document.createElement("div");
            dimsLabel.style.cssText = "width:100%; text-align:center; font-size:11px; " +
                "color:#aaa; margin-top:2px; display:none;";

            const previewWrap = document.createElement("div");
            previewWrap.style.cssText = "width:100%;";
            previewWrap.appendChild(previewBox);
            previewWrap.appendChild(dimsLabel);
            // Not added to the node yet -- see the "Sort By" widget below,
            // which is added first so the preview appears under it.

            let previewObjectUrl = null;
            let previewRequestId = 0;
            let origWidth = null;
            let origHeight = null;
            let lastPreviewFilename = node.widgets?.find((w) => w.name === "filename")?.value ?? null;
            let dragState = null;

            function getCropWidget() {
                return node.widgets?.find((w) => w.name === "crop_rect");
            }

            function getCropRect() {
                return parseCropRect(getCropWidget()?.value);
            }

            function setCropRect(rect) {
                const w = getCropWidget();
                if (!w) return;
                w.value = rect ? formatCropRect(rect) : "";
            }

            function resizeCanvasToBox() {
                // Measured from cropCanvas itself, not previewBox -- so this
                // always agrees with pointerPos()'s own getBoundingClientRect
                // call below, regardless of any border/padding either
                // element might pick up later.
                const rect = cropCanvas.getBoundingClientRect();
                const w = Math.max(1, Math.round(rect.width));
                const h = Math.max(1, Math.round(rect.height));
                if (cropCanvas.width !== w || cropCanvas.height !== h) {
                    cropCanvas.width = w;
                    cropCanvas.height = h;
                }
                return { w, h };
            }

            function drawOverlay() {
                const { w: boxW, h: boxH } = resizeCanvasToBox();
                const ctx = cropCanvas.getContext("2d");
                ctx.clearRect(0, 0, boxW, boxH);
                if (!origWidth || !origHeight) return;

                const contain = computeContainRect(boxW, boxH, origWidth, origHeight);
                const rect = dragState?.previewRect ?? getCropRect();
                if (!rect) return;

                const px = {
                    x: contain.x + rect.x * contain.w,
                    y: contain.y + rect.y * contain.h,
                    w: rect.w * contain.w,
                    h: rect.h * contain.h,
                };

                ctx.fillStyle = "rgba(0,0,0,0.55)";
                ctx.fillRect(0, 0, boxW, px.y);
                ctx.fillRect(0, px.y + px.h, boxW, boxH - px.y - px.h);
                ctx.fillRect(0, px.y, px.x, px.h);
                ctx.fillRect(px.x + px.w, px.y, boxW - px.x - px.w, px.h);

                ctx.strokeStyle = "#4af";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(px.x + 0.5, px.y + 0.5, Math.max(0, px.w - 1), Math.max(0, px.h - 1));
                ctx.setLineDash([]);

                ctx.fillStyle = "#4af";
                for (const [, hx, hy] of cornerPoints(px)) {
                    ctx.fillRect(hx - 3, hy - 3, 6, 6);
                }
            }

            function pointerPos(evt) {
                const rect = cropCanvas.getBoundingClientRect();
                return { mx: evt.clientX - rect.left, my: evt.clientY - rect.top };
            }

            cropCanvas.addEventListener("mousedown", (evt) => {
                if (!origWidth || !origHeight) return;
                const { w: boxW, h: boxH } = resizeCanvasToBox();
                const contain = computeContainRect(boxW, boxH, origWidth, origHeight);
                const { mx, my } = pointerPos(evt);
                const existing = getCropRect();
                const existingPx = existing ? {
                    x: contain.x + existing.x * contain.w, y: contain.y + existing.y * contain.h,
                    w: existing.w * contain.w, h: existing.h * contain.h,
                } : null;

                let mode = "draw";
                let corner = null;
                if (existingPx) {
                    corner = hitCorner(existingPx, mx, my);
                    if (corner) mode = "resize";
                    else if (insideRect(existingPx, mx, my)) mode = "move";
                }

                dragState = {
                    mode, corner, startX: mx, startY: my, moved: false,
                    contain, existingPx, previewRect: existing,
                };
                evt.preventDefault();
            });

            cropCanvas.addEventListener("mousemove", (evt) => {
                if (!dragState) return;
                const { mx, my } = pointerPos(evt);
                if (Math.hypot(mx - dragState.startX, my - dragState.startY) > CLICK_SLOP_PX) {
                    dragState.moved = true;
                }
                const { contain } = dragState;
                const clampPt = (x, y) => ({
                    x: Math.min(Math.max(x, contain.x), contain.x + contain.w),
                    y: Math.min(Math.max(y, contain.y), contain.y + contain.h),
                });

                let px = null;
                if (dragState.mode === "draw") {
                    const a = clampPt(dragState.startX, dragState.startY);
                    const b = clampPt(mx, my);
                    px = {
                        x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
                        w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y),
                    };
                } else if (dragState.mode === "move") {
                    const dx = mx - dragState.startX;
                    const dy = my - dragState.startY;
                    const base = dragState.existingPx;
                    const x = Math.min(Math.max(base.x + dx, contain.x), contain.x + contain.w - base.w);
                    const y = Math.min(Math.max(base.y + dy, contain.y), contain.y + contain.h - base.h);
                    px = { x, y, w: base.w, h: base.h };
                } else if (dragState.mode === "resize") {
                    const base = dragState.existingPx;
                    let { x, y, w, h } = base;
                    const c = clampPt(mx, my);
                    if (dragState.corner.includes("w")) { w = x + w - c.x; x = c.x; }
                    if (dragState.corner.includes("e")) { w = c.x - x; }
                    if (dragState.corner.includes("n")) { h = y + h - c.y; y = c.y; }
                    if (dragState.corner.includes("s")) { h = c.y - y; }
                    if (w < 0) { x += w; w = -w; }
                    if (h < 0) { y += h; h = -h; }
                    px = { x, y, w, h };
                }

                if (px && px.w > 1 && px.h > 1) {
                    dragState.previewRect = {
                        x: (px.x - contain.x) / contain.w,
                        y: (px.y - contain.y) / contain.h,
                        w: px.w / contain.w,
                        h: px.h / contain.h,
                    };
                }
                drawOverlay();
            });

            function endDrag() {
                if (!dragState) return;
                if (!dragState.moved && dragState.mode === "draw") {
                    // Plain click outside any existing rectangle -- clear.
                    setCropRect(null);
                } else if (dragState.previewRect) {
                    setCropRect(dragState.previewRect);
                }
                dragState = null;
                drawOverlay();
                node.graph?.setDirtyCanvas(true, true);
            }
            cropCanvas.addEventListener("mouseup", endDrag);
            cropCanvas.addEventListener("mouseleave", () => { if (dragState) endDrag(); });

            let resizeObserver = null;
            if (typeof ResizeObserver !== "undefined") {
                resizeObserver = new ResizeObserver(() => drawOverlay());
                resizeObserver.observe(previewBox);
            }

            async function updatePreview() {
                const filenameWidget = node.widgets?.find((w) => w.name === "filename");
                const folderWidget = node.widgets?.find((w) => w.name === "folder_type");
                const customPathWidget = node.widgets?.find((w) => w.name === "custom_path");
                const filename = filenameWidget?.value;

                // A crop drawn for one image is meaningless on another, so
                // clear it the moment the resolved filename actually changes
                // -- whether from picking a different file, a refresh that
                // had to fall back because the saved one vanished, or a
                // clipboard paste -- but NOT on the very first call after the
                // node is placed/loaded, so a saved workflow's crop survives.
                if (filename !== lastPreviewFilename) {
                    setCropRect(null);
                    lastPreviewFilename = filename;
                }

                if (!filename || filename.startsWith("<no image files")) {
                    previewBox.style.display = "none";
                    dimsLabel.style.display = "none";
                    origWidth = null;
                    origHeight = null;
                    return;
                }
                const folderType = folderWidget?.value ?? "input";
                const customPath = customPathWidget?.value ?? "";
                const params = new URLSearchParams({ folder_type: folderType, custom_path: customPath, filename });

                // Guard against an earlier, slower request resolving after a
                // newer selection's -- only the latest requested preview wins.
                const requestId = ++previewRequestId;
                try {
                    const res = await fetch(`/universal_image_hub/thumbnail?${params}`);
                    if (!res.ok) throw new Error(`server returned ${res.status} ${res.statusText}`);
                    const width = res.headers.get("X-Image-Width");
                    const height = res.headers.get("X-Image-Height");
                    const blob = await res.blob();
                    if (requestId !== previewRequestId) return;

                    const url = URL.createObjectURL(blob);
                    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
                    previewObjectUrl = url;
                    previewImg.src = url;
                    previewBox.style.display = "";

                    origWidth = width ? parseInt(width, 10) : null;
                    origHeight = height ? parseInt(height, 10) : null;
                    if (width && height) {
                        dimsLabel.textContent = `${width} × ${height}`;
                        dimsLabel.style.display = "";
                    } else {
                        dimsLabel.style.display = "none";
                    }
                    // Deferred one frame so previewBox's just-changed display
                    // has taken effect before measuring it for the canvas.
                    requestAnimationFrame(() => drawOverlay());
                    node.graph?.setDirtyCanvas(true, true);
                } catch (err) {
                    if (requestId !== previewRequestId) return;
                    console.error("UniversalImageHub preview failed:", err);
                    previewBox.style.display = "none";
                    dimsLabel.style.display = "none";
                    origWidth = null;
                    origHeight = null;
                }
            }

            // `filename` is now a searchable COMBO widget (same UX as the checkpoint/
            // LoRA loaders -- click to open, type to filter), backed by a directory
            // scan done server-side in image_hub.py. That scan only happens when
            // ComfyUI rebuilds this node's schema (server start / page reload), so
            // this button re-runs it on demand mid-session, and folder_type changes
            // trigger it automatically below.
            async function refreshFileList(alertIfEmpty, force) {
                const filenameWidget = node.widgets?.find((w) => w.name === "filename");
                const folderWidget = node.widgets?.find((w) => w.name === "folder_type");
                const customPathWidget = node.widgets?.find((w) => w.name === "custom_path");
                const sortByWidget = node.widgets?.find((w) => w.name === "sort_by");
                if (!filenameWidget) return;

                const folderType = folderWidget?.value ?? "input";
                const customPath = customPathWidget?.value ?? "";
                const sortBy = sortByWidget?.value ?? "Name (A-Z)";
                const params = new URLSearchParams({ folder_type: folderType, custom_path: customPath, sort_by: sortBy });
                // The server caches scans (per folder + sort order) for a few
                // seconds so placing several of these nodes at once doesn't
                // re-walk a large folder repeatedly -- the explicit Refresh
                // button bypasses that to always be current.
                if (force) params.set("force", "1");

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
                updatePreview();
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

            // UI-only widget (not part of this node's Python INPUT_TYPES, so it
            // isn't sent to the backend or saved in the workflow) that controls
            // how the filename dropdown is ordered. Values must match the exact
            // strings image_hub.py's _SORT_NAME/_SORT_NEWEST expect.
            node.addWidget("combo", "sort_by", "Name (A-Z)", () => refreshFileList(false), {
                values: ["Name (A-Z)", "Newest First"],
                serialize: false,
            });

            node.addDOMWidget("image_preview", "preview", previewWrap, { serialize: false });

            // crop_rect IS a real Python input (image_hub.py applies it at load
            // time), but it's driven entirely by dragging on the preview above,
            // not meant to be hand-typed -- so once ComfyUI has created its
            // normal text widget, collapse it to zero height instead of adding
            // a second, competing UI for the same value. It stays a real
            // widget (so its value still serializes into the saved workflow
            // and prompt), just an invisible one.
            const cropWidget = getCropWidget();
            if (cropWidget) {
                cropWidget.computeSize = () => [0, -4];
                cropWidget.draw = () => {};
            }

            // Update the preview whenever the user picks a different file from
            // the dropdown directly (refreshFileList already covers the cases
            // where filename's value is changed programmatically).
            const filenameWidgetForPreview = this.widgets?.find((w) => w.name === "filename");
            if (filenameWidgetForPreview) {
                const origFilenameCallback = filenameWidgetForPreview.callback;
                filenameWidgetForPreview.callback = function (...args) {
                    const ret = origFilenameCallback?.apply(this, args);
                    updatePreview();
                    return ret;
                };
            }

            this.addWidget("button", "🔄 Refresh File List", null, () => refreshFileList(true, true));

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
                        updatePreview();
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

            // Release the preview's blob URL and stop watching for resizes
            // when the node is deleted, rather than leaking them for the
            // lifetime of the browser tab.
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function (...args) {
                if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
                resizeObserver?.disconnect();
                return origOnRemoved?.apply(this, args);
            };

            // The combo's initial choices (from INPUT_TYPES) were only current as of
            // the last full page load / object_info fetch -- do one live re-scan as
            // soon as the node is placed so it reflects whatever's on disk right now.
            refreshFileList(false);
        };
    }
});
