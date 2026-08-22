import io
import os
import time
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths

from ._file_utils import strip_wrapping as _strip_wrapping

try:
    from server import PromptServer
    from aiohttp import web
except ImportError:
    # Only needed for the optional refresh-button API route below; the node
    # itself works fine without it (just without live re-scan from the UI).
    PromptServer = None
    web = None


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_NO_FILES_SENTINEL = "<no image files found -- add files and click Refresh>"

# Short-lived cache of {base_path: (scanned_at, result)} so placing/pasting
# several UniversalImageHub nodes at once (each triggering its own scan on
# creation) doesn't re-walk a large folder once per node within the same
# few seconds. The explicit "Refresh File List" button bypasses this via
# force=True, so it's always accurate on demand.
_scan_cache = {}
_SCAN_CACHE_TTL = 3.0  # seconds


def _scan_image_files(base_path, force=False):
    """Recursively list files under base_path as base_path-relative, forward-slash
    paths, sorted. Prefers recognized image extensions; if none are found, falls
    back to every file (so an unusual extension doesn't just make the dropdown
    mysteriously empty). Returns a single sentinel entry if base_path doesn't
    exist or truly has nothing in it, since a COMBO widget can't have zero
    choices -- `_resolve_path` recognizes and rejects that sentinel with a
    clear error rather than trying to load a file literally named that."""
    if not base_path or not os.path.isdir(base_path):
        return [_NO_FILES_SENTINEL]

    now = time.monotonic()
    if not force:
        cached = _scan_cache.get(base_path)
        if cached is not None and (now - cached[0]) < _SCAN_CACHE_TTL:
            return cached[1]

    matched, all_files = [], []
    for root, _dirs, files in os.walk(base_path):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), base_path).replace(os.sep, "/")
            all_files.append(rel)
            if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                matched.append(rel)

    result = sorted(matched) if matched else sorted(all_files)
    result = result if result else [_NO_FILES_SENTINEL]
    _scan_cache[base_path] = (now, result)
    return result


def _base_path_for(folder_type, custom_path):
    if folder_type == "input":
        return folder_paths.get_input_directory()
    elif folder_type == "output":
        return folder_paths.get_output_directory()
    return _strip_wrapping(custom_path)


def _resolve_path(folder_type, filename, custom_path):
    """Shared by load_it/IS_CHANGED/VALIDATE_INPUTS so all three agree on the
    same file. Raises ValueError on anything that can't be resolved safely."""
    filename = _strip_wrapping(filename)
    if not filename or filename == _NO_FILES_SENTINEL:
        raise ValueError(
            "UniversalImageHub: no file selected -- the folder looked empty when the "
            "dropdown was last scanned. Add an image to the folder, then click "
            "\"Refresh File List\" (or paste one via the clipboard button)."
        )

    base_path = _base_path_for(folder_type, custom_path)
    if folder_type == "custom" and not base_path:
        raise ValueError("UniversalImageHub: folder_type is 'custom' but custom_path is empty.")

    image_path = os.path.abspath(os.path.join(base_path, filename))

    if folder_type in ("input", "output"):
        # "custom" is explicitly meant to be any absolute path; input/output
        # are meant to stay scoped to ComfyUI's own folders the way the
        # built-in loaders are, so a filename like "..\..\something" can't
        # walk out of them.
        base_real = os.path.realpath(base_path)
        path_real = os.path.realpath(image_path)
        try:
            inside = os.path.commonpath([base_real, path_real]) == base_real
        except ValueError:
            inside = False  # e.g. different drives on Windows
        if not inside:
            raise ValueError(f"UniversalImageHub: '{filename}' resolves outside the {folder_type} "
                              f"folder ({image_path}). Use folder_type='custom' for paths elsewhere.")

    return image_path


# Registered once at import time, the same pattern other custom node packs use
# for their own refresh-combo buttons (ComfyUI-Manager, ComfyUI-Custom-Scripts,
# etc). Backs the "Refresh File List" button in web/image_hub.js: the JS reads
# the node's current folder_type/custom_path widgets and asks this route to
# re-scan that exact folder, so the dropdown reflects files added after
# ComfyUI last loaded -- no browser reload or ComfyUI restart needed.
if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    try:
        @PromptServer.instance.routes.get("/universal_image_hub/files")
        async def _universal_image_hub_files(request):
            folder_type = request.rel_url.query.get("folder_type", "input")
            custom_path = request.rel_url.query.get("custom_path", "")
            force = request.rel_url.query.get("force", "") in ("1", "true", "yes")
            base_path = _base_path_for(folder_type, custom_path)
            return web.json_response({"files": _scan_image_files(base_path, force=force)})

        # Backs the preview thumbnail shown under the filename dropdown in
        # web/image_hub.js. Always re-reads the file (Cache-Control: no-store)
        # so editing/replacing it on disk is reflected without a manual
        # re-queue -- consistent with how IS_CHANGED already tracks mtime for
        # the node's real (non-preview) output.
        @PromptServer.instance.routes.get("/universal_image_hub/thumbnail")
        async def _universal_image_hub_thumbnail(request):
            folder_type = request.rel_url.query.get("folder_type", "input")
            custom_path = request.rel_url.query.get("custom_path", "")
            filename = request.rel_url.query.get("filename", "")
            try:
                image_path = _resolve_path(folder_type, filename, custom_path)
            except ValueError as exc:
                return web.Response(status=400, text=str(exc))
            if not os.path.exists(image_path):
                return web.Response(status=404, text=f"'{filename}' does not exist")
            try:
                with Image.open(image_path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((320, 320), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="JPEG", quality=82)
            except Exception as exc:
                return web.Response(status=415, text=f"couldn't render a preview for '{filename}': {exc}")
            return web.Response(body=buf.getvalue(), content_type="image/jpeg",
                                 headers={"Cache-Control": "no-store"})
    except Exception as exc:
        # e.g. route already registered by a hot-reload -- not fatal (the node
        # still works via its normal INPUT_TYPES scan), but log it since a real
        # registration failure would otherwise only show up as a silent 404
        # when the "Refresh File List" button is clicked.
        print(f"[UniversalImageHub] couldn't register /universal_image_hub/files route: {exc}")


class UniversalImageHub:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_type": (["input", "output", "custom"],),
                # A searchable COMBO, same UX as the checkpoint/LoRA loaders (click
                # to open, type to filter). Scanned fresh every time ComfyUI (re)builds
                # this schema -- e.g. on server start or browser reload -- and can also
                # be re-scanned live mid-session via the "Refresh File List" button.
                "filename": (_scan_image_files(folder_paths.get_input_directory()),),
            },
            "optional": {
                "custom_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_it"
    CATEGORY = "image"
    DESCRIPTION = ("Loads an image from a searchable dropdown (like the checkpoint/LoRA loaders) "
                   "instead of ComfyUI's built-in LoadImage. 'input'/'output' stay scoped to those "
                   "ComfyUI folders; 'custom' accepts any absolute path via custom_path. The dropdown "
                   "is scanned fresh whenever ComfyUI rebuilds its node schema (server start / page "
                   "reload); use the \"Refresh File List\" button (web/image_hub.js) to re-scan "
                   "mid-session without either of those, or the \"Paste from Clipboard\" button to "
                   "upload straight into the input folder and select it automatically. Scans of a "
                   "given folder are cached for a few seconds so placing several of these nodes at "
                   "once doesn't re-walk a large folder repeatedly -- the Refresh button always "
                   "bypasses that cache. Shows a thumbnail preview of the selected file under the "
                   "dropdown, updated whenever the selection changes.")

    @classmethod
    def VALIDATE_INPUTS(cls, folder_type, filename, custom_path=""):
        try:
            _resolve_path(folder_type, filename, custom_path)
        except ValueError as exc:
            return str(exc)
        return True

    def load_it(self, folder_type, filename, custom_path=""):
        image_path = _resolve_path(folder_type, filename, custom_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"UniversalImageHub Error: File does not exist at '{image_path}'")

        # Standard ComfyUI Image Loading logic
        try:
            i = Image.open(image_path)
        except Exception as exc:
            raise ValueError(
                f"UniversalImageHub: '{filename}' doesn't look like a readable image ({exc}). "
                f"Note: when a folder has no recognized image extensions, the dropdown falls back "
                f"to listing every file in it -- make sure you picked an actual image."
            ) from exc
        i = ImageOps.exif_transpose(i)
        img_rgb = i.convert("RGB")

        # Convert to ComfyUI standard float32 tensor [B, H, W, C]
        image_np = np.array(img_rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        # Dynamic Mask Generation (always matches the loaded image's own
        # dimensions, unlike ComfyUI's own core LoadImage node, which
        # defaults to a fixed 64x64 mask when there's no alpha channel)
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask_tensor = 1. - torch.from_numpy(mask)
            mask_tensor = mask_tensor[None,]  # Shape: [1, H, W]
        else:
            height, width = image_np.shape[0], image_np.shape[1]
            mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)

        return (image_tensor, mask_tensor)

    @classmethod
    def IS_CHANGED(s, folder_type, filename, custom_path=""):
        # Tells ComfyUI to re-run the node if the file's modification time changes
        try:
            image_path = _resolve_path(folder_type, filename, custom_path)
        except ValueError:
            return float("NaN")  # let load_it raise the real, descriptive error
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return float("NaN")
