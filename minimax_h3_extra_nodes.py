"""MiniMax H3 companion nodes.

These are best-effort, from-scratch reimplementations of five node types
found in a third-party MiniMax H3 workflow JSON that don't exist in any
public repository:

    MiniMaxH3UnifiedToVideo
    MiniMaxH3ConcatAVLatent
    MiniMaxH3ResolutionSelector
    MiniMaxH3AudioLock
    MiniMaxH3MultimodalChat

Three other node types referenced by that same workflow (MiniMaxH3SigmaShift,
MiniMaxH3MemoryEfficientSageAttentionPatch, MiniMaxH3ReferenceSplitter) are
real and ship in ComfyUI core / ComfyUI-KJNodes / the "Fantastic H3 Prompt
Builder" pack respectively -- install/update those instead of looking for
them here.

The math and latent-shape helpers below (align_frame_count, video_latent_t,
temporal_shape, adapt_canvas, _resize, _encode_ref_audio, _empty_av_latent,
and the constants above them) are copied verbatim from ComfyUI's own native
MiniMax H3 node file so the outputs of these nodes are byte-for-byte
compatible with the real MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo /
MiniMaxH3AddGuide / EmptyMiniMaxH3LatentAV nodes. Everything past that point
(MiniMaxH3UnifiedToVideo's mode-merging logic, MiniMaxH3ConcatAVLatent,
MiniMaxH3ResolutionSelector's preset table, MiniMaxH3AudioLock's blend
behavior, and all of MiniMaxH3MultimodalChat) is inferred from the workflow
JSON's node graph and widget values -- there is no public source for any of
it, so treat it as a working approximation, not a faithful clone of
whatever the original, privately-distributed pack actually did internally.
"""

import base64
import io as _io
import json
import math
import os
import re
import urllib.error
import urllib.request

import torch
import torchaudio

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers

try:
    from nodes import MAX_RESOLUTION
except Exception:
    MAX_RESOLUTION = 16384

try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web
except ImportError:
    # Only needed for the optional MiniMaxH3ResolutionSelector preset-table
    # route below; the node itself works fine without it (aspect_ratio just
    # won't filter the resolution dropdown from the JS side).
    PromptServer = None
    _aiohttp_web = None


def _apply_dotted_kwargs(values, kwargs):
    """Some ComfyUI frontend builds auto-group sequentially-numbered sibling
    optional inputs (e.g. ref_image_0/1/2/3, ref_audio_0/1) into a UI
    collection and submit them to the backend as a dotted key such as
    'ref_images.ref_image_0' instead of the plain 'ref_image_0' a classic
    (non-**kwargs) node function signature declares -- ComfyUI core has the
    same class of issue with its own 'hidden_inputs' parameter (see
    Comfy-Org/ComfyUI#9215), and the fix there is the same one used here:
    accept **kwargs and recover the real name instead of crashing with
    "unexpected keyword argument".

    `values` is a dict of {param_name: current_value} for every optional
    param on the caller; only slots that are still None get filled in, so
    this can't clobber a value that arrived normally under its plain name.
    """
    if not kwargs:
        return values
    for key, value in kwargs.items():
        plain_name = key.rsplit(".", 1)[-1]
        if plain_name in values and values[plain_name] is None:
            values[plain_name] = value
    return values


# ---------------------------------------------------------------------------
# Verbatim from ComfyUI's native MiniMax H3 node file (see module docstring).
# ---------------------------------------------------------------------------

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


# ---------------------------------------------------------------------------
# MiniMaxH3ResolutionSelector
#
# Not found in any public repo. Widget values seen in the wild:
#   ["2:3 (Portrait Photo)", "384×576"]
#   ["2:3 (Portrait Photo)", "736×1120"]
# i.e. an aspect-ratio label plus a literal "W×H" resolution string. The real
# node almost certainly used JS to re-populate the second combo's options
# whenever the first one changed; a plain Python node can't do that
# dynamically, so this version makes the second widget authoritative (it's
# parsed directly for width/height) and the first widget purely a label.
# Both of the exact values above are included in the preset list so old
# saved workflows keep loading with a valid selection.
# ---------------------------------------------------------------------------

def _capped_canvas(ratio_w, ratio_h, short_edge):
    max_pixels = (short_edge / BASE_SHORT_EDGE) ** 2 * MAX_PIXELS
    ratio = ratio_w / ratio_h
    if ratio >= 1.0:
        nom_w, nom_h = short_edge * ratio, short_edge
    else:
        nom_w, nom_h = short_edge, short_edge / ratio
    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _exact_ratio_scales(w, h, max_mp):
    """All valid (k, width, height) scales at the exact W:H ratio, both axes
    divisible by 32, capped at max_mp megapixels. Ported (credit where due)
    from the real, already-installed ResolutionSelectorZerohackz pack's
    ImageRatioSelectorZerohackz node -- it's a cleaner way to size off a
    reference image's exact ratio than anything the rest of this file does."""
    w = max(32, (w // 32) * 32)
    h = max(32, (h // 32) * 32)
    g = math.gcd(w // 32, h // 32)
    u_w, u_h = (w // 32) // g, (h // 32) // g
    scales = []
    k = 1
    while True:
        out_w, out_h = u_w * k * 32, u_h * k * 32
        if (out_w * out_h) / 1_000_000 > max_mp:
            break
        scales.append((k, out_w, out_h))
        k += 1
    return scales


_RATIOS = [
    ("1:1 (Square)", 1, 1),
    ("16:9 (Landscape Widescreen)", 16, 9),
    ("9:16 (Vertical Video)", 9, 16),
    ("4:3 (Standard Photo)", 4, 3),
    ("3:4 (Portrait Standard)", 3, 4),
    ("3:2 (Landscape Photo)", 3, 2),
    ("2:3 (Portrait Photo)", 2, 3),
    ("21:9 (Cinematic Widescreen)", 21, 9),
    ("9:21 (Tall Cinematic)", 9, 21),
]
_TIERS = [("Small", 384), ("Medium", 576), ("Large — H3 default", 768),
          ("XL", 960), ("XXL", 1152)]

_RATIO_LABELS = [r[0] for r in _RATIOS] + ["Custom / manual"]


def _build_resolution_presets():
    seen = {}
    for label, rw, rh in _RATIOS:
        for tier_name, short_edge in _TIERS:
            w, h = _capped_canvas(rw, rh, short_edge)
            seen[f"{w}×{h}"] = True
    # legacy values observed in real saved workflows, kept selectable
    for legacy in ("384×576", "736×1120"):
        seen.setdefault(legacy, True)
    values = sorted(seen.keys(), key=lambda s: [int(x) for x in re.split("×", s)])
    return values


_RESOLUTION_PRESETS = _build_resolution_presets()
_DEFAULT_RESOLUTION = "1344×768" if "1344×768" in _RESOLUTION_PRESETS else _RESOLUTION_PRESETS[0]

# Backs web/minimax_h3_resolution_selector.js's aspect_ratio -> resolution
# filtering. The JS deliberately does NOT reimplement _capped_canvas's math
# itself -- it fetches this table so the two can never drift out of sync,
# same reasoning as image_hub.py's own /files route feeding its JS.
if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    try:
        @PromptServer.instance.routes.get("/minimax_h3_resolution_selector/presets")
        async def _minimax_h3_resolution_presets(request):
            table = {}
            for label, rw, rh in _RATIOS:
                row = {}
                for tier_name, short_edge in _TIERS:
                    w, h = _capped_canvas(rw, rh, short_edge)
                    row[tier_name] = f"{w}×{h}"
                table[label] = row
            return _aiohttp_web.json_response({
                "ratios": [r[0] for r in _RATIOS],
                "tiers": [t[0] for t in _TIERS],
                "table": table,
            })
    except Exception as exc:
        print(f"[MiniMaxH3ResolutionSelector] couldn't register /minimax_h3_resolution_selector/presets route: {exc}")


class MiniMaxH3ResolutionSelector:
    """Best-effort reimplementation -- see module docstring. Widget 2 (the
    literal 'W×H' string) is authoritative; widget 1 (aspect_ratio) is still
    never read by select() below -- it has no effect on this Python side at
    all. Its only job now is driving web/minimax_h3_resolution_selector.js,
    which filters `resolution`'s dropdown to that ratio's presets whenever
    aspect_ratio changes (fetching the exact table from the
    /minimax_h3_resolution_selector/presets route above, rather than
    reimplementing _capped_canvas's math in JS). Selecting "Custom / manual"
    restores the full unfiltered list. This never happens automatically on
    workflow load -- only on an actual aspect_ratio interaction -- so loading
    an older saved workflow never silently changes its resolution value.

    Two things below are adapted from the real, already-installed
    ResolutionSelectorZerohackz / ImageRatioSelectorZerohackz nodes rather
    than invented from scratch: the duration -> frame-count `length` output
    (their `snap_frames`, which is the same 17k+5 grid math as this file's
    own align_frame_count), and the optional `reference_image` input, which
    reuses their exact-ratio scale-stepping so you can size off an actual
    reference image instead of only a fixed named ratio."""

    DESCRIPTION = ("Best-effort reimplementation of a third-party workflow node "
                   "with no public source, with a couple of pieces borrowed from "
                   "the real ResolutionSelectorZerohackz pack (duration->length "
                   "frame snapping, exact-ratio-from-image sizing). Pick a preset "
                   "(aspect_ratio filters resolution's dropdown to that ratio's "
                   "sizes -- see web/minimax_h3_resolution_selector.js), wire a "
                   "reference_image to match its ratio instead, or set "
                   "custom_width/custom_height (>0) to override everything.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (_RATIO_LABELS, {"default": _RATIO_LABELS[1]}),
                "resolution": (_RESOLUTION_PRESETS, {"default": _DEFAULT_RESOLUTION}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 150.0, "step": 0.1,
                             "tooltip": "Seconds; snapped to MiniMax H3's frame grid for the `length` output."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 32,
                                          "tooltip": "0 = use the resolution preset above (or reference_image, if wired)."}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 32}),
                "reference_image": ("IMAGE", {"tooltip": "If wired, output the largest exact-ratio size that "
                                    "preserves this image's own W:H and fits the `resolution` preset's pixel "
                                    "budget -- overrides aspect_ratio/resolution (not custom_width/height)."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "INT")
    RETURN_NAMES = ("width", "height", "size", "length")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/custom"

    def select(self, aspect_ratio, resolution, duration, fps,
               custom_width=0, custom_height=0, reference_image=None):
        using_custom = custom_width > 0 and custom_height > 0
        if using_custom:
            w = max(CANVAS_MULTIPLE, round(custom_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            h = max(CANVAS_MULTIPLE, round(custom_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        else:
            m = re.match(r"^\s*(\d+)\s*×\s*(\d+)\s*$", resolution)
            if not m:
                raise ValueError(f"MiniMaxH3ResolutionSelector: unparseable resolution '{resolution}'")
            w, h = int(m.group(1)), int(m.group(2))

        # custom_width/custom_height are documented (see reference_image's own
        # tooltip above) to win over reference_image, not the other way around.
        if reference_image is not None and not using_custom:
            budget_mp = (w * h) / 1_000_000
            h_img, w_img = int(reference_image.shape[1]), int(reference_image.shape[2])
            scales = _exact_ratio_scales(w_img, h_img, budget_mp)
            if scales:
                _, w, h = scales[-1]  # largest exact-ratio step that still fits the budget

        length = align_frame_count(max(5, round(duration * fps)))
        return (w, h, f"{w}×{h}", length)


# ---------------------------------------------------------------------------
# MiniMaxH3ConcatAVLatent
#
# Not found in any public repo, but low-risk to rebuild: it just packs a
# plain video latent and a plain audio latent (as produced by ComfyUI's own
# VAEEncode / VAEEncodeAudio) into the same NestedTensor pairing that
# EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo
# already use for their av_latent output.
# ---------------------------------------------------------------------------

class MiniMaxH3ConcatAVLatent:
    DESCRIPTION = ("Best-effort reimplementation of a third-party workflow node "
                   "with no public source. Packs a plain video LATENT and a plain "
                   "audio LATENT into one MiniMax H3 AV NestedTensor latent, the same "
                   "structure EmptyMiniMaxH3LatentAV produces.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "concat"
    CATEGORY = "MiniMax H3/custom"

    @staticmethod
    def _unwrap(latent, expected_channels, name):
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            # someone fed an already-packed AV latent in by mistake; grab the
            # matching stream instead of failing outright
            for t in samples.tensors:
                if t.ndim >= 2 and t.shape[1] == expected_channels:
                    return t
            raise ValueError(f"MiniMaxH3ConcatAVLatent: '{name}' is a nested AV latent "
                              f"with no {expected_channels}-channel stream")
        if samples.ndim < 2 or samples.shape[1] != expected_channels:
            raise ValueError(f"MiniMaxH3ConcatAVLatent: '{name}' has shape {list(samples.shape)}, "
                              f"expected channel dim {expected_channels} (video=24, audio=32)")
        return samples

    def concat(self, video_latent, audio_latent):
        video = self._unwrap(video_latent, 24, "video_latent")
        audio = self._unwrap(audio_latent, 32, "audio_latent")
        return ({"samples": comfy.nested_tensor.NestedTensor((video, audio))},)


# ---------------------------------------------------------------------------
# MiniMaxH3AudioLock
#
# Not found in any public repo -- this is a genuine guess. Widget values seen:
# ["lock", 0.35]. Best-effort reading: encode the given reference audio via
# audio_vae, then write it into the target av_latent's audio stream, blended
# by `strength` (0 = leave target untouched, 1 = fully replace). `mode`
# controls how a shorter reference is fit to a longer target: "lock" tiles/
# loops it across the whole duration (for a consistent voice/style through
# the clip), "blend" only seeds the start and lets the rest stay at the
# target's original content. There is no way to verify this against the
# real node's actual behavior.
# ---------------------------------------------------------------------------

class MiniMaxH3AudioLock:
    DESCRIPTION = ("GUESS -- no public source exists for this node. Best-effort "
                   "reading of a third-party workflow node: re-encodes a reference "
                   "audio clip and blends it into the audio stream of `av_latent` at "
                   "`strength` (0=untouched, 1=fully replaced). 'lock' tiles the "
                   "reference across the whole clip, 'blend' only seeds the start. "
                   "The reference audio can be wired directly via `audio_1`, or "
                   "supplied through a `references` bundle from MiniMaxH3MediaLoader / "
                   "MiniMaxH3PromptBuilder (its first `audios` entry is used if "
                   "`audio_1` isn't connected) -- no MiniMaxH3ReferenceSplitter "
                   "required. Verify against your own renders before trusting this.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "mode": (["lock", "blend"], {"default": "lock"}),
                "strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "audio_1": ("AUDIO", {"tooltip": "Reference audio clip to lock/blend into the "
                                                   "av_latent's audio stream. Falls back to the "
                                                   "first entry of references.audios if not "
                                                   "connected directly."}),
                "references": ("H3_REFS", {"tooltip": "Optional references bundle from "
                                                        "MiniMaxH3MediaLoader/MiniMaxH3PromptBuilder. "
                                                        "Its first audios[] item is used as a fallback "
                                                        "when audio_1 isn't wired directly."}),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "STRING")
    RETURN_NAMES = ("av_latent", "audio", "report")
    FUNCTION = "lock"
    CATEGORY = "MiniMax H3/custom"

    def lock(self, av_latent, audio_vae, mode, strength, audio_1=None, references=None, **kwargs):
        values = _apply_dotted_kwargs({"audio_1": audio_1, "references": references}, kwargs)
        audio_1 = values["audio_1"]
        references = values["references"]

        bundle = references if isinstance(references, dict) else {}
        audio = _merge_direct_and_bundle([audio_1], bundle.get("audios"), 1)[0]
        if audio is None:
            raise ValueError(
                "MiniMaxH3AudioLock: no reference audio was supplied -- wire a clip into "
                "audio_1, or connect a references bundle whose audios[] is non-empty. "
                "(Tip: if you're feeding this from a MiniMaxH3ReferenceSplitter, make sure "
                "you used its audio_1 output, not an empty video_audio_1 slot.)"
            )

        samples = av_latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise ValueError("MiniMaxH3AudioLock expects a MiniMax H3 AV latent (video+audio NestedTensor)")
        video, target_audio = samples.tensors
        if video.ndim != 5 or video.shape[1] != 24 or target_audio.shape[1] != 32:
            raise ValueError("MiniMaxH3AudioLock: av_latent doesn't look like a MiniMax H3 AV latent")

        ref_latent, ref_t = _encode_ref_audio(audio_vae, audio)
        target_t = target_audio.shape[-1]

        if ref_t == 0 or target_t == 0:
            new_audio = target_audio.clone()
        else:
            if mode == "lock":
                reps = -(-target_t // ref_t)  # ceil div, tile to cover the whole clip
                fitted = ref_latent.repeat(1, 1, 1, reps)[..., :target_t]
            else:  # "blend": seed only the start, leave the rest as the target had it
                fitted = target_audio.clone()
                n = min(ref_t, target_t)
                fitted[..., :n] = ref_latent[..., :n]
            new_audio = target_audio * (1.0 - strength) + fitted.to(target_audio.dtype) * strength

        new_samples = comfy.nested_tensor.NestedTensor((video, new_audio))
        report = (f"MiniMaxH3AudioLock (best-effort): mode={mode} strength={strength:.2f} "
                  f"ref_audio_t={ref_t} target_audio_t={target_t}. "
                  f"This node's exact original behavior is unverified -- inspect the render.")
        return ({"samples": new_samples}, audio, report)


# ---------------------------------------------------------------------------
# MiniMaxH3UnifiedToVideo
#
# Not found in any public repo. Rebuilt on top of the real native building
# blocks (EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo
# logic, copied verbatim above) rather than guessed from nothing, so this is
# lower-risk than AudioLock even though the exact merge behavior is inferred.
#
# Design, inferred from the workflow's own saved prompt text (which tags
# first_frame as "<Picture 1>" and a reference audio as "<Audio 1>" in the
# SAME prompt): unlike the native nodes, which use two separate mechanisms
# (plain "images=" for keyframes vs. "<Picture N>" ref tags for references),
# this unified node puts *everything* -- first_frame, last_frame, ref images,
# ref video(+its audio), ref audio -- into one tag-numbered presentation
# (pictures, then videos+their paired audio, then standalone audio), while
# first_frame/last_frame ALSO get anchored as literal keyframes the way
# MiniMaxH3ImageToVideo does. `mode` lets you force-disable one pathway;
# "auto" just uses whatever is wired.
#
# Reference socket naming and the `references` bundle input below are
# deliberately matched to the real, separately-installed "Fantastic
# MiniMax H3 Prompt Builder" pack (ComfyUI-Fantastic-MiniMaxH3-PromptBuilder,
# nodes.py): its MiniMaxH3ReferenceSplitter outputs picture_1..9, video_1..3,
# video_audio_1..3, audio_1..3 (see its `_media_names()`), and both its
# MiniMaxH3MediaLoader and MiniMaxH3PromptBuilder emit a plain dict under the
# type name "H3_REFS" shaped {"pictures": [...], "videos": [...],
# "video_audios": [...], "audios": [...]}. Reusing that exact type string and
# dict shape (no import needed -- ComfyUI sockets match by type string alone)
# lets MediaLoader's or PromptBuilder's `references` output wire straight
# into this node's `references` input instead of needing a Reference
# Splitter in between. Direct sockets are only kept for the first few slots
# of each group (picture_1-4, video_1, video_audio_1, audio_1-2) to avoid
# turning this node into a wall of near-identical sockets; wire a
# `references` bundle instead for the fuller 9/3/3/3 capacity that pack
# supports. Per-slot, a direct wire always wins over the bundle.
# ---------------------------------------------------------------------------

H3_REF_PICTURES = 9
H3_REF_VIDEOS = 3
H3_REF_VIDEO_AUDIOS = 3
H3_REF_AUDIOS = 3


def _merge_direct_and_bundle(direct_values, bundle_seq, count):
    """Build a `count`-long list where each slot prefers its own directly-
    wired value and falls back to the matching item from a references
    bundle sequence, mirroring MiniMaxH3PromptBuilder.build()'s own
    "own input, else matching bundle item" fallback in the real pack this
    node interoperates with. `direct_values` may be shorter than `count`
    (this node only exposes a handful of direct sockets per group)."""
    out = []
    for i in range(count):
        value = direct_values[i] if i < len(direct_values) else None
        if value is None and bundle_seq is not None and i < len(bundle_seq):
            value = bundle_seq[i]
        out.append(value)
    return out


class MiniMaxH3UnifiedToVideo:
    DESCRIPTION = ("Best-effort reimplementation of a third-party workflow node "
                   "with no public source, built from ComfyUI's real native MiniMax H3 "
                   "building blocks. Combines EmptyMiniMaxH3LatentAV + MiniMaxH3ImageToVideo "
                   "+ MiniMaxH3ReferenceToVideo into one node, tagging first_frame/last_frame "
                   "and every reference into one <Picture N>/<Video N>/<Audio N> sequence. "
                   "Reference inputs are named to match the real, already-installed "
                   "MiniMaxH3ReferenceSplitter's output labels (picture_N/video_N/"
                   "video_audio_N/audio_N). Wire an H3_REFS `references` bundle straight from "
                   "MiniMaxH3MediaLoader or MiniMaxH3PromptBuilder instead of using a Reference "
                   "Splitter -- direct sockets win per-slot when both are wired, the bundle "
                   "fills in the rest (up to 9 pictures / 3 videos / 3 paired video_audios / "
                   "3 standalone audios).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "mode": (["auto", "keyframe", "reference", "hybrid", "text"], {"default": "auto",
                          "tooltip": "auto/hybrid = use whatever is wired. keyframe = ignore ref_* "
                                     "inputs. reference = ignore first_frame/last_frame. text = ignore all media."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 150.0, "step": 0.1,
                             "tooltip": "Seconds; converted to a frame count via fps and snapped to H3's grid."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "ref_image_size": (["match", "max"], {"default": "match",
                                    "tooltip": "match = scale refs to the generation's pixel area. "
                                               "max = use the 2048px reference pipeline (slower, higher fidelity)."}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "references": ("H3_REFS", {"tooltip": "Wire this straight from MiniMaxH3MediaLoader's "
                               "or MiniMaxH3PromptBuilder's 'references' output instead of using a "
                               "Reference Splitter. Fills in whichever picture_N/video_N/video_audio_N/"
                               "audio_N slot below isn't directly wired (up to 9 pictures / 3 videos / "
                               "3 paired video_audios / 3 standalone audios)."}),
                "picture_1": ("IMAGE",),
                "picture_2": ("IMAGE",),
                "picture_3": ("IMAGE",),
                "picture_4": ("IMAGE",),
                "video_1": ("IMAGE",),
                "video_audio_1": ("AUDIO",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "av_latent", "conditioned_prompt", "media_map_json", "report")
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3/custom"

    def execute(self, clip, video_vae, prompt, mode, width, height, duration, fps, ref_image_size,
                audio_vae=None, first_frame=None, last_frame=None, references=None,
                picture_1=None, picture_2=None, picture_3=None, picture_4=None,
                video_1=None, video_audio_1=None, audio_1=None, audio_2=None,
                **kwargs):
        # See _apply_dotted_kwargs -- some ComfyUI frontends submit
        # picture_1..4 / audio_1..2 (and possibly others) under a dotted
        # group key instead of the plain name below. Recover them.
        if kwargs:
            values = _apply_dotted_kwargs({
                "audio_vae": audio_vae, "first_frame": first_frame, "last_frame": last_frame,
                "references": references,
                "picture_1": picture_1, "picture_2": picture_2,
                "picture_3": picture_3, "picture_4": picture_4,
                "video_1": video_1, "video_audio_1": video_audio_1,
                "audio_1": audio_1, "audio_2": audio_2,
            }, kwargs)
            audio_vae, first_frame, last_frame = values["audio_vae"], values["first_frame"], values["last_frame"]
            references = values["references"]
            picture_1, picture_2 = values["picture_1"], values["picture_2"]
            picture_3, picture_4 = values["picture_3"], values["picture_4"]
            video_1, video_audio_1 = values["video_1"], values["video_audio_1"]
            audio_1, audio_2 = values["audio_1"], values["audio_2"]

        use_keyframe = mode in ("auto", "keyframe", "hybrid")
        use_reference = mode in ("auto", "reference", "hybrid")
        if mode == "text":
            use_keyframe = use_reference = False

        first_frame = first_frame if use_keyframe else None
        last_frame = last_frame if use_keyframe else None
        if not use_reference:
            picture_1 = picture_2 = picture_3 = picture_4 = None
            video_1 = video_audio_1 = None
            audio_1 = audio_2 = None
            references = None

        bundle = references if isinstance(references, dict) else {}
        pictures = _merge_direct_and_bundle(
            [picture_1, picture_2, picture_3, picture_4], bundle.get("pictures"), H3_REF_PICTURES)
        videos_in = _merge_direct_and_bundle([video_1], bundle.get("videos"), H3_REF_VIDEOS)
        video_audios_in = _merge_direct_and_bundle(
            [video_audio_1], bundle.get("video_audios"), H3_REF_VIDEO_AUDIOS)
        audios_in = _merge_direct_and_bundle([audio_1, audio_2], bundle.get("audios"), H3_REF_AUDIOS)

        length_frames = max(5, round(duration * fps))
        latent, frame_count = _empty_av_latent(width, height, length_frames)

        ref_items = []
        ref_blocks = []
        keyframes = []
        tag_order = []
        warnings = []
        audio_tag_n = 0
        video_tag_n = 0

        # Pictures: first_frame, last_frame, then picture_1..9 (direct sockets
        # for 1-4, the rest only reachable via a `references` bundle), all
        # sharing one <Picture N> numbering pool.
        if first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            keyframes.append({"resolved_frame_index": 0, "latent": video_vae.encode(img)})
            ref_items.append({"type": "image", "data": img})
            tag_order.append(f"<Picture {len(ref_items)}> = first_frame")
        if last_frame is not None:
            # "disabled" (no crop) to match first_frame and every other resize
            # in this node -- center-cropping just this one input would lose
            # edge content that first_frame keeps for the same mismatched-ratio case.
            img = _resize(last_frame[:1], width, height, "disabled")
            keyframes.append({"resolved_frame_index": frame_count - 1, "latent": video_vae.encode(img)})
            ref_items.append({"type": "image", "data": img})
            tag_order.append(f"<Picture {len(ref_items)}> = last_frame")

        for pic_idx, ref_img in enumerate(pictures):
            if ref_img is None:
                continue
            h, w = ref_img.shape[1], ref_img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(ref_img[:1], tw, th, "disabled")
            z = video_vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
            tag_order.append(f"<Picture {len(ref_items)}> = picture_{pic_idx + 1}")

        # Videos (+ each one's own paired audio, tagged right before it).
        for vid_idx, vid in enumerate(videos_in):
            if vid is None:
                continue
            vh, vw = vid.shape[1], vid.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(vid, cw, ch, "disabled")
            orig_frame_count = frames.shape[0]
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError(f"MiniMaxH3UnifiedToVideo: video_{vid_idx + 1} needs at least "
                                  f"5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            if frames.shape[0] < orig_frame_count:
                warnings.append(
                    f"video_{vid_idx + 1}: trimmed from {orig_frame_count} to {frames.shape[0]} frames "
                    f"to fit the {frame_count}-frame output and H3's frame grid")
            z = video_vae.encode(frames)

            vid_audio = video_audios_in[vid_idx] if vid_idx < len(video_audios_in) else None
            audio_latent, ref_audio_t = (None, 0)
            if vid_audio is not None:
                if audio_vae is None:
                    raise ValueError(f"MiniMaxH3UnifiedToVideo: video_audio_{vid_idx + 1} needs audio_vae")
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, vid_audio)
                ref_items.append({"type": "audio"})
                audio_tag_n += 1
                tag_order.append(f"<Audio {audio_tag_n}> = video_audio_{vid_idx + 1}")

            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [t / 2.0 for t in range(len(sample_idx))]})
            video_tag_n += 1
            tag_order.append(f"<Video {video_tag_n}> = video_{vid_idx + 1}")
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        # Standalone audio.
        for aud_idx, aud in enumerate(audios_in):
            if aud is None:
                continue
            if audio_vae is None:
                raise ValueError(f"MiniMaxH3UnifiedToVideo: audio_{aud_idx + 1} needs audio_vae")
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, aud)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
            audio_tag_n += 1
            tag_order.append(f"<Audio {audio_tag_n}> = audio_{aud_idx + 1}")

        if ref_items:
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        else:
            tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)

        extra_vals = {}
        if keyframes:
            extra_vals["minimax_keyframes"] = keyframes
        if ref_blocks:
            extra_vals["minimax_refs"] = ref_blocks
        if extra_vals:
            cond = node_helpers.conditioning_set_values(cond, extra_vals)

        media_map = {
            "mode_requested": mode,
            "has_first_frame": first_frame is not None,
            "has_last_frame": last_frame is not None,
            "has_references_bundle": bool(bundle),
            "ref_image_count": sum(1 for p in pictures if p is not None),
            "ref_video_count": sum(1 for v in videos_in if v is not None),
            "ref_audio_count": sum(1 for a in audios_in if a is not None)
                                + sum(1 for a in video_audios_in if a is not None),
            "tag_order": tag_order,
            "warnings": warnings,
            "frame_count": frame_count,
            "duration_s": duration,
            "fps": fps,
            "width": width,
            "height": height,
        }
        report = (f"MiniMaxH3UnifiedToVideo (best-effort): {frame_count} frames @ {width}x{height}, "
                 f"{len(tag_order)} tagged reference(s){' [references bundle wired]' if bundle else ''}: "
                 f"{', '.join(tag_order) if tag_order else 'none'}.")
        if warnings:
            report += " Warnings: " + "; ".join(warnings) + "."

        return (cond, latent, prompt, json.dumps(media_map, ensure_ascii=False), report)


# ---------------------------------------------------------------------------
# MiniMaxH3MultimodalChat (local LLM edition)
#
# The original node called a cloud API (endpoint redacted to "https://api/v1",
# model "gemini-3.6-flash") to help write MiniMax H3 style prompts through
# conversation, and was disabled (mode 4) in the source workflow anyway. This
# is a fresh, purpose-built replacement that talks to a locally-hosted model
# through Ollama, LM Studio, or KoboldCpp instead -- it is not a drop-in
# replacement for the original node's saved widget values.
#
# The overall structure/tagging below is reconstructed from the real assistant
# replies saved inside that workflow's own chat_history widget (not invented
# from nothing). The dialogue-handling rules (speaker IDs, <d> tags, verbatim
# preservation, off-screen voiceover, <scenetrans>/<cutoff>) are instead taken
# directly from Video_Prompt_Writing_Guide.pdf, bundled with the real,
# separately-installed "Fantastic H3 Prompt Builder" pack (its
# web/Video_Prompt_Writing_Guide.pdf, sections 1.4.4-1.4.6) -- that PDF is
# MiniMax's actual H3 prompt-writing spec, not a guess, so treat those rules
# as ground truth and re-check that file if MiniMax revises the format.
# ---------------------------------------------------------------------------

_DEFAULT_H3_SYSTEM_PROMPT = """You write prompts for the MiniMax H3 video model inside ComfyUI.

Any reference image, video, or audio clip the user has attached is numbered in a fixed
order and referred to by tag: pictures first (<Picture 1>, <Picture 2>, ...), then each
video with its own soundtrack tagged as an audio reference immediately before it
(<Audio N> then <Video N>), then standalone audio clips last. Only use tags for media that
was actually attached this turn -- never invent a tag for something that isn't there.

Write the finished prompt as plain, concrete, shot-by-shot description in English,
regardless of what language the user writes in. For a simple single-shot request, use:

    <description of the scene, action, camera movement, referencing tags where relevant>

    overall_soundscape: <ambient sound, physical/action sounds, non-verbal human sounds>

    non_diegetic_music: <score/music style, or "None" if there shouldn't be any>

For a multi-shot request, prefix each shot with [Shot N] and, from the second shot
onward, a cut time like "At 00:05.000,", keeping every cut time inside the target
duration and strictly increasing.

If the user's message includes any spoken dialogue -- quoted lines, a script, anything a
character is meant to say or sing -- you MUST carry it into the description verbatim.
Never summarize, paraphrase, drop, or move dialogue the user gave you into
overall_soundscape; it belongs in the main description. Give each speaking subject a
stable ID like (S1), (S2), reused for that same subject in every later shot it speaks in;
use a compound ID like (S1,S2) when subjects speak or sing together. Write it as: a short
identifying phrase, the ID, the speaking/singing action, and the delivery style OUTSIDE
the tag, then the line itself inside <d>...</d> containing ONLY a language tag and the
exact original words and punctuation -- e.g. "the man with a low, steady voice (S1) says:
<d>[English] I get off at the next station.</d>". Do not translate, rewrite, or trim what's
inside <d>; write [unclear] instead of guessing at anything inaudible. For an off-screen
voiceover, use the exact phrase "says in an off-screen voiceover" and immediately state
that the on-screen character's lips remain closed. If a line of dialogue continues across
a shot cut, add <scenetrans> in both shots and say the audio continues across the cut; if
speech is cut off by the video ending, add <cutoff> instead.

If the user's message includes on-screen text -- a sign, subtitle, or banner that should
actually be visible in the shot -- put it in plain double quotation marks in the
description, verbatim and untranslated; this is separate from spoken dialogue and never
goes in overall_soundscape or non_diegetic_music either.

If the user is referencing a picture as the literal first or last frame of the video
(not just a style/identity reference), say so plainly, e.g. "Starting from <Picture 1>,
...". If they're driving lip-sync/performance timing from a reference audio clip, say the
performance is synchronized to that <Audio N> track.

Keep the whole prompt tightly scoped to the stated duration -- don't describe more action
than fits.

Always wrap ONLY the finished, ready-to-use H3 prompt in a fenced ```text block at the end
of your reply. You can explain your reasoning before that block, but the fenced block must
contain nothing except the prompt itself, since it's parsed back out automatically.
"""

_BACKEND_DEFAULTS = {
    "ollama": "http://127.0.0.1:11434",
    "lmstudio": "http://127.0.0.1:1234",
    "kobold": "http://127.0.0.1:5001",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    # Google's OpenAI-compatibility layer -- same request/response shape as
    # real OpenAI, just a different base URL and API key. Verified against
    # https://ai.google.dev/gemini-api/docs/openai -- note the endpoint path
    # is .../openai/chat/completions, NOT .../openai/v1/chat/completions, so
    # this backend uses a different endpoint_path than openai/lmstudio/kobold
    # (see _call_openai_compatible's endpoint_path parameter).
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

# The env var each cloud backend's API key is read from when the widget field
# is left blank -- matches the name each provider's own official SDK looks
# for, so a key already exported for other tools works here for free.
_ENV_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _resolve_api_key(backend, widget_key):
    """(key, source) where source is 'widget' (typed directly -- works, but
    gets saved in plaintext inside the workflow JSON) or 'env' (the
    recommended path), or (None, None) if neither is set. An explicitly
    typed widget value always wins over the environment variable, since
    typing one in is a deliberate choice that shouldn't be silently
    overridden."""
    widget_key = (widget_key or "").strip()
    if widget_key:
        return widget_key, "widget"
    env_name = _ENV_KEY_NAMES.get(backend)
    if env_name:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return env_val, "env"
    return None, None

# Matches MiniMaxH3MultimodalChat.DESCRIPTION's stated "up to 2 pictures + 1
# video frame" cap -- keeps local-LLM context/latency bounded regardless of
# how many optional picture sockets are wired.
MAX_CHAT_PICTURES = 2

# -- "comfyui_clip" backend: system prompts per detected model family --------
#
# ComfyUI can already run real text generation directly off a loaded CLIP/text
# encoder for the LLM-based families it uses as conditioning (Qwen3-VL, Qwen3,
# Gemma, etc.) -- see comfy/text_encoders/llama.py's BaseGenerate mixin and the
# core comfy_extras/nodes_textgen.py TextGenerate node, which does exactly
# this. Reusing an already-loaded CLIP this way costs no extra VRAM/process.

_KREA2_SYSTEM_PROMPT = """You write image-generation prompts for the Krea 2 model.

Krea 2 was trained to condition on prompts written as thorough visual descriptions --
its own internal captioning instruction is: "describe the image by detailing the color,
shape, size, texture, quantity, text, spatial relationships of the objects and background."
Follow that same style when expanding the user's request.

Write ONE descriptive paragraph, not a list or a shot script. Cover, in whatever order
reads most naturally: the main subject(s) and their color/shape/size/texture, any text
that should appear in the image (quote it exactly), how many of anything there are, and
the spatial relationship between objects and the background. Be concrete and visual --
avoid mood words that don't translate into something the model can render (e.g. "epic,"
"beautiful") unless the user explicitly asked for that vocabulary.

If the user's request is already a full, detailed description, only tighten or reorganize
it into this style rather than inventing unrelated new content. If it's short or vague,
invent concrete, plausible visual details to fill it out.

Output ONLY the finished prompt as plain text -- no headings, no quotes around the whole
thing, no explanation before or after.
"""

_KLEIN_SYSTEM_PROMPT = """You write image-generation prompts for a FLUX.2 "Klein" checkpoint.

NOTE: unlike the Krea 2 prompt in this node, this system prompt is a generic best-effort
default, not grounded in an official Klein captioning spec -- edit it if you know the
style Klein was actually trained on.

Expand the user's request into one detailed, concrete visual description suitable for a
text-to-image model: subject, pose/action, setting, lighting, color palette, camera
framing, and any text that should appear in the image (quote it exactly). Write it as
flowing prose, not a bullet list or tag string. Stay faithful to everything the user
actually asked for -- add plausible concrete detail to fill gaps, but don't contradict or
drop anything they specified.

Output ONLY the finished prompt as plain text -- no headings, no quotes around the whole
thing, no explanation before or after.
"""

_ZIMAGE_SYSTEM_PROMPT = """You write image-generation prompts for a Z-Image checkpoint.

NOTE: like the Klein prompt in this node, this is a generic best-effort default, not
grounded in an official Z-Image captioning spec -- edit it if you know the style Z-Image
was actually trained on.

Expand the user's request into one detailed, concrete visual description suitable for a
text-to-image model: subject, pose/action, setting, lighting, color palette, camera
framing, and any text that should appear in the image (quote it exactly). Write it as
flowing prose, not a bullet list or tag string. Stay faithful to everything the user
actually asked for -- add plausible concrete detail to fill gaps, but don't contradict or
drop anything they specified.

Output ONLY the finished prompt as plain text -- no headings, no quotes around the whole
thing, no explanation before or after.
"""

# Keyed by the exact Python class name of clip.tokenizer -- verified against
# ComfyUI's own comfy/text_encoders/*.py source. These are the only families
# in this ComfyUI build with BOTH a generate()-capable underlying LLM AND a
# tokenizer whose tokenize_with_weights() accepts a llama_template override
# (needed to inject our own system prompt). MiniMax H3's own tokenizer
# (MiniMaxH3Tokenizer) is deliberately NOT included here: its
# tokenize_with_weights() has a completely different, H3-specific signature
# (minimax_ref_items=...) with no llama_template support at all, so it can't
# be driven as a general chat model this way -- use the other backends above
# for H3 prompt-writing instead.
_CLIP_CHAT_FAMILIES = {
    "Krea2Tokenizer": ("Krea2", _KREA2_SYSTEM_PROMPT),
    "KleinTokenizer": ("Klein", _KLEIN_SYSTEM_PROMPT),
    "KleinTokenizer8B": ("Klein", _KLEIN_SYSTEM_PROMPT),
    "ZImageTokenizer": ("Z-Image", _ZIMAGE_SYSTEM_PROMPT),
}


def _detect_clip_family(clip):
    """Returns (family_name, system_prompt) if clip.tokenizer's class is a
    recognized, generate()-capable family, else None."""
    tokenizer = getattr(clip, "tokenizer", None)
    return _CLIP_CHAT_FAMILIES.get(type(tokenizer).__name__)


def _chatml_template(system_prompt):
    """ChatML markup shared by every family in _CLIP_CHAT_FAMILIES (all are
    Qwen-based instruct models). Braces in system_prompt are escaped since
    the tokenizer fills the single remaining {} via str.format()."""
    escaped = system_prompt.replace("{", "{{").replace("}", "}}")
    return f"<|im_start|>system\n{escaped}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"


def _tensor_to_png_b64(image_bt, max_edge=1024):
    """image_bt: [B,H,W,C] float 0..1 tensor, first frame only, PNG base64 (no data-uri prefix)."""
    from PIL import Image
    import numpy as np

    frame = image_bt[0].clamp(0, 1).cpu().numpy()
    h, w = frame.shape[0], frame.shape[1]
    scale = min(1.0, max_edge / max(h, w))
    img = Image.fromarray((frame * 255.0).round().astype("uint8"))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class MiniMaxH3MultimodalChat:
    DESCRIPTION = ("Local-LLM prompt-writing chat, talking to Ollama, LM Studio, KoboldCpp, or "
                   "(new) directly to a CLIP/text-encoder already loaded elsewhere in this workflow "
                   "-- see `backend`. Not a drop-in replacement for the original third-party workflow "
                   "node (which called a cloud API and was disabled in the source workflow anyway) -- "
                   "a fresh build. Reference sockets are named to match MiniMaxH3ReferenceSplitter's "
                   "output labels (picture_N/video_N/video_audio_N/audio_N -- see "
                   "MiniMaxH3UnifiedToVideo's docstring above), and a `references` H3_REFS bundle can "
                   "fill them in from MiniMaxH3MediaLoader/MiniMaxH3PromptBuilder. For the "
                   "ollama/lmstudio/kobold backends this node shows the model up to 2 pictures + 1 "
                   "video frame regardless of source, to keep local-LLM context/latency sane; the "
                   "comfyui_clip backend doesn't send images/video/audio at all yet (text only) -- "
                   "attached media is still noted in the message text either way.\n\n"
                   "`backend: comfyui_clip` runs generation directly on a `clip` you wire in, using "
                   "ComfyUI's own native model.generate() (see comfy/text_encoders/llama.py's "
                   "BaseGenerate and the core 'Generate Text' node) -- no separate server, no extra "
                   "copy of the model in VRAM, and it participates in ComfyUI's normal model "
                   "load/offload management. `auto_system_prompt` picks a system prompt matching the "
                   "detected model family (Krea2, Klein, Z-Image currently -- MiniMax H3's own CLIP "
                   "isn't supported this way, its tokenizer has an incompatible interface; use the "
                   "other backends for H3 prompt-writing) via `clip.tokenizer`'s exact class name; "
                   "turn it off to use `system_prompt` verbatim instead, which is also required for "
                   "any unrecognized family.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["ollama", "lmstudio", "kobold", "comfyui_clip", "openai", "anthropic", "gemini"],
                             {"default": "ollama",
                             "tooltip": "comfyui_clip runs generation on the `clip` input below instead "
                                        "of calling out to a server -- see DESCRIPTION. openai/anthropic/"
                                        "gemini need an API key -- see the matching *_api_key input below "
                                        "(environment variable preferred over typing it in directly)."}),
                "base_url": ("STRING", {"default": "", "tooltip": "Leave blank to use the backend's usual "
                             "address (Ollama 11434, LM Studio 1234, KoboldCpp 5001, or the real "
                             "OpenAI/Anthropic/Gemini API). Unused for comfyui_clip."}),
                "model": ("STRING", {"default": "", "tooltip": "Model name/tag as loaded in your backend, "
                          "e.g. 'llama3.1' (Ollama), whatever's currently loaded (LM Studio/Kobold), or "
                          "the provider's model id (e.g. 'gpt-5.1', 'claude-opus-4-5', 'gemini-3-pro') for "
                          "openai/anthropic/gemini. Unused for comfyui_clip -- wire `clip` instead."}),
                "system_prompt": ("STRING", {"multiline": True, "default": _DEFAULT_H3_SYSTEM_PROMPT,
                                  "tooltip": "Ignored for comfyui_clip when auto_system_prompt is on."}),
                "user_message": ("STRING", {"multiline": True, "default": ""}),
                "new_chat": ("BOOLEAN", {"default": False, "tooltip": "Clear chat_history before this turn."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 16, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed_mode": (["fixed", "random"], {"default": "fixed"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 150.0, "step": 0.1,
                             "tooltip": "Told to the model as the target video length, for pacing."}),
            },
            "optional": {
                "chat_history": ("STRING", {"multiline": True, "default": "[]",
                                  "tooltip": "JSON [{role,content}, ...]. Feed the 'chat_history' output "
                                             "back in here (e.g. via a Get/Set node loop) for multi-turn memory."}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "references": ("H3_REFS", {"tooltip": "Wire this from MiniMaxH3MediaLoader's or "
                               "MiniMaxH3PromptBuilder's 'references' output instead of using a Reference "
                               "Splitter. Fills in picture_1/2, video_1, video_audio_1, audio_1/2 below "
                               "when they aren't directly wired -- anything past that in the bundle isn't "
                               "sent to the local model (see DESCRIPTION)."}),
                "picture_1": ("IMAGE",),
                "picture_2": ("IMAGE",),
                "video_1": ("IMAGE",),
                "video_audio_1": ("AUDIO",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "unload_after_generating": ("BOOLEAN", {"default": False, "tooltip": "ollama: adds "
                                            "keep_alive=0 so it unloads right after replying. lmstudio: "
                                            "looks up and calls its /api/v1/models/unload endpoint "
                                            "(best-effort -- see lmstudio_api_key). comfyui_clip: calls "
                                            "ComfyUI's own model-unload function on the wired clip. kobold: "
                                            "not automated -- its unload API isn't consistently documented "
                                            "across versions; use KoboldCpp's own --admin/--adminunloadtimeout "
                                            "flags instead."}),
                "lmstudio_api_key": ("STRING", {"default": "", "tooltip": "Only used to authorize the "
                                     "unload call above when backend=lmstudio and LM Studio has an API "
                                     "key configured (it does by default)."}),
                "openai_api_key": ("STRING", {"default": "", "tooltip": "Only used when backend=openai. "
                                   "WARNING: typed here, this is saved in PLAIN TEXT inside the workflow "
                                   "JSON -- anyone you share, screenshot, or commit that file to sees it. "
                                   "Prefer setting the OPENAI_API_KEY environment variable and leaving "
                                   "this blank; it's used automatically when this field is empty."}),
                "anthropic_api_key": ("STRING", {"default": "", "tooltip": "Only used when "
                                      "backend=anthropic. WARNING: typed here, this is saved in PLAIN "
                                      "TEXT inside the workflow JSON -- anyone you share, screenshot, or "
                                      "commit that file to sees it. Prefer setting the ANTHROPIC_API_KEY "
                                      "environment variable and leaving this blank; it's used "
                                      "automatically when this field is empty."}),
                "gemini_api_key": ("STRING", {"default": "", "tooltip": "Only used when backend=gemini. "
                                   "WARNING: typed here, this is saved in PLAIN TEXT inside the workflow "
                                   "JSON -- anyone you share, screenshot, or commit that file to sees it. "
                                   "Prefer setting the GEMINI_API_KEY environment variable and leaving "
                                   "this blank; it's used automatically when this field is empty."}),
                "clip": ("CLIP", {"tooltip": "Required when backend=comfyui_clip. Wire this from any "
                         "CLIPLoader already in your workflow -- e.g. a Krea2/Klein/Z-Image checkpoint's "
                         "text encoder -- to draft prompts using that same, already-loaded model instead "
                         "of a separate local/cloud LLM."}),
                "auto_system_prompt": ("BOOLEAN", {"default": True, "tooltip": "comfyui_clip only: pick "
                                       "the system prompt automatically from the wired clip's detected "
                                       "model family (see DESCRIPTION). Off = use system_prompt verbatim "
                                       "(required for a family this node doesn't recognize)."}),
                "do_sample": ("BOOLEAN", {"default": True, "tooltip": "comfyui_clip only: sample from the "
                              "distribution (temperature/top_k/top_p/min_p apply) vs. always taking the "
                              "single most likely next token."}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000, "tooltip": "comfyui_clip only."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "comfyui_clip only."}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "comfyui_clip only."}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01,
                                       "tooltip": "comfyui_clip only."}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01,
                                     "tooltip": "comfyui_clip only."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("reply", "prompt_text", "chat_history", "report")
    FUNCTION = "chat"
    CATEGORY = "MiniMax H3/custom"

    @classmethod
    def VALIDATE_INPUTS(cls, backend, openai_api_key="", anthropic_api_key="", gemini_api_key=""):
        # Hard-blocks the queue with a specific, actionable message rather
        # than letting it run and fail deep inside an HTTP call -- there's
        # genuinely nothing this node can do for these backends without a key.
        key_widgets = {"openai": openai_api_key, "anthropic": anthropic_api_key, "gemini": gemini_api_key}
        if backend in key_widgets:
            key, _source = _resolve_api_key(backend, key_widgets[backend])
            if key is None:
                env_name = _ENV_KEY_NAMES[backend]
                return (f"MiniMaxH3MultimodalChat: backend={backend} needs an API key -- set the "
                        f"{env_name} environment variable, or fill in the {backend}_api_key field.")
        return True

    # -- backend request/response plumbing ---------------------------------

    def _post_json(self, url, payload, timeout=180, headers=None):
        data = json.dumps(payload).encode("utf-8")
        all_headers = {"Content-Type": "application/json"}
        if headers:
            all_headers.update(headers)
        req = urllib.request.Request(url, data=data, headers=all_headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call_ollama(self, base_url, model, messages, images_b64, temperature, max_tokens, seed, unload=False):
        msgs = [dict(m) for m in messages]
        if images_b64 and msgs and msgs[-1]["role"] == "user":
            msgs[-1] = dict(msgs[-1])
            msgs[-1]["images"] = images_b64
        payload = {
            "model": model, "messages": msgs, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        }
        if unload:
            # Ollama-specific: keep_alive=0 on this same request unloads the
            # model right after it replies. Documented Ollama behavior, not a
            # guess -- see https://github.com/ollama/ollama/blob/main/docs/api.md
            payload["keep_alive"] = 0
        result = self._post_json(base_url.rstrip("/") + "/api/chat", payload)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("message", {}).get("content", "")

    def _call_openai_compatible(self, base_url, model, messages, images_b64, temperature, max_tokens, seed,
                                api_key=None, endpoint_path="/v1/chat/completions"):
        msgs = [dict(m) for m in messages]
        if images_b64 and msgs and msgs[-1]["role"] == "user":
            content = [{"type": "text", "text": msgs[-1]["content"]}]
            for b64 in images_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            msgs[-1] = {"role": "user", "content": content}
        payload = {
            "model": model, "messages": msgs, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed, "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        result = self._post_json(base_url.rstrip("/") + endpoint_path, payload, headers=headers)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result["choices"][0]["message"]["content"]

    def _call_anthropic(self, base_url, model, system_prompt, messages, images_b64, temperature, max_tokens, api_key):
        """Anthropic's native Messages API -- NOT the OpenAI-compatible shape
        _call_openai_compatible uses. Verified against
        https://platform.claude.com/docs/en/api/messages: system prompt is a
        separate top-level field (not a 'system' role message), image blocks
        use {"type": "image", "source": {...}}, and there's no request-level
        seed/determinism control at all, unlike the other backends."""
        msgs = [dict(m) for m in messages if m.get("role") != "system"]
        if images_b64 and msgs and msgs[-1]["role"] == "user":
            content = [{"type": "text", "text": msgs[-1]["content"]}]
            for b64 in images_b64:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                              "data": b64}})
            msgs[-1] = {"role": "user", "content": content}
        payload = {
            "model": model, "messages": msgs, "system": system_prompt,
            "max_tokens": max_tokens, "temperature": temperature,
        }
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        result = self._post_json(base_url.rstrip("/") + "/v1/messages", payload, headers=headers)
        if "error" in result:
            err = result["error"]
            raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        text_blocks = [b.get("text", "") for b in result.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(text_blocks)

    def _unload_lmstudio(self, base_url, model, api_key):
        """Best-effort: LM Studio's /api/v1/models/unload takes an instance_id,
        not the plain model name, so this looks it up via GET /api/v1/models
        first. Returns None on success, or a short string describing what
        went wrong -- never raises, since this always runs after a chat reply
        has already succeeded and a failed unload shouldn't fail the node."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/api/v1/models", headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return f"couldn't list LM Studio models to find an instance to unload: {exc}"

        instances = data.get("loaded_instances") or data.get("data") or []
        instance_id = None
        for inst in instances:
            inst_id = inst.get("id") or inst.get("modelKey") or inst.get("model")
            if inst_id and (inst_id == model or str(inst_id).startswith(model + ":")):
                instance_id = inst_id
                break
        if instance_id is None:
            return f"no loaded LM Studio instance matched model '{model}' -- nothing unloaded"

        try:
            payload = json.dumps({"instance_id": instance_id}).encode("utf-8")
            req = urllib.request.Request(base_url.rstrip("/") + "/api/v1/models/unload", data=payload,
                                         headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
        except Exception as exc:
            return f"found instance '{instance_id}' but the unload request failed: {exc}"
        return None

    def _call_comfyui_clip(self, clip, system_prompt, prompt_text, do_sample, max_length, temperature,
                           top_k, top_p, min_p, repetition_penalty, presence_penalty, seed):
        """Runs real generation on an already-loaded CLIP via ComfyUI's own
        CLIP.generate() (comfy/sd.py) -- same mechanism as ComfyUI core's
        'Generate Text' node (comfy_extras/nodes_textgen.py), just with our
        own system prompt injected via a llama_template override instead of
        that node's per-model hardcoded templates."""
        template = _chatml_template(system_prompt)
        tokens = clip.tokenize(prompt_text, llama_template=template)
        generated_ids = clip.generate(
            tokens, do_sample=do_sample, max_length=max_length, temperature=temperature,
            top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, seed=seed,
        )
        return clip.decode(generated_ids)

    # -- node entrypoint -----------------------------------------------------

    def chat(self, backend, base_url, model, system_prompt, user_message, new_chat,
              temperature, max_tokens, seed, seed_mode, duration,
              chat_history="[]", first_frame=None, last_frame=None, references=None,
              picture_1=None, picture_2=None, video_1=None, video_audio_1=None,
              audio_1=None, audio_2=None, unload_after_generating=False, lmstudio_api_key="",
              openai_api_key="", anthropic_api_key="", gemini_api_key="",
              clip=None, auto_system_prompt=True, do_sample=True, top_k=64, top_p=0.95,
              min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0,
              **kwargs):
        # See _apply_dotted_kwargs -- same dotted-group-key issue as
        # MiniMaxH3UnifiedToVideo.execute() can hit here too (picture_1/2,
        # audio_1/2 are the same sequentially-numbered-sibling shape).
        if kwargs:
            # chat_history isn't part of this -- it defaults to "[]", not
            # None, so it's excluded here on purpose (the None-check below
            # would never fire for it anyway, and it isn't one of the
            # numbered-sibling sockets this recovery is for).
            values = _apply_dotted_kwargs({
                "first_frame": first_frame, "last_frame": last_frame, "references": references,
                "picture_1": picture_1, "picture_2": picture_2, "video_1": video_1,
                "video_audio_1": video_audio_1, "audio_1": audio_1, "audio_2": audio_2,
                "clip": clip,
            }, kwargs)
            first_frame, last_frame = values["first_frame"], values["last_frame"]
            references = values["references"]
            picture_1, picture_2 = values["picture_1"], values["picture_2"]
            video_1, video_audio_1 = values["video_1"], values["video_audio_1"]
            audio_1, audio_2 = values["audio_1"], values["audio_2"]
            clip = values["clip"]

        # Same "own input wins, references bundle fills the rest" precedence
        # as MiniMaxH3UnifiedToVideo, but capped at this node's own small
        # slot count (2 pictures, 1 video, 1 video_audio, 2 audios) -- see
        # DESCRIPTION for why this node doesn't scale up to the bundle's
        # full 9/3/3/3 capacity the way the video node does.
        bundle = references if isinstance(references, dict) else {}
        pictures = _merge_direct_and_bundle([picture_1, picture_2], bundle.get("pictures"), 2)
        videos_in = _merge_direct_and_bundle([video_1], bundle.get("videos"), 1)
        video_audios_in = _merge_direct_and_bundle([video_audio_1], bundle.get("video_audios"), 1)
        audios_in = _merge_direct_and_bundle([audio_1, audio_2], bundle.get("audios"), 2)
        ref_video_0 = videos_in[0]

        import random
        if seed_mode == "random":
            seed = random.randint(0, 0xffffffff)

        base_url = (base_url or _BACKEND_DEFAULTS.get(backend, "")).strip()
        model = (model or "").strip()

        try:
            history = json.loads(chat_history) if chat_history and not new_chat else []
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

        tag_notes = []
        images_b64 = []
        picture_labels = [("Picture (first_frame)", first_frame), ("Picture (last_frame)", last_frame)]
        picture_labels += [(f"Picture (picture_{i + 1})", p) for i, p in enumerate(pictures)]

        if backend == "comfyui_clip":
            # v1 scope: the comfyui_clip backend only sends text -- see
            # DESCRIPTION. Still note attached media so the model at least
            # knows it exists, same treatment audio already gets below on
            # every backend.
            for label, img in picture_labels:
                if img is not None:
                    tag_notes.append(f"{label} attached this turn but not sent to the comfyui_clip "
                                     f"backend (image input isn't wired up for this backend yet)")
            if ref_video_0 is not None:
                tag_notes.append(f"video_1 attached this turn but not sent to the comfyui_clip backend "
                                 f"(image input isn't wired up for this backend yet)")
        else:
            try:
                # DESCRIPTION promises "up to 2 pictures + 1 video frame" sent to the
                # local model, to keep context/latency sane -- enforce that here
                # instead of silently sending every wired picture.
                for label, img in picture_labels:
                    if img is None:
                        continue
                    if len(images_b64) < MAX_CHAT_PICTURES:
                        images_b64.append(_tensor_to_png_b64(img[:1]))
                        tag_notes.append(f"<Picture {len(images_b64)}> attached this turn = {label}")
                    else:
                        tag_notes.append(f"{label} attached this turn but not shown to the model "
                                         f"(limit of {MAX_CHAT_PICTURES} pictures reached)")
                if ref_video_0 is not None:
                    images_b64.append(_tensor_to_png_b64(ref_video_0[:1]))
                    tag_notes.append(f"<Video 1> attached this turn (first frame shown as an image; "
                                     f"{ref_video_0.shape[0]} frames total)")
            except Exception as exc:
                tag_notes.append(f"(could not encode one or more images for the chat request: {exc})")

        audio_notes = [(f"video_audio_{i + 1}", a) for i, a in enumerate(video_audios_in)]
        audio_notes += [(f"audio_{i + 1}", a) for i, a in enumerate(audios_in)]
        audio_tag_n = 0
        for label, aud in audio_notes:
            if aud is not None:
                audio_tag_n += 1
                tag_notes.append(f"<Audio {audio_tag_n}> attached this turn = {label} (audio bytes are not "
                                 f"sent to the local model -- describe it in your message if it matters)")

        context_note = f"Target video duration: {duration:.1f}s."
        if tag_notes:
            context_note += " Media attached this turn: " + "; ".join(tag_notes) + "."

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and "content" in turn:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"{context_note}\n\n{user_message}".strip()})

        family_name = None
        unload_note = None
        key_note = None
        try:
            if backend == "comfyui_clip":
                if clip is None:
                    raise ValueError("comfyui_clip backend selected but no `clip` is wired -- connect a "
                                      "CLIP loaded elsewhere in this workflow (e.g. a Krea2/Klein/Z-Image "
                                      "checkpoint's text encoder).")
                if auto_system_prompt:
                    detected = _detect_clip_family(clip)
                    if detected is None:
                        tokenizer_class = type(getattr(clip, "tokenizer", None)).__name__
                        raise ValueError(
                            f"comfyui_clip: auto_system_prompt is on, but '{tokenizer_class}' isn't a "
                            f"recognized model family (supported: Krea2, Klein, Z-Image). Turn off "
                            f"auto_system_prompt and set system_prompt yourself, or wire a CLIP from a "
                            f"supported model.")
                    family_name, effective_system_prompt = detected
                else:
                    effective_system_prompt = system_prompt

                # These models take one flat prompt string, not a chat messages
                # array -- flatten history + the new turn into plain text.
                flattened = []
                for turn in history:
                    if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and "content" in turn:
                        speaker = "User" if turn["role"] == "user" else "Assistant"
                        flattened.append(f"{speaker}: {turn['content']}")
                flattened.append(f"User: {context_note}\n\n{user_message}".strip())
                prompt_text_in = "\n\n".join(flattened)

                reply = self._call_comfyui_clip(clip, effective_system_prompt, prompt_text_in, do_sample,
                                                max_tokens, temperature, top_k, top_p, min_p,
                                                repetition_penalty, presence_penalty, seed)
                if unload_after_generating:
                    comfy.model_management.unload_model_and_clones(clip.patcher)
                    unload_note = "unloaded clip via comfy.model_management"
            elif backend in ("openai", "anthropic", "gemini"):
                key_widgets = {"openai": openai_api_key, "anthropic": anthropic_api_key,
                              "gemini": gemini_api_key}
                api_key, key_source = _resolve_api_key(backend, key_widgets[backend])
                if api_key is None:
                    # VALIDATE_INPUTS should already have caught this before the
                    # queue started -- this is just a defensive re-check for
                    # paths that skip validation (e.g. direct API submission).
                    env_name = _ENV_KEY_NAMES[backend]
                    raise ValueError(f"backend={backend} needs an API key -- set the {env_name} "
                                      f"environment variable, or fill in the {backend}_api_key field.")
                if not model:
                    raise ValueError("model is empty -- set it to a model name for this backend, e.g. "
                                      "'gpt-5.1' (openai), 'claude-opus-4-5' (anthropic), 'gemini-3-pro' "
                                      "(gemini).")
                if key_source == "widget":
                    key_note = (f"using a manually-entered {backend}_api_key -- consider "
                               f"{_ENV_KEY_NAMES[backend]} instead")
                    print(f"[MiniMaxH3MultimodalChat] {key_note}")

                if backend == "anthropic":
                    reply = self._call_anthropic(base_url, model, system_prompt, messages, images_b64,
                                                 temperature, max_tokens, api_key)
                else:
                    # Gemini's OpenAI-compat layer uses a different endpoint path
                    # than openai/lmstudio/kobold -- see _BACKEND_DEFAULTS's note.
                    endpoint_path = "/chat/completions" if backend == "gemini" else "/v1/chat/completions"
                    reply = self._call_openai_compatible(base_url, model, messages, images_b64, temperature,
                                                         max_tokens, seed, api_key=api_key,
                                                         endpoint_path=endpoint_path)
                if unload_after_generating:
                    unload_note = "not applicable for cloud backends (no local VRAM to free)"
            elif not model:
                raise ValueError("model is empty -- set it to a model name loaded in your backend")
            elif backend == "ollama":
                reply = self._call_ollama(base_url, model, messages, images_b64, temperature, max_tokens,
                                          seed, unload=unload_after_generating)
                if unload_after_generating:
                    unload_note = "requested via keep_alive=0"
            else:
                reply = self._call_openai_compatible(base_url, model, messages, images_b64, temperature,
                                                     max_tokens, seed)
                if unload_after_generating:
                    if backend == "lmstudio":
                        unload_note = self._unload_lmstudio(base_url, model, lmstudio_api_key) or "unloaded"
                    else:
                        unload_note = ("not automated for KoboldCpp -- its unload API isn't consistently "
                                       "documented across versions; use --admin/--adminunloadtimeout on "
                                       "the KoboldCpp side instead")
            if not reply:
                reply = ""
            error = None
        except (urllib.error.URLError, TimeoutError) as exc:
            reply = ""
            error = f"could not reach {backend} at {base_url}: {exc}"
        except Exception as exc:
            reply = ""
            error = f"{backend} call failed: {exc}"

        if error:
            report = f"MiniMaxH3MultimodalChat: {error}"
            prompt_text = user_message
            return (f"[error] {error}", prompt_text, chat_history if not new_chat else "[]", report)

        m = re.search(r"```text\s*\n(.*?)```", reply, re.DOTALL)
        prompt_text = m.group(1).strip() if m else reply.strip()

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        new_history_json = json.dumps(history, ensure_ascii=False)

        report = (f"MiniMaxH3MultimodalChat: {backend} model={model or family_name or '(empty)'} "
                 f"turns={len(history)//2} media_attached={len(tag_notes)} "
                 f"prompt_extracted={'fenced block' if m else 'whole reply (no ```text block found)'}")
        if family_name:
            report += f" detected_family={family_name}"
        if unload_note:
            report += f" unload={unload_note}"
        if key_note:
            report += f" | {key_note}"

        return (reply, prompt_text, new_history_json, report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3UnifiedToVideo": MiniMaxH3UnifiedToVideo,
    "MiniMaxH3ConcatAVLatent": MiniMaxH3ConcatAVLatent,
    "MiniMaxH3ResolutionSelector": MiniMaxH3ResolutionSelector,
    "MiniMaxH3AudioLock": MiniMaxH3AudioLock,
    "MiniMaxH3MultimodalChat": MiniMaxH3MultimodalChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3UnifiedToVideo": "MiniMax H3 Unified To Video (best-effort)",
    "MiniMaxH3ConcatAVLatent": "MiniMax H3 Concat AV Latent (best-effort)",
    "MiniMaxH3ResolutionSelector": "MiniMax H3 Resolution Selector (best-effort)",
    "MiniMaxH3AudioLock": "MiniMax H3 Audio Lock (best-effort, unverified)",
    "MiniMaxH3MultimodalChat": "MiniMax H3 Multimodal Chat (local LLM)",
}
