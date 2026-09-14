# ComfyUI_My_Custom_Nodes

A personal collection of 11 ComfyUI custom nodes: 6 general-purpose workflow utilities, plus 5 nodes that fill in gaps around a third-party MiniMax H3 video workflow that referenced node types with no public source.

## Table of Contents

- [Installation](#installation)
- [Utility Nodes](#utility-nodes)
  - [Dynamic Image Router](#dynamic-image-router)
  - [Image Hub](#image-hub)
  - [Model Switch (DualModelDPDTSwitch)](#model-switch-dualmodeldpdtswitch)
  - [Sampler Switch (MultiInputSamplerSwitch)](#sampler-switch-multiinputsamplerswitch)
  - [LTX Latent Resizer](#ltx-latent-resizer)
  - [Random Line From File (Wildcard)](#random-line-from-file-wildcard)
- [MiniMax H3 Nodes](#minimax-h3-nodes)
  - [MiniMax H3 Resolution Selector](#minimax-h3-resolution-selector)
  - [MiniMax H3 Concat AV Latent](#minimax-h3-concat-av-latent)
  - [MiniMax H3 Unified To Video](#minimax-h3-unified-to-video)
  - [MiniMax H3 Audio Lock](#minimax-h3-audio-lock)
  - [MiniMax H3 Multimodal Chat](#minimax-h3-multimodal-chat)
- [Appendix A: Seed Configuration Matrix](#appendix-a-seed-configuration-matrix)
- [Appendix B: Wildcard Text File Formatting Guide](#appendix-b-wildcard-text-file-formatting-guide)

## Installation

Licensed under [MIT](./LICENSE).

Drop this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. `LTXLatentResizer`, `RandomLineFromFile`, `MiniMaxH3ConcatAVLatent`, and `MiniMaxH3AudioLock` have no extra dependencies beyond ComfyUI itself. `UniversalImageHub` ships a paired JS file (`web/image_hub.js`, loaded via `WEB_DIRECTORY`) that adds "🔄 Refresh File List" and "Paste from Clipboard" buttons to the node. `MiniMaxH3ResolutionSelector` similarly ships `web/minimax_h3_resolution_selector.js`, which makes its `aspect_ratio` widget actually filter `resolution`'s dropdown (see that node's own section below).

`MiniMaxH3UnifiedToVideo` and `MiniMaxH3ConcatAVLatent`/`MiniMaxH3AudioLock` require a ComfyUI build with MiniMax H3 support (`comfy.nested_tensor`, `minimax_ref_items`/`minimax_keyframes`/`minimax_refs` conditioning support). `MiniMaxH3MultimodalChat` requires a locally-running LLM backend (Ollama, LM Studio, or KoboldCpp) — nothing is called over the network except to whatever `base_url` you point it at.

Three other node names that show up in the same MiniMax H3 workflow this pack was built to support are real and **not** included here — install/update these instead of looking for them in this pack:

| Node | Where it actually lives |
|---|---|
| `MiniMaxH3SigmaShift` | ComfyUI core |
| `MiniMaxH3MemoryEfficientSageAttentionPatch` | [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |
| `MiniMaxH3ReferenceSplitter` | the "Fantastic H3 Prompt Builder" pack |

## Utility Nodes

### Dynamic Image Router

**Category:** `utils` · **File:** `dynamic_image_switch.py`

Routes one incoming `IMAGE` batch to exactly one of 10 numbered outputs (`out_1`...`out_10`), selected by an integer widget.

**Inputs:** `images` (IMAGE, required), `select_path` (INT, 1–10, default 1).

**Outputs:** `out_1` through `out_10` (all IMAGE). Exactly one carries the input batch; the other nine are blocked.

**Behavior:** The 9 unselected outputs use ComfyUI's `ExecutionBlocker` mechanism, which prevents anything downstream of them from executing at all — not just from running on empty data. This means a node on an unselected branch that does `image[0]` instead of `image[:1]` won't crash on a path you never chose, because that node simply never runs. On ComfyUI builds that predate `ExecutionBlocker`, the node falls back to handing unselected outputs an empty `(0, 64, 64, 3)` image tensor instead — downstream nodes that loop over a batch (PreviewImage, SaveImage, VAEEncode) handle that fine, but anything indexing with `image[0]` on that fallback path can still error.

**Practical use case:** switching between multiple possible image sources (e.g. different LoadImage nodes, or different upstream generation branches) feeding into one downstream pipeline, without the unused branches burning compute or throwing preview/save errors.

### Image Hub

**Category:** `image` · **File:** `image_hub.py` + `web/image_hub.js`

Loads a single image (plus its mask) from a searchable dropdown, the same way ComfyUI's checkpoint/LoRA loaders work, but scoped to `input`, `output`, or any custom folder you point it at.

**Inputs:** `folder_type` (combo: `input`/`output`/`custom`, required), `filename` (combo, required — populated by scanning `folder_type`'s directory; click to open and type to filter, same as a checkpoint/LoRA dropdown), `custom_path` (STRING, optional, only used when `folder_type` is `"custom"`), `crop_rect` (STRING, optional, hidden — see **Interactive crop** below), `max_megapixels` (FLOAT, optional, default `0` = disabled), `force_width`/`force_height` (INT, optional, default `0` = disabled — see **Output sizing** below).

**Outputs:** `IMAGE`, `MASK`.

**Behavior:** `filename`'s choices come from recursively scanning the selected folder for image files (`.png`/`.jpg`/`.jpeg`/`.webp`/`.bmp`/`.gif`/`.tif`/`.tiff`); files in subfolders show up as `subfolder/name.ext`. If nothing with a recognized image extension is found, it falls back to listing every file in the folder instead of showing an empty dropdown — attempting to actually load a non-image file from that fallback list now raises a clear error naming the file instead of an opaque PIL traceback. This scan only happens automatically when ComfyUI rebuilds its node schema (server start or a browser page reload) — see the buttons below for rescanning without either of those. A given folder's scan result is cached for a few seconds server-side, so placing or pasting several Image Hub nodes pointed at the same large folder at once doesn't re-walk it once per node. When `folder_type` is `input` or `output`, the resolved path is checked to make sure it can't escape that folder via a `..`-style filename — attempts to do so raise a validation error instead of silently reading a file outside ComfyUI's own directories. `folder_type: custom` intentionally allows any absolute path via `custom_path`. Mask generation always matches the loaded image's actual width/height: if the image has an alpha channel, the (inverted) alpha becomes the mask; otherwise a zero mask is generated at the image's own dimensions, rather than ComfyUI core's fixed 64×64 default for images with no alpha. `IS_CHANGED` tracks the resolved file's modification time, so editing the file on disk (same filename) is picked up without a manual re-queue-all.

**Sort By widget:** a `web/image_hub.js`-only combo (`Name (A-Z)` / `Newest First`) that controls the order `filename`'s choices are scanned into — it isn't part of the node's Python inputs, so it isn't sent to the backend or saved with the workflow (it resets to `Name (A-Z)` on reload). Changing it re-scans immediately. The server-side scan cache mentioned below is keyed per folder *and* sort order, so switching back and forth doesn't re-walk the folder each time.

**🔄 Refresh File List button:** a widget added by `web/image_hub.js`. Re-scans whichever folder `folder_type`/`custom_path` currently point at, in whichever order `Sort By` is set to (via a small API route the node registers, `/universal_image_hub/files`) and repopulates the dropdown immediately, so a file you just dropped into that folder shows up without restarting ComfyUI or reloading the browser tab. This button always bypasses the short-lived server-side scan cache mentioned above, so it's always current. It also fires automatically (using the cache) whenever you change the `folder_type` or `Sort By` widgets, and once when the node is first placed.

**Paste from Clipboard button:** reads an image off your system clipboard, uploads it to ComfyUI's `input` folder via `/upload/image`, and both adds it to the `filename` dropdown and selects it (forcing `folder_type` to `input`). Requires a secure context (HTTPS or localhost) and clipboard permission in your browser.

**Thumbnail preview:** a small image preview appears under the `filename` dropdown, showing the currently-selected file (via a server-resized JPEG served from `/universal_image_hub/thumbnail` — display only, not what's actually loaded for the IMAGE/MASK output), with the *original* file's width × height shown as a caption underneath (not the downscaled thumbnail's own size). It updates whenever `filename`, `folder_type`, or `custom_path` changes, including after a Refresh or a clipboard paste. Always re-reads the file from disk (no caching), so replacing a file on disk under the same name updates the preview and dimensions on next selection.

**Interactive crop:** drag directly on the thumbnail preview to draw a crop rectangle — drag inside it to move, drag a corner to resize, click (without dragging) outside it to clear. With no crop drawn, the full image is used. The rectangle is stored as normalized coordinates in the hidden `crop_rect` input, so it's applied to the real, full-resolution file at load time (not the downscaled preview) and survives save/reload with the workflow. Changing `filename` (picking a different file, a Refresh that has to fall back because the saved one vanished, or a clipboard paste) clears the crop, since a rectangle drawn for one image isn't meaningful on another. *Idea credited to [noEmbryo](https://github.com/noembryo)'s [ComfyUI-noEmbryo](https://github.com/noembryo/ComfyUI-noEmbryo) pack, whose "Load Image (from path)" node offers the same crop/drag/resize/clear interaction set on its own preview — the canvas math and event handling here are an independent implementation, not copied from that project's source.*

**Output sizing:** `max_megapixels` caps the (possibly cropped) output by downscaling only — never upscaling — if it's over the cap (`1.0` = 1024×1024, `0` disables it). `force_width`/`force_height`, when **both** are set, instead force an exact output size, center-cropping first if the aspect ratio doesn't already match (then up- or downscaling to fit exactly), and take priority over `max_megapixels` when both are configured.

⚠️ **Migration note:** `filename` changed from a free-typed STRING to a COMBO dropdown. An existing saved workflow's stored filename value should still load and run correctly even if it's not in the freshly-scanned list, but to pick a *different* file from the dropdown, click "🔄 Refresh File List" first if the one you want isn't showing yet (e.g. it was added after your last ComfyUI restart/page reload).

**Practical use case:** pulling in a reference image you've just copied (from a browser, screenshot tool, or another app) without saving it to disk yourself first, or browsing/picking images out of `input`, `output`, or any other folder on disk without retyping paths.

### Model Switch (DualModelDPDTSwitch)

**Category:** `utils` · **File:** `model_switches.py`

A DPDT-style (double-pole, double-throw) switch between two pairs of `MODEL` inputs — e.g. a "High/Low" GGUF pair for text-to-video and a separate "High/Low" pair for image-to-video.

**Inputs:** `switch` (combo: `"Set_A (T2V GGUF)"` / `"Set_B (I2V GGUF)"`, default Set_A), `set_a_high`, `set_a_low`, `set_b_high`, `set_b_low` (all MODEL, required, lazy).

**Outputs:** `MODEL_HIGH`, `MODEL_LOW`.

**Behavior:** All four MODEL inputs are marked lazy, so only the pair matching the current `switch` selection is actually evaluated upstream — the other pair's loader nodes never run, meaning that model's checkpoint never loads into memory or onto the GPU. If the selected pair isn't fully wired (e.g. `Set_A` is chosen but `set_a_low` has nothing connected), the node raises a clear `ValueError` naming exactly what's missing, instead of silently returning `None` into the rest of the graph.

**Practical use case:** keeping two full model configurations (e.g. a T2V and an I2V GGUF setup) permanently wired into one workflow and flipping between them with a single dropdown, without paying the load cost of the one you're not using on that run.

### Sampler Switch (MultiInputSamplerSwitch)

**Category:** `utils` · **File:** `multi_input_sampler_switch.py`

A 5-way multiplexer for `(positive, negative, latent)` sampler input sets.

**Inputs:** `select_set` (INT, 1–5, default 1), `set_1_pos`/`set_1_neg`/`set_1_latent` (CONDITIONING/CONDITIONING/LATENT, required), `set_2_pos` through `set_5_latent` (12 CONDITIONING/LATENT sockets, optional, lazy).

**Outputs:** `POSITIVE`, `NEGATIVE`, `LATENT`.

**Behavior:** Set 1 is the required, always-evaluated fallback. Sets 2–5 are lazy, so wiring up all five sets doesn't mean all five actually execute on every queue — only `select_set`'s chosen set (plus set 1, as the fallback) is evaluated. If the selected set (2–5) isn't fully wired — say you only connected `set_3_pos` and `set_3_neg` but forgot `set_3_latent` — the node falls back to set 1's value for whichever piece is missing, and prints a console warning naming the gap, rather than crashing or silently passing `None` downstream. `select_set` is validated to be 1–5 before use and raises a clear `ValueError` outside that range — the widget UI already enforces this, but a workflow submitted directly via the ComfyUI API bypasses widget bounds, and an out-of-range value used to silently fall back to set 1 with only a console print.

**Practical use case:** A/B/C-testing several complete prompt/latent combinations against the same KSampler chain by flipping one integer, without every untested combination's conditioning nodes running on every single queue.

### LTX Latent Resizer

**Category:** `LTXVideo/Utils` · **File:** `ltx_latent_resizer.py`

Despite the name, this resizes pixel-space `IMAGE` tensors (not `LATENT` tensors) — it's meant to run immediately before a `VAEEncode`, rounding the result to a multiple ComfyUI's video VAEs require.

**Inputs:** `image` (IMAGE, required), `scale_factor` (FLOAT, default 1.0, 0.1–8.0, required), `multiple_of` (INT, default 32, 1–256, optional), `resize_method` (combo: `bilinear`/`bicubic`/`nearest`/`area`, default `bilinear`, optional), `video_preset` (combo: `none (spatial resize only)` / `LTX-Video (8k+1 frames)` / `Hunyuan Video (4k+1 frames)` / `WAN 2.1 / 2.2 (4k+1 frames)` / `Custom`, default `none`, optional), `custom_frame_multiple`/`custom_frame_remainder` (INT, defaults 8/1, optional — only used when `video_preset` is `Custom`).

**Outputs:** `IMAGE`, `width` (INT), `height` (INT), `frame_count` (INT — the batch/frame dimension after any trimming from `video_preset`; equals the input's own frame count when `video_preset` is `none`).

**Behavior:** Multiplies the input's width/height by `scale_factor`, then rounds both dimensions **up** to the next multiple of `multiple_of` (default 32, which fits LTX and most other video VAEs — change it if yours needs a different alignment). The resize itself uses `torch.nn.functional.interpolate` with the chosen method; `bilinear`/`bicubic` use `align_corners=False`, while `nearest`/`area` don't accept that argument at all so it's omitted for those. Output is clamped to `[0, 1]` — bicubic can overshoot slightly past that range at sharp edges, so the clamp keeps all four resize methods safe to feed straight into a VAEEncode regardless of which one you pick.

`video_preset` additionally trims the batch (frame) dimension down to the nearest valid frame count for the chosen video model's causal VAE — each family has its own "count % multiple == remainder" constraint (LTX-Video needs `8k+1`; Hunyuan Video and WAN 2.1/2.2 both need `4k+1`). This only ever **trims** frames from the end, never pads/duplicates them, since there's no way to invent frames that don't exist — if you need a specific count, feed in at least that many frames before this node. `Custom` lets you supply your own multiple/remainder for a model not in the list. The default, `none (spatial resize only)`, leaves the frame count untouched — this is the node's original, pre-existing behavior.

**Practical use case:** taking an arbitrary-resolution (and, optionally, arbitrary-length) input image/video batch and getting it to a VAE-safe resolution and frame count for your target model in one step, with an adjustable scale factor for quick "generate at 0.5×, upscale later" style workflows.

### Random Line From File (Wildcard)

**Category:** `My_Custom_Nodes` · **File:** `random_text_loader.py`

Picks one random line from a plain text file, with prompt randomization decoupled from KSampler randomization.

**Inputs:** `text_path` (STRING, required — path to your `.txt` file), `seed` (INT, required — normally connected to your KSampler's seed), `prompt_mode` (combo: `Follow KSampler/Seed Field` / `Use Dedicated Prompt Seed` / `True Independent Random`, required), `prompt_fixed_seed` (INT, required — only used in "Use Dedicated Prompt Seed" mode), `lora_trigger` (STRING, optional-by-default — replaces the literal text `[trigger]`), `prefix_text` (STRING, multiline, optional-by-default — prepended to the chosen line).

**Outputs:** `random_line` (STRING), `seed_output` (INT — passes the `seed` input straight through, handy for wiring the same value onward).

**Behavior:** See the [Seed Configuration Matrix](#appendix-a-seed-configuration-matrix) below for exactly how `prompt_mode` changes what drives the random pick. Internally the node uses its own private `random.Random()` instance rather than reseeding Python's global `random` module — the earlier version of this node did reseed the global module, which meant every other node/script in the same ComfyUI process that also called `random.*` could have its own randomness silently disturbed by this node running, and vice versa. That's fixed; this node's randomness is now fully isolated. If `text_path` doesn't exist, or exists but has no non-empty lines, the node now **raises an error** (`FileNotFoundError`/`ValueError`) instead of returning a string like `"Error: ..."` as the actual prompt output — the old behavior meant a broken file path could silently feed a literal error string into a text encoder or video model as if it were a real prompt. A file that isn't valid UTF-8 (e.g. saved as UTF-16 by Notepad, a common Windows default) now raises a clear `ValueError` explaining that, instead of an unhandled `UnicodeDecodeError` traceback. Lines starting with `#` are treated as comments and skipped, same as blank lines — see Appendix B. The parsed line list is cached in memory keyed by the file's own modification time, so a large wildcard file isn't fully re-read and re-split from disk on every single execution — only when it actually changes. `IS_CHANGED` includes the wildcard file's own modification time (not just the seed), so hand-editing the `.txt` file's contents and re-queuing picks up the change even with an identical seed.

**Practical use case:** wildcard-style prompt variation (e.g. rotating through a list of character descriptions, styles, or LoRA trigger phrases) that can be pinned to your image seed, pinned to its own separate seed, or fully independent, per the matrix below — see Appendix B for how to format the text file itself.

## MiniMax H3 Nodes

These five nodes reimplement node types found in a privately-distributed MiniMax H3 workflow that have **no public source available anywhere**. They were built by reverse-engineering the workflow's saved node graph and widget values, grounded as much as possible in ComfyUI's own real, public MiniMax H3 node code (the frame-count/canvas/latent-shape math below is copied verbatim from that real source, so outputs stay byte-for-byte compatible with it). Anything past that math is inferred, not confirmed — see the [model card](./MODEL_CARD.md) for the full honesty/limitations breakdown per node before relying on these for anything important. None of these nodes are affiliated with, endorsed by, or verified against whatever the original private pack actually did.

### MiniMax H3 Resolution Selector

**Category:** `MiniMax H3/custom` · **File:** `minimax_h3_extra_nodes.py`

Picks a width/height/frame-length combination for a MiniMax H3 generation from a named aspect ratio and resolution tier, with two optional overrides.

**Inputs:** `aspect_ratio` (combo — see below), `resolution` (combo of `"W×H"` preset strings, authoritative, required), `duration` (FLOAT, seconds, default 5.0, 0.2–150.0, required), `fps` (INT, default 24, 1–60, required), `custom_width`/`custom_height` (INT, default 0 = disabled, optional — set both above 0 to override the preset entirely), `reference_image` (IMAGE, optional — overrides the ratio/resolution preset, see below).

**Outputs:** `width` (INT), `height` (INT), `size` (STRING, `"W×H"`), `length` (INT, frame count).

**Behavior:** `resolution` is the widget that's actually parsed for width/height on the Python side — `aspect_ratio` is never read there at all, by design. Instead, `aspect_ratio` drives `web/minimax_h3_resolution_selector.js`: changing it filters `resolution`'s dropdown down to just that ratio's presets, fetched from a small `/minimax_h3_resolution_selector/presets` route so the JS never needs its own copy of the sizing math (can't drift out of sync with the Python side). Switching ratios preserves whichever size *tier* you were on (e.g. going from 16:9's Large `1344×768` to 9:16 lands on Large's `768×1344`, not a random entry), falling back to the "Large — H3 default" tier if the previous tier can't be determined. This filtering only ever happens on an actual `aspect_ratio` interaction — never automatically when a saved workflow loads — so opening an older workflow never silently changes its resolution value out from under you. Picking `"Custom / manual"` restores the full unfiltered list. The preset list covers 9 named ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 21:9, 9:21) across 5 size tiers (Small/384 through XXL/1152, with Large/768 matching MiniMax H3's own default), plus two legacy `"384×576"`/`"736×1120"` values so older saved workflows still load with a valid selection (these aren't ratio-tagged, so they drop out of the list the first time you touch `aspect_ratio`). If `custom_width`/`custom_height` are both set above 0, they override the preset entirely (rounded to the nearest multiple of 32 — MiniMax H3's own real spatial alignment requirement). If a `reference_image` is wired, its exact width:height ratio overrides the aspect_ratio/resolution preset (though not the custom width/height override) — the node steps through valid 32-aligned sizes at that exact ratio and picks the largest one that still fits the selected preset's pixel budget. `duration`/`fps` are converted to a frame count and snapped to MiniMax H3's own frame grid (`17k + 5` frames) via the same `align_frame_count` math the real native nodes use.

Two pieces of this node are directly adapted (with credit) from the real, separately-installed [ComfyUI-ResolutionSelector](https://github.com/zerohackz/ComfyUI-ResolutionSelector) pack's `ImageRatioSelectorZerohackz` node: the duration→frame-count `length` output, and the exact-ratio-from-`reference_image` sizing logic.

**Practical use case:** picking a MiniMax H3 canvas size by familiar aspect-ratio names instead of hand-typing width/height, optionally locking the output ratio to match a reference image you're already using elsewhere in the same graph, and getting a frame count that's already snapped to a valid value for the model.

### MiniMax H3 Concat AV Latent

**Category:** `MiniMax H3/custom` · **File:** `minimax_h3_extra_nodes.py`

Packs a plain video `LATENT` and a plain audio `LATENT` into the combined audio+video `NestedTensor` structure MiniMax H3's own nodes (`EmptyMiniMaxH3LatentAV`, `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`) use for their `av_latent` output.

**Inputs:** `video_latent` (LATENT, required — a plain 24-channel video latent, e.g. from a standard `VAEEncode`), `audio_latent` (LATENT, required — a plain 32-channel audio latent, e.g. from `VAEEncodeAudio`).

**Outputs:** `av_latent` (LATENT).

**Behavior:** Validates each input's channel dimension before packing (video must be 24-channel, audio 32-channel) and raises a descriptive `ValueError` naming which input is wrong if the shapes don't match. If an already-packed AV latent is accidentally fed into either socket, the node searches its component tensors for one matching the expected channel count instead of failing outright.

**Practical use case:** building a MiniMax H3 AV latent yourself out of independently-encoded video and audio (e.g. when you want to encode each with different settings, or you already have both from a different pipeline) rather than only being able to get one from the all-in-one native nodes.

### MiniMax H3 Unified To Video

**Category:** `MiniMax H3/custom` · **File:** `minimax_h3_extra_nodes.py`

Combines the functionality of MiniMax H3's native `EmptyMiniMaxH3LatentAV` + `MiniMaxH3ImageToVideo` + `MiniMaxH3ReferenceToVideo` into a single node that accepts every kind of reference media (keyframes, reference images, a reference video with its own audio, and standalone reference audio) at once, tagged into one unified `<Picture N>` / `<Video N>` / `<Audio N>` sequence.

**Inputs:**
- Required: `clip` (CLIP), `video_vae` (VAE), `prompt` (STRING, multiline), `mode` (combo: `auto`/`keyframe`/`reference`/`hybrid`/`text`, default `auto`), `width` (INT, default 1344), `height` (INT, default 768), `duration` (FLOAT, seconds, default 5.0), `fps` (INT, default 24), `ref_image_size` (combo: `match`/`max`, default `match`).
- Optional: `audio_vae` (VAE), `first_frame`/`last_frame` (IMAGE), `references` (H3_REFS — see below), `picture_1` through `picture_4` (IMAGE), `video_1` (IMAGE), `video_audio_1` (AUDIO), `audio_1`/`audio_2` (AUDIO).

**Outputs:** `positive` (CONDITIONING), `av_latent` (LATENT), `conditioned_prompt` (STRING, the raw prompt text as submitted), `media_map_json` (STRING, a JSON summary of what was attached and how it was tagged, including a `warnings` list — see below), `report` (STRING, human-readable summary).

**Behavior:** `mode` controls which reference pathways are active: `keyframe` uses only `first_frame`/`last_frame` and ignores every reference input (including `references`); `reference` does the opposite; `hybrid`/`auto` use whatever's actually wired; `text` ignores all media entirely. `first_frame`/`last_frame` are anchored as literal keyframes (frame index 0 and the final frame) exactly the way the native `MiniMaxH3ImageToVideo` does, encoded through `video_vae`, both resized without cropping so mismatched-aspect-ratio inputs keep their full content rather than one of the two getting silently center-cropped. Every other piece of media — `first_frame`/`last_frame` *also*, plus every picture/video/audio slot below — gets folded into one combined `<Picture N>`/`<Video N>`/`<Audio N>` tag sequence and passed to `clip.tokenize(prompt, minimax_ref_items=...)`. This unified single-sequence tagging (rather than the two separate mechanisms the native nodes use) is inferred from a real saved prompt in the source workflow that tagged a keyframe and a reference audio clip together in one sentence. `ref_image_size` controls how hard reference images are downscaled: `match` scales them to the same pixel budget as the generation's own width×height (cheaper), `max` uses MiniMax H3's higher-fidelity 2048px-short-edge reference pipeline (slower). The `av_latent` output is a correctly-shaped empty AV latent at the requested duration/resolution, ready for a sampler.

A reference video (`video_1` or one supplied via the `references` bundle) longer than the requested output gets trimmed down to fit — both `report` and `media_map_json`'s `warnings` list now say exactly how many frames were dropped from which reference, instead of silently discarding them.

**Reference sockets and the `references` bundle:** the individual reference sockets are named `picture_1`–`4`, `video_1`, `video_audio_1`, `audio_1`/`audio_2` to match the output labels of the real, separately-installed `MiniMaxH3ReferenceSplitter` node (from the "Fantastic H3 Prompt Builder" pack), which uses `picture_1`–`9`, `video_1`–`3`, `video_audio_1`–`3`, `audio_1`–`3`. This node also accepts an `H3_REFS` `references` input — wire it straight from that pack's `MiniMaxH3MediaLoader` or `MiniMaxH3PromptBuilder` `references` output instead of routing through a Reference Splitter. Per slot, a directly-wired socket always wins; anything not directly wired falls back to the matching item pulled from the `references` bundle, up to that bundle's own capacity (9 pictures / 3 videos / 3 paired video_audios / 3 standalone audios) — so a `references` bundle can supply pictures 5–9, a 2nd/3rd video, etc. that this node has no dedicated socket for. `media_map_json`'s `tag_order` always shows the final resolved numbering regardless of source.

⚠️ If you're updating an existing workflow that already has this node wired up: the reference sockets were renamed from `ref_image_0`–`3` / `ref_video_0` / `ref_video_audio_0` / `ref_audio_0`/`1` (0-indexed) to the `picture_1`/`video_1`/`video_audio_1`/`audio_1`/`2` names above (1-indexed) to match the Reference Splitter convention. Delete the existing node instance and place a fresh one from the node menu, then rewire — reloading the old instance in place won't pick up the new socket names cleanly.

**Practical use case:** a single node to drive a MiniMax H3 generation from any mix of keyframes, reference images, a reference video+audio clip, and standalone audio, instead of wiring together 3+ separate native nodes and manually keeping their tag numbering consistent.

### MiniMax H3 Audio Lock

**Category:** `MiniMax H3/custom` · **File:** `minimax_h3_extra_nodes.py`

⚠️ **Unverified guess** — see the [model card](./MODEL_CARD.md) before relying on this one. Re-encodes a reference audio clip and blends it into an existing AV latent's audio stream.

**Inputs:**
- Required: `av_latent` (LATENT, required — must already be a MiniMax H3 AV NestedTensor latent, e.g. from `MiniMaxH3ConcatAVLatent` or a native node), `audio_vae` (VAE, required), `mode` (combo: `lock`/`blend`, default `lock`), `strength` (FLOAT, default 0.35, 0.0–1.0).
- Optional: `audio_1` (AUDIO — the reference clip to lock/blend in), `references` (H3_REFS — see below).

At least one of `audio_1` or a `references` bundle with a non-empty `audios` list must be supplied — with neither, the node now raises a clear error instead of crashing.

**Outputs:** `av_latent` (LATENT, modified), `audio` (AUDIO, passed through unchanged), `report` (STRING).

**Behavior:** Encodes the resolved reference audio through `audio_vae`, then blends it into `av_latent`'s existing audio stream by `strength` (0.0 = target's original audio untouched, 1.0 = fully replaced by the reference). `mode: lock` tiles/loops the reference audio across the entire target duration, for a consistent voice or style throughout the whole clip. `mode: blend` only seeds the very start of the clip with the reference and leaves the rest as the target's original content. There is no way to verify this matches whatever the original private node actually did — treat the `report` output's disclaimer literally and check your renders.

**Reference socket and the `references` bundle:** `audio` was renamed to `audio_1` to match `MiniMaxH3ReferenceSplitter`'s output labels. You can now also wire a `references` (H3_REFS) bundle straight from `MiniMaxH3MediaLoader`/`MiniMaxH3PromptBuilder` instead of routing through a `MiniMaxH3ReferenceSplitter` — a directly-wired `audio_1` always wins, and the bundle's first `audios[]` item is used only as a fallback when `audio_1` isn't connected. If you previously hit a `TypeError: 'NoneType' object is not subscriptable` crash here, it was because the old `audio` input resolved to `None` at runtime (commonly from wiring an empty `video_audio_1` Splitter output instead of its populated `audio_1` output) with no validation catching it first — that case is now a plain, actionable `ValueError` explaining what to check.

⚠️ Same migration note as the other MiniMax H3 nodes: if you have an existing workflow wired to the old `audio` socket name, delete this node instance and place a fresh one, then rewire.

**Practical use case:** nudging or locking the "voice"/audio character of an existing MiniMax H3 AV latent toward a reference audio clip, without re-running the whole video generation.

### MiniMax H3 Multimodal Chat

**Category:** `MiniMax H3/custom` · **File:** `minimax_h3_extra_nodes.py`

A chat node for iteratively writing MiniMax H3-style prompts, talking to a locally-hosted Ollama, LM Studio, or KoboldCpp server; to real OpenAI, Anthropic, or Gemini APIs; or generating directly on a `clip` already loaded elsewhere in your workflow (Krea2/Klein/Z-Image), with no separate server or API call at all. This is a fresh, purpose-built node, not a reimplementation of the original private node (which called an undisclosed cloud API and was disabled in the source workflow anyway).

**Inputs:**
- Required: `backend` (combo: `ollama`/`lmstudio`/`kobold`/`comfyui_clip`/`openai`/`anthropic`/`gemini`, default `ollama`), `base_url` (STRING, default blank = auto-fills the backend's usual address — local ports 11434/1234/5001, or the real OpenAI/Anthropic/Gemini API endpoint; unused for `comfyui_clip`), `model` (STRING, the model name/tag loaded in your backend, or the provider's model id for the cloud backends e.g. `gpt-5.1`/`claude-opus-4-5`/`gemini-3-pro`; unused for `comfyui_clip`), `system_prompt` (STRING, multiline, defaults to a built-in MiniMax H3 prompt-writing system prompt; ignored for `comfyui_clip` when `auto_system_prompt` is on), `user_message` (STRING, multiline), `new_chat` (BOOLEAN, default False — clears `chat_history` before this turn), `temperature` (FLOAT, default 0.7), `max_tokens` (INT, default 2048 — also `comfyui_clip`'s `max_length`), `seed` (INT — unused for `anthropic`, which has no request-level seed), `seed_mode` (combo: `fixed`/`random`), `duration` (FLOAT, seconds, default 5.0 — told to the model for pacing).
- Optional: `chat_history` (STRING, JSON, default `"[]"` — feed the `chat_history` output back into this for multi-turn memory), `first_frame`/`last_frame` (IMAGE), `references` (H3_REFS — see below), `picture_1`/`picture_2`/`video_1` (IMAGE), `video_audio_1`/`audio_1`/`audio_2` (AUDIO), `unload_after_generating` (BOOLEAN, default False — see "Unloading after generating" below), `lmstudio_api_key`/`openai_api_key`/`anthropic_api_key`/`gemini_api_key` (STRING, default blank — see "Cloud backends & API keys" below), `clip` (CLIP, required when `backend` is `comfyui_clip`), `auto_system_prompt` (BOOLEAN, default True, `comfyui_clip` only), `do_sample`/`top_k`/`top_p`/`min_p`/`repetition_penalty`/`presence_penalty` (sampling controls, `comfyui_clip` only — defaults match ComfyUI's own core "Generate Text" node).

**Outputs:** `reply` (STRING, the model's full response), `prompt_text` (STRING, just the extracted finished prompt), `chat_history` (STRING, JSON, updated with this turn), `report` (STRING).

**Behavior:** Sends `system_prompt` plus the parsed `chat_history` plus a new user turn (your `user_message`, prefixed with a short note about the target duration and which reference media was attached this turn) to the chosen backend. For `ollama`/`lmstudio`/`kobold`/`openai`/`gemini`, attached images (`first_frame`, `last_frame`, `picture_1`/`2`, and the first frame of `video_1`) are downscaled (long edge capped at 1024px), base64-encoded, and sent using whichever image format that backend expects — Ollama's native `images` array for `ollama`, OpenAI-style `image_url` content blocks for `lmstudio`/`kobold`/`openai`/`gemini`, and Anthropic's own `{"type": "image", "source": {...}}` content blocks for `anthropic`. This node caps what's actually sent at 2 pictures + 1 video frame, matching its own design goal of keeping context/latency small — if more than 2 of `first_frame`/`last_frame`/`picture_1`/`picture_2` are wired, the extras are noted in the message text as attached-but-not-shown rather than all being sent. Audio inputs are **not** actually transmitted to any backend — they're only noted in the message text (numbered `<Audio 1>`, `<Audio 2>`, ... in the same order `MiniMaxH3UnifiedToVideo` would tag them) so the model knows they exist and can refer to them consistently. The built-in `system_prompt` instructs the model to wrap the finished, ready-to-use prompt in a fenced ` ```text ` block; `prompt_text` is extracted from that block via regex (falling back to the whole reply if no fenced block is found). If the backend can't be reached or errors, the node does not crash the queue — it returns an `[error] ...` reply and a descriptive `report` instead.

**Cloud backends & API keys (`openai`/`anthropic`/`gemini`):** `openai` and `gemini` both speak the same OpenAI-compatible chat-completions shape `lmstudio`/`kobold` already use (Gemini via [Google's own OpenAI-compatibility endpoint](https://ai.google.dev/gemini-api/docs/openai)) -- `anthropic` uses Anthropic's own native Messages API instead, since its request/response shape genuinely differs (system prompt is a separate top-level field, not a role in the messages array). For the API key, each backend checks its own field (`openai_api_key`/`anthropic_api_key`/`gemini_api_key`) **only if the matching environment variable isn't set** -- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, the same names each provider's own official SDK looks for. Typing a key directly into the field works, but it gets saved in **plain text inside the workflow JSON** -- the field's tooltip warns about this, and if you do it anyway, `report` and the ComfyUI console both note that a manually-entered key was used, as a gentle nudge rather than a block. If neither the env var nor the field has a value, the node refuses to queue at all with a specific message naming exactly which environment variable to set -- there's nothing it can do without a key, so this fails before execution rather than deep inside a failed HTTP call.

**`backend: comfyui_clip`:** runs real text generation directly on the `clip` you wire in, using ComfyUI's own native `CLIP.generate()` (`comfy/sd.py`) — the same mechanism ComfyUI core's own "Generate Text" node uses. This works for any CLIP whose underlying model is a generation-capable LLM (Qwen3-VL, Qwen3, Gemma, etc. — the ones Krea2/Klein/Z-Image/Qwen-Image use), **not** for a classic contrastive CLIP or a T5 encoder, which have no `.generate()` at all. Since it's ComfyUI's own already-loaded model, there's no separate process and no second copy of the weights in VRAM. `auto_system_prompt` (default on) detects the model family from `clip.tokenizer`'s exact Python class name and picks a matching system prompt automatically (see the table below); turn it off to use `system_prompt` verbatim, which is also required for a family this node doesn't recognize (it raises a clear error naming the unrecognized class rather than guessing). **v1 scope:** this backend is text-only for now — attached pictures/video are noted in the message text but not actually sent as images, unlike the HTTP backends above. `report` shows the detected family (`detected_family=...`) when auto-detection was used.

| Detected class | Family | Prompt style | Grounded in |
|---|---|---|---|
| `Krea2Tokenizer` | Krea2 | detailed visual-description captioning | Krea2's own real conditioning template text |
| `KleinTokenizer` / `KleinTokenizer8B` | Klein (FLUX.2) | generic image-prompt expansion | best-effort, not an official spec |
| `ZImageTokenizer` | Z-Image | generic image-prompt expansion | best-effort, not an official spec |
| anything else, including MiniMax H3's own `MiniMaxH3Tokenizer` | — | — | not supported — its tokenizer has an incompatible interface (no chat template support); use the other backends for H3 prompt-writing |

**Unloading after generating:** `unload_after_generating` frees the model once this node is done with it, so the rest of the workflow (or the next chat turn against a different model) has the VRAM back. Support differs per backend: **ollama** adds `keep_alive: 0` to the same request, which is documented Ollama behavior; **lmstudio** looks up the loaded instance via `GET /api/v1/models` and calls `POST /api/v1/models/unload` (best-effort — LM Studio requires an API key by default, see `lmstudio_api_key`, and a failed unload is noted in `report` rather than failing the node, since the chat reply already succeeded by that point); **comfyui_clip** calls ComfyUI's own `comfy.model_management.unload_model_and_clones()` on the wired clip, so it participates in ComfyUI's normal model management just like any other unload; **kobold** is **not automated** — KoboldCpp does have an admin-API unload feature, but its exact endpoint isn't consistently documented across versions, so this node doesn't guess at it. Use KoboldCpp's own `--admin`/`--adminunloadtimeout` launch flags instead if you want it to free itself automatically. **openai/anthropic/gemini** ignore this flag (`report` notes it as not applicable) — there's no local VRAM involved with a cloud API call.

**Dialogue handling:** the default `system_prompt` tells the model to carry any spoken dialogue in your `user_message` into the prompt verbatim, using MiniMax H3's real dialogue syntax pulled from the bundled `Video_Prompt_Writing_Guide.pdf` (from the same real "Fantastic H3 Prompt Builder" pack): stable speaker IDs like `(S1)`/`(S2)` reused across shots, the line itself wrapped in `<d>[Language] exact words</d>` with the descriptive/delivery text kept outside the tag, `says in an off-screen voiceover` for voiceover plus a note that the on-screen character's lips stay closed, and `<scenetrans>`/`<cutoff>` for dialogue that crosses a cut or gets cut off by the video ending. It also explicitly tells the model dialogue must **not** be summarized into `overall_soundscape` — that field is ambient/physical sound only. If your own edited `system_prompt` predates this, dialogue you type into `user_message` may get dropped or reduced to a vague "voices" mention in `overall_soundscape` instead of being written out — reset to the default or add these rules yourself.

**Reference sockets and the `references` bundle:** same naming and bundle mechanism as `MiniMaxH3UnifiedToVideo` above (see that section for the full explanation) — `picture_1`/`2`, `video_1`, `video_audio_1`, `audio_1`/`2` match `MiniMaxH3ReferenceSplitter`'s labels, and a directly-wired socket always wins over the `references` bundle. Unlike the video node, this one is capped at exactly 2 pictures + 1 video frame regardless of source — it's a lightweight prompt-drafting aid talking to a local model, not the actual generation step, so anything past that cap in a wired `references` bundle is simply not sent (no error, just omitted).

⚠️ Same migration note as `MiniMaxH3UnifiedToVideo`: if you have an existing workflow using the old `ref_image_0`/`1`/`ref_video_0`/`ref_video_audio_0`/`ref_audio_0`/`1` socket names, delete this node instance and place a fresh one, then rewire.

**Practical use case:** iteratively drafting a MiniMax H3 prompt through conversation with a local model, optionally showing it your actual reference images so its shot description references them accurately, with the finished prompt output ready to wire straight into `MiniMaxH3UnifiedToVideo`'s `prompt` input. With `backend: comfyui_clip`, the same node doubles as a zero-extra-VRAM prompt-enhancer for Krea2/Klein/Z-Image image workflows -- wire in the checkpoint's own CLIP instead of loading a separate chat model.

## Appendix A: Seed Configuration Matrix

How `RandomLineFromFile`'s `prompt_mode` widget determines which line gets picked, relative to your `seed` and `prompt_fixed_seed` inputs:

| `prompt_mode` | Line selection driven by | Same line every re-queue with unchanged inputs? | Typical use |
|---|---|---|---|
| `Follow KSampler/Seed Field` | `seed` input (normally wired to the same value as your KSampler's seed) | Yes, until `seed` or the file's contents change | Prompt varies exactly in lockstep with your image seed — deterministic and reproducible per-seed |
| `Use Dedicated Prompt Seed` | `prompt_fixed_seed` input only — `seed` is ignored for line selection (it's still passed through on `seed_output`) | Yes, until `prompt_fixed_seed` or the file's contents change | Lock the prompt to one specific line while still freely randomizing the image seed elsewhere in the graph |
| `True Independent Random` | OS entropy (`os.urandom`), fresh every single execution | No — always different, every run | Pure randomness with no reproducibility, independent of any seed in the graph |

Regardless of mode, editing the target `.txt` file's contents on disk and re-queuing will pick up the change (the node's `IS_CHANGED` tracks the file's modification time), even with an otherwise-identical seed.

## Appendix B: Wildcard Text File Formatting Guide

`RandomLineFromFile` reads its target file as plain UTF-8 text, one candidate line per line:

- Save the file with UTF-8 encoding — other encodings (e.g. UTF-16, a common Notepad default on Windows) raise a clear error rather than being silently misread.
- Blank lines are skipped automatically — you don't need to remove spacing between entries.
- Lines starting with `#` are treated as comments and skipped too, so you can temporarily disable an entry without deleting it.
- Leading/trailing whitespace on each line is stripped before use.
- Use the literal placeholder `[trigger]` anywhere in a line to have it replaced with whatever you connect to the `lora_trigger` input at runtime — handy for keeping a wildcard file generic across multiple LoRAs that each need their own trigger word.

Example file contents:

```
a woman with [trigger] wearing a red jacket, city street at night
a man with [trigger] in a forest clearing, soft morning light
[trigger], studio portrait, neutral background
```

With `lora_trigger` set to `"short blonde hair"`, a random line like `"a woman with [trigger] wearing a red jacket, city street at night"` becomes `"a woman with short blonde hair wearing a red jacket, city street at night"`. If `prefix_text` is also set (e.g. to a consistent style tag), it's prepended with a single space before the chosen line, e.g. `"cinematic film still, a woman with short blonde hair wearing a red jacket, city street at night"`.

If the file is missing or ends up with zero non-empty lines, the node now raises an error rather than returning an `"Error: ..."` string as the prompt itself — check the ComfyUI console/queue error for the exact problem rather than expecting an error message to show up as generated text.
