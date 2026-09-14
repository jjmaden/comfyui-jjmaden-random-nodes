# Model Card: ComfyUI_My_Custom_Nodes

This is a code repository (a ComfyUI custom node pack), not a trained model. This card follows the model-card format because it's a familiar, structured way to disclose provenance, intended use, and limitations before publishing — everything below refers to node *behavior*, not model *weights*.

## Overview

`ComfyUI_My_Custom_Nodes` is a personal collection of 11 custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI):

- **6 general-purpose workflow utilities**, each fixing a specific limitation encountered in real use: `DynamicImageRouter`, `UniversalImageHub`, `DualModelDPDTSwitch`, `MultiInputSamplerSwitch`, `LTXLatentResizer`, `RandomLineFromFile`.
- **5 MiniMax H3 companion nodes** (`MiniMaxH3ResolutionSelector`, `MiniMaxH3ConcatAVLatent`, `MiniMaxH3UnifiedToVideo`, `MiniMaxH3AudioLock`, `MiniMaxH3MultimodalChat`), built to fill gaps in a privately-distributed MiniMax H3 workflow that referenced several node types with no public source anywhere.

Full per-node documentation of inputs, outputs, and behavior is in [README.md](./README.md). This card focuses on provenance, honesty about what's verified vs. guessed, and what a downstream user needs to know before relying on any of this.

## What's Included / Provenance

### The 6 utility nodes

Written and iterated on directly against real ComfyUI workflows; each one addresses a bug or missing feature found through actual use (see the "Behavior" section of each node in the README for specifics — lazy-evaluation gaps that loaded unused models, stale image-cache behavior, global random-state pollution, path traversal on custom file paths, an out-of-range `select_set` silently falling back instead of erroring, aliased output tensors on `DynamicImageRouter`'s pre-`ExecutionBlocker` fallback path, an unhandled `UnicodeDecodeError` on non-UTF-8 wildcard files, unnecessary re-scanning/re-reading of the filesystem on every execution, and one factually-inaccurate detail in this pack's own prior documentation). `LTXLatentResizer` additionally gained an optional `video_preset` mode that trims the batch/frame dimension to a valid count for LTX-Video/Hunyuan Video/WAN's own VAE constraints, not just the spatial resize it originally did. These carry no third-party attribution beyond standard ComfyUI APIs (`comfy_execution.graph_utils.ExecutionBlocker`, `folder_paths`, `torch.nn.functional.interpolate`).

### The 5 MiniMax H3 nodes

These were built to support a specific MiniMax H3 workflow JSON that referenced 8 "MiniMaxH3"-prefixed node types. Of those 8:

- **3 are real, publicly available nodes** and are *not* included in this pack: `MiniMaxH3SigmaShift` (ComfyUI core), `MiniMaxH3MemoryEfficientSageAttentionPatch` ([ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)), and `MiniMaxH3ReferenceSplitter` (the "Fantastic H3 Prompt Builder" pack). If you're missing these, install/update those packs — do not expect this repository to provide them.
- **5 have no public source anywhere** and were reconstructed from scratch by reading the workflow JSON's saved node graph, input/output wiring, and widget values. These are the 5 nodes this repository actually adds.

None of these 5 nodes are affiliated with, derived from source code belonging to, or verified against whatever the original privately-distributed pack's nodes actually did internally. They are best-effort reconstructions built primarily from:

1. **ComfyUI's own real, public MiniMax H3 node source** — the frame-count grid math (`align_frame_count`, 17k+5), canvas-sizing math (`adapt_canvas`, 32-multiple rounding, 768px short edge / 768×1344px area cap), and latent-shape math (`video_latent_t`, `temporal_shape`) are copied **verbatim** from ComfyUI's native MiniMax H3 node file, so outputs from `MiniMaxH3ResolutionSelector`, `MiniMaxH3ConcatAVLatent`, and `MiniMaxH3UnifiedToVideo` stay byte-for-byte shape-compatible with ComfyUI's real `EmptyMiniMaxH3LatentAV` / `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` nodes.
2. **The workflow JSON's saved widget values and node graph** — e.g. the exact `["2:3 (Portrait Photo)", "384×576"]`-style values seen wired into the real resolution-selector node, or a saved prompt that tagged a keyframe and a reference audio clip together in one sentence (informing `MiniMaxH3UnifiedToVideo`'s unified tagging design).
3. **Deliberately borrowed logic from a different, real, already-installed pack** — `MiniMaxH3ResolutionSelector`'s duration→frame-count snapping and its `reference_image` exact-ratio sizing are adapted, with credit, from [ComfyUI-ResolutionSelector](https://github.com/zerohackz/ComfyUI-ResolutionSelector)'s `ImageRatioSelectorZerohackz` node.

A post-release review of these 5 nodes against their own documented contracts found and fixed several internal inconsistencies (none affect the provenance/confidence notes above, since all were implementation bugs against already-documented intent, not new inference): `MiniMaxH3ResolutionSelector`'s `reference_image` input was silently overriding `custom_width`/`custom_height` despite its own tooltip saying it shouldn't; `MiniMaxH3UnifiedToVideo` resized `first_frame` and `last_frame` with different crop modes (now both uncropped, matching every other resize in the node) and silently truncated an over-length reference video with no note in its `report`/`media_map_json` (now recorded in a `warnings` field); `MiniMaxH3MultimodalChat`'s own `DESCRIPTION` claimed a hard cap of "2 pictures + 1 video frame" sent to the local model that the code didn't actually enforce (now enforced), and its audio reference notes weren't numbered the way its own system prompt's `<Audio N>` convention expects (now numbered, matching `MiniMaxH3UnifiedToVideo`'s own numbering order).

### Per-node confidence

| Node | Confidence | Why |
|---|---|---|
| `MiniMaxH3ConcatAVLatent` | **Highest** | Just packs two plain latents into the same `NestedTensor` structure ComfyUI's own real nodes already produce — low surface area for the logic to be wrong. |
| `MiniMaxH3ResolutionSelector` | **High** | Preset math is copied verbatim from real ComfyUI source. `aspect_ratio` now actually filters `resolution`'s dropdown (via `web/minimax_h3_resolution_selector.js`, added after the initial release) instead of being a dead label — the filtering fetches its size table from the same real `_capped_canvas` math server-side rather than reimplementing it in JS, so it can't drift out of sync, but the JS itself (tier-preservation across ratio switches, not touching a loaded workflow's value automatically) hasn't been visually verified in a live browser in this environment. |
| `MiniMaxH3UnifiedToVideo` | **Medium** | Built on real, verbatim native building blocks, but the *merging* of keyframe + reference pathways into one unified tag sequence is inferred from one example in the workflow JSON, not confirmed against the original node's actual internal logic. |
| `MiniMaxH3MultimodalChat` | **N/A — not a reimplementation** | A fresh node for a different (local) backend than whatever the original used; not trying to replicate anything, so there's nothing to be unfaithful to. Its H3 system prompt is reconstructed from real saved chat replies in the source workflow, but MiniMax's official H3 prompt-writing documentation should be treated as the actual source of truth if you have access to it. Its later-added `backend: comfyui_clip` path is a fresh, separately-verified feature (see the note below) with its own, independent confidence levels per detected model family. |
| `MiniMaxH3AudioLock` | **Lowest — a genuine guess** | No equivalent real node exists to check the math against at all; the `lock`/`blend` behavior is inferred purely from two widget values (`"lock"`, `0.35`) seen in the workflow JSON, with no way to verify against real output. Inspect your renders before trusting this one for anything you care about. |

### `MiniMaxH3MultimodalChat`'s `comfyui_clip` backend

Added after the initial 5-node release, this backend runs generation directly on a `clip` wired in from elsewhere in the workflow, via ComfyUI's own `CLIP.generate()` (`comfy/sd.py`) -- the same mechanism ComfyUI core's own `comfy_extras/nodes_textgen.py` "Generate Text" node uses. This part is **not a guess**: the calling contract (`clip.tokenize(prompt, llama_template=...)` → `clip.generate(tokens, do_sample=, max_length=, ...)` → `clip.decode(ids)`) was read directly out of this installation's own `comfy/sd.py`, `comfy/text_encoders/llama.py` (`BaseGenerate`), and `comfy/text_encoders/{krea2,flux,z_image,qwen3vl}.py` before writing this feature, and exercised with mock `CLIP` objects matching those exact signatures (see the pack's test history) -- it has not, however, been run against a real loaded Krea2/Klein/Z-Image checkpoint, since that requires a GPU and the actual model weights.

Confidence differs by piece:
- **The generation mechanism itself (High):** directly verified against ComfyUI's real source in this installation, not inferred from behavior.
- **Family detection (High):** `type(clip.tokenizer).__name__` matching against `Krea2Tokenizer`/`KleinTokenizer`/`KleinTokenizer8B`/`ZImageTokenizer` is exact-class-name matching against real, currently-shipping ComfyUI source -- but it's a snapshot of one ComfyUI version; if a future ComfyUI release renames or restructures these classes, detection for that family would silently stop matching (falling through to the "unrecognized family" error, not silently misbehaving) until this pack is updated.
- **Krea2's system prompt (High):** its instructions are lifted directly from Krea2's own real conditioning template text (`comfy/text_encoders/krea2.py`'s `KREA2_TEMPLATE`), not invented.
- **Klein's and Z-Image's system prompts (Low — explicit best-effort guesses):** there's no equivalent official captioning spec available for these the way Krea2 conveniently exposes its own; they're generic "write a detailed visual description" prompts, clearly labeled as such in the node's own `DESCRIPTION` and in their text. Edit them if you know the actual style either model was trained on.
- **MiniMax H3's own CLIP is deliberately excluded, not attempted:** `MiniMaxH3Tokenizer` (`comfy/text_encoders/minimax.py`) has a completely different `tokenize_with_weights()` signature built around H3's own reference-tagging scheme (`minimax_ref_items=...`), with no chat-template mechanism at all -- it isn't a "maybe works" case, it's structurally incompatible with this backend, so it was left out rather than shipped half-working.

### `MiniMaxH3MultimodalChat`'s cloud backends (`openai`/`anthropic`/`gemini`)

Also added after the initial release. `openai` and `gemini` reuse the existing OpenAI-compatible request path (`lmstudio`/`kobold` already spoke this shape) with a real provider base URL and API key -- Gemini specifically via [Google's documented OpenAI-compatibility endpoint](https://ai.google.dev/gemini-api/docs/openai), verified to use a *different* URL suffix (`/chat/completions`, not `/v1/chat/completions`) than OpenAI itself, which is easy to get wrong and was checked directly against Google's own docs before implementing. `anthropic` uses a separate method for Anthropic's native Messages API (endpoint, headers, system-prompt-as-top-level-field, and image content-block shape all verified against [Anthropic's own API reference](https://platform.claude.com/docs/en/api/messages)) rather than being forced into the OpenAI shape it doesn't share. All three were exercised with mocked HTTP responses matching each provider's real documented shape, not against live API calls (that would cost real money per test run and isn't something to do routinely).

API key handling was designed deliberately, not just bolted on: each cloud backend prefers its provider's own standard environment variable (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GEMINI_API_KEY` -- the same names each provider's official SDK reads) over a widget field, specifically because a key typed into a node widget is saved in plain text inside the workflow JSON. `VALIDATE_INPUTS` refuses to queue at all if neither is set, naming the exact environment variable to set rather than letting the run fail deep inside an HTTP call; if a key *is* typed directly, that's still honored (typing one in is a legitimate choice), but it's flagged in both the console and the node's `report` output as a nudge toward the safer option.

## Intended Use

- Personal and community ComfyUI workflows that need the specific gaps these nodes fill: lazy model/sampler switching, path-safe custom image loading, LTX-safe pixel resizing, wildcard prompt files, or MiniMax H3 video generation where the 5 missing node types above are needed and the confidence caveats above are acceptable.
- Use by people comfortable reading a "best-effort reconstruction" disclaimer and verifying output themselves, especially for the MiniMax H3 nodes.

## Out-of-Scope / Not Intended For

- **Not a replacement for, or endorsement of, any specific privately-distributed workflow or pack.** These nodes do not contain, redistribute, or derive from any such pack's source code — only from observing the *shape* of its saved graph and widget values in one workflow JSON, plus real public ComfyUI source.
- **Not verified for production or commercial pipelines**, particularly `MiniMaxH3AudioLock` and `MiniMaxH3UnifiedToVideo`'s reference-merging behavior — treat any output from these as a starting point to inspect, not a guaranteed-correct result.
- `MiniMaxH3MultimodalChat` sends whatever you type, plus downscaled copies of any attached reference images, to whatever backend/`base_url` you configure. For `ollama`/`lmstudio`/`kobold`/`comfyui_clip` that's local by default and makes no other network calls unless you point `base_url` at a remote server yourself. For `openai`/`anthropic`/`gemini`, sending your prompt (and any attached images) to that provider's real cloud API is the explicit point of selecting that backend — this is not a hidden or accidental network call, but it does mean your data leaves your machine under that provider's own data-handling terms, not this pack's.

## Requirements / Installation

- ComfyUI (current version recommended). MiniMax H3 nodes additionally require a ComfyUI build with MiniMax H3 support (`comfy.nested_tensor`, `minimax_ref_items`/`minimax_keyframes`/`minimax_refs`).
- `MiniMaxH3MultimodalChat` requires a running local LLM server (Ollama, LM Studio, or KoboldCpp) with a vision-capable model loaded if you want it to see attached reference images -- or, for `backend: comfyui_clip`, a `clip` wired in from a generation-capable model already loaded in the workflow (see below), no server required.
- No additional Python dependencies beyond what a standard ComfyUI install already provides (`torch`, `torchaudio`, `numpy`, `Pillow` are all already ComfyUI requirements).
- Install by placing this folder in `ComfyUI/custom_nodes/` and restarting ComfyUI. See [README.md](./README.md) for full per-node usage.

## Limitations

- The 5 MiniMax H3 nodes are reverse-engineered from workflow metadata, not from the original pack's source code (which was never available to reconstruct from). They will not necessarily produce identical output to whatever the original private nodes did, even where the math is copied verbatim from real ComfyUI source, because the parts that aren't verbatim (see the confidence table above) are inferred.
- `MiniMaxH3AudioLock` in particular has no real reference implementation to check against at all and should be treated as experimental.
- `MiniMaxH3MultimodalChat`'s `comfyui_clip` backend has been verified against ComfyUI's own real source and tested with mock objects matching its exact interfaces, but not yet run against a real loaded Krea2/Klein/Z-Image checkpoint (no GPU was available while building it) -- and its Klein/Z-Image system prompts are explicit best-effort guesses, not grounded in an official spec (see the note above).
- `MiniMaxH3MultimodalChat`'s `openai`/`anthropic`/`gemini` backends were verified against each provider's own documented API reference and tested with mocked HTTP responses, but not against live API calls -- a provider changing their API shape after this was written could break the corresponding backend, the same version-drift risk any integration like this carries.
- `MiniMaxH3ResolutionSelector`'s `aspect_ratio` → `resolution` filtering (`web/minimax_h3_resolution_selector.js`) has been reasoned through and its server-side data source unit-tested, but not visually exercised in a live browser in this environment -- if you hit unexpected behavior with it, that JS file is the first place to look.
- None of the utility nodes have been tested against every possible ComfyUI version/build; the lazy-evaluation and `ExecutionBlocker` features depend on reasonably current ComfyUI internals, with fallbacks where practical (see `DynamicImageRouter`'s documented fallback for pre-`ExecutionBlocker` ComfyUI builds).
- `UniversalImageHub`'s interactive crop overlay (`web/image_hub.js`): the coordinate math (mapping a drag to a normalized crop rectangle through the preview's `object-fit: contain` letterboxing, for both letterbox orientations, plus move/resize/clamp/click-to-clear behavior) was verified by extracting it into a standalone HTML page and driving real mouse drags against it in a browser, confirming the computed rectangles against hand-calculated expected values. It has not been exercised inside a live ComfyUI page in this environment (ComfyUI itself was never launched here) — if the rectangle seems misaligned against the actual image in practice, that file is the first place to check. The backend crop/resize math (`image_hub.py`'s `_apply_crop_and_resize`) was separately unit-tested with real Pillow images.
- This pack has not been submitted to or reviewed by the ComfyUI Manager registry.

## License

MIT — see [LICENSE](./LICENSE). MIT is the most common choice across the ComfyUI custom-node ecosystem, and permissively allows use, modification, and redistribution (including commercially) with attribution.

## Credits / Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the frame-count, canvas-sizing, and latent-shape math in the MiniMax H3 nodes is copied verbatim from ComfyUI's own native MiniMax H3 node source, per its license.
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — provides the real `MiniMaxH3MemoryEfficientSageAttentionPatch` node referenced by, but not included in, this pack.
- The "Fantastic H3 Prompt Builder" pack — provides the real `MiniMaxH3ReferenceSplitter`, `MiniMaxH3MediaLoader`, and `MiniMaxH3PromptBuilder` nodes referenced by, but not included in, this pack, and its bundled `Video_Prompt_Writing_Guide.pdf` is the direct source for `MiniMaxH3MultimodalChat`'s dialogue-formatting rules (speaker IDs, `<d>` tags, off-screen voiceover, `<scenetrans>`/`<cutoff>`) and for `MiniMaxH3UnifiedToVideo`/`MiniMaxH3MultimodalChat`/`MiniMaxH3AudioLock`'s `references`-socket naming.
- [ComfyUI-ResolutionSelector](https://github.com/zerohackz/ComfyUI-ResolutionSelector) (`ImageRatioSelectorZerohackz`) — `MiniMaxH3ResolutionSelector`'s duration→frame-count snapping and exact-ratio-from-reference-image sizing logic are adapted from this pack, with credit, as noted directly in the node's own source comments.
- [ComfyUI-noEmbryo](https://github.com/noembryo/ComfyUI-noEmbryo) — `UniversalImageHub`'s interactive crop rectangle (drag to draw, drag inside to move, drag a corner to resize, click outside to clear) is credited to the same interaction set on this pack's "Load Image (from path)" node. Only the *idea* is credited; the canvas math and event handling in `web/image_hub.js` are an independent implementation, not adapted from that project's source.

## Disclaimer

This pack is an independent, personal project. It is not affiliated with, endorsed by, or sponsored by MiniMax, any MiniMax H3 model creator, ComfyUI, or the creator(s) of the privately-distributed workflow that motivated the 5 MiniMax H3 nodes above. All best-effort/guessed behavior is labeled as such directly in each affected node's `DESCRIPTION` string, visible in the ComfyUI node info panel at runtime, in addition to this document.
