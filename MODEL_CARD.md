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
| `MiniMaxH3ResolutionSelector` | **High** | Preset math is copied verbatim from real ComfyUI source; the only inferred part is the aspect-ratio-label/resolution-string pairing UX, which doesn't affect correctness once `resolution` is set. |
| `MiniMaxH3UnifiedToVideo` | **Medium** | Built on real, verbatim native building blocks, but the *merging* of keyframe + reference pathways into one unified tag sequence is inferred from one example in the workflow JSON, not confirmed against the original node's actual internal logic. |
| `MiniMaxH3MultimodalChat` | **N/A — not a reimplementation** | A fresh node for a different (local) backend than whatever the original used; not trying to replicate anything, so there's nothing to be unfaithful to. Its system prompt is reconstructed from real saved chat replies in the source workflow, but MiniMax's official H3 prompt-writing documentation should be treated as the actual source of truth if you have access to it. |
| `MiniMaxH3AudioLock` | **Lowest — a genuine guess** | No equivalent real node exists to check the math against at all; the `lock`/`blend` behavior is inferred purely from two widget values (`"lock"`, `0.35`) seen in the workflow JSON, with no way to verify against real output. Inspect your renders before trusting this one for anything you care about. |

## Intended Use

- Personal and community ComfyUI workflows that need the specific gaps these nodes fill: lazy model/sampler switching, path-safe custom image loading, LTX-safe pixel resizing, wildcard prompt files, or MiniMax H3 video generation where the 5 missing node types above are needed and the confidence caveats above are acceptable.
- Use by people comfortable reading a "best-effort reconstruction" disclaimer and verifying output themselves, especially for the MiniMax H3 nodes.

## Out-of-Scope / Not Intended For

- **Not a replacement for, or endorsement of, any specific privately-distributed workflow or pack.** These nodes do not contain, redistribute, or derive from any such pack's source code — only from observing the *shape* of its saved graph and widget values in one workflow JSON, plus real public ComfyUI source.
- **Not verified for production or commercial pipelines**, particularly `MiniMaxH3AudioLock` and `MiniMaxH3UnifiedToVideo`'s reference-merging behavior — treat any output from these as a starting point to inspect, not a guaranteed-correct result.
- `MiniMaxH3MultimodalChat` sends whatever you type, plus downscaled copies of any attached reference images, to whatever `base_url` you configure. It defaults to local ports (Ollama/LM Studio/KoboldCpp) and makes no other network calls, but if you point `base_url` at a remote server, that data leaves your machine — this is entirely under your control via the `base_url` widget, not hardcoded.

## Requirements / Installation

- ComfyUI (current version recommended). MiniMax H3 nodes additionally require a ComfyUI build with MiniMax H3 support (`comfy.nested_tensor`, `minimax_ref_items`/`minimax_keyframes`/`minimax_refs`).
- `MiniMaxH3MultimodalChat` requires a running local LLM server (Ollama, LM Studio, or KoboldCpp) with a vision-capable model loaded if you want it to see attached reference images.
- No additional Python dependencies beyond what a standard ComfyUI install already provides (`torch`, `torchaudio`, `numpy`, `Pillow` are all already ComfyUI requirements).
- Install by placing this folder in `ComfyUI/custom_nodes/` and restarting ComfyUI. See [README.md](./README.md) for full per-node usage.

## Limitations

- The 5 MiniMax H3 nodes are reverse-engineered from workflow metadata, not from the original pack's source code (which was never available to reconstruct from). They will not necessarily produce identical output to whatever the original private nodes did, even where the math is copied verbatim from real ComfyUI source, because the parts that aren't verbatim (see the confidence table above) are inferred.
- `MiniMaxH3AudioLock` in particular has no real reference implementation to check against at all and should be treated as experimental.
- None of the utility nodes have been tested against every possible ComfyUI version/build; the lazy-evaluation and `ExecutionBlocker` features depend on reasonably current ComfyUI internals, with fallbacks where practical (see `DynamicImageRouter`'s documented fallback for pre-`ExecutionBlocker` ComfyUI builds).
- This pack has not been submitted to or reviewed by the ComfyUI Manager registry.

## License

MIT — see [LICENSE](./LICENSE). MIT is the most common choice across the ComfyUI custom-node ecosystem, and permissively allows use, modification, and redistribution (including commercially) with attribution.

## Credits / Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the frame-count, canvas-sizing, and latent-shape math in the MiniMax H3 nodes is copied verbatim from ComfyUI's own native MiniMax H3 node source, per its license.
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — provides the real `MiniMaxH3MemoryEfficientSageAttentionPatch` node referenced by, but not included in, this pack.
- The "Fantastic H3 Prompt Builder" pack — provides the real `MiniMaxH3ReferenceSplitter`, `MiniMaxH3MediaLoader`, and `MiniMaxH3PromptBuilder` nodes referenced by, but not included in, this pack, and its bundled `Video_Prompt_Writing_Guide.pdf` is the direct source for `MiniMaxH3MultimodalChat`'s dialogue-formatting rules (speaker IDs, `<d>` tags, off-screen voiceover, `<scenetrans>`/`<cutoff>`) and for `MiniMaxH3UnifiedToVideo`/`MiniMaxH3MultimodalChat`/`MiniMaxH3AudioLock`'s `references`-socket naming.
- [ComfyUI-ResolutionSelector](https://github.com/zerohackz/ComfyUI-ResolutionSelector) (`ImageRatioSelectorZerohackz`) — `MiniMaxH3ResolutionSelector`'s duration→frame-count snapping and exact-ratio-from-reference-image sizing logic are adapted from this pack, with credit, as noted directly in the node's own source comments.

## Disclaimer

This pack is an independent, personal project. It is not affiliated with, endorsed by, or sponsored by MiniMax, any MiniMax H3 model creator, ComfyUI, or the creator(s) of the privately-distributed workflow that motivated the 5 MiniMax H3 nodes above. All best-effort/guessed behavior is labeled as such directly in each affected node's `DESCRIPTION` string, visible in the ComfyUI node info panel at runtime, in addition to this document.
