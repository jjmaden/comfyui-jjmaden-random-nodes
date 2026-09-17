import comfy.hooks
import comfy.utils
import folder_paths

# ExecutionBlocker is ComfyUI's real mechanism for "this output is invalid,
# don't run whatever's downstream of it" (see dynamic_image_switch.py for the
# original use of this exact pattern in this pack). Used here for whichever of
# positive/negative wasn't actually connected, so a workflow that accidentally
# wires up the unused output doesn't get a silent None conditioning.
try:
    from comfy_execution.graph_utils import ExecutionBlocker
    _HAS_EXECUTION_BLOCKER = True
except Exception:
    try:
        from comfy_execution.graph import ExecutionBlocker
        _HAS_EXECUTION_BLOCKER = True
    except Exception:
        ExecutionBlocker = None
        _HAS_EXECUTION_BLOCKER = False


_NO_LORA = "None"
_NUM_SLOTS = 8


class PowerLoraHooks:
    """Stacks up to NUM_SLOTS LoRAs into one HookGroup instead of patching MODEL/CLIP
    directly like rgthree-comfy's Power Lora Loader does (idea credited there -- see
    readme.md), and optionally applies the accumulated hooks to positive/negative
    CONDITIONING. Fixed-slot UI (plain widgets) rather than rgthree's dynamic,
    custom-canvas-drawn rows -- see the "Power Lora Hooks" readme section for why."""

    NUM_SLOTS = _NUM_SLOTS
    CATEGORY = "advanced/hooks/create"
    EXPERIMENTAL = True  # built directly on ComfyUI's own EXPERIMENTAL hooks API

    def __init__(self):
        # Per-instance cache of loaded lora state dicts, keyed by absolute file
        # path. Unlike CreateHookLora's single-slot self.loaded_lora (fine for a
        # node that only ever handles one lora), this node juggles NUM_SLOTS
        # loras at once, so a dict avoids re-reading every slot's file from disk
        # whenever only one slot actually changed between runs. Left unbounded
        # since it naturally caps at "how many distinct loras this node's own
        # dropdowns have ever been set to", not the whole loras folder.
        self._loaded = {}

    @classmethod
    def INPUT_TYPES(cls):
        lora_names = [_NO_LORA] + folder_paths.get_filename_list("loras")
        optional = {
            "prev_hooks": ("HOOKS", {
                "tooltip": "Optional. Hooks from an upstream hook-creating node (another "
                           "PowerLoraHooks, or a native Create Hook LoRA / Create Hook Model as "
                           "LoRA, etc.) -- combined with this node's own enabled slots, same "
                           "chaining convention as ComfyUI's native hook-creating nodes."}),
            "positive": ("CONDITIONING", {
                "tooltip": "Optional. If connected, the accumulated hooks are applied to it "
                           "(comfy.hooks.set_hooks_for_conditioning) and it's passed through "
                           "'positive' below. If not connected, 'positive' out is blocked."}),
            "negative": ("CONDITIONING", {
                "tooltip": "Optional. Same as 'positive', for the negative conditioning."}),
        }
        for i in range(1, cls.NUM_SLOTS + 1):
            optional[f"lora_{i}_enabled"] = ("BOOLEAN", {"default": True})
            optional[f"lora_{i}"] = (lora_names, {"default": _NO_LORA})
            optional[f"lora_{i}_strength_model"] = ("FLOAT", {
                "default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01})
            optional[f"lora_{i}_strength_clip"] = ("FLOAT", {
                "default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("HOOKS", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("hooks", "positive", "negative")
    FUNCTION = "build_hooks"
    DESCRIPTION = (
        "Fixed-slot alternative to stacking multiple LoraLoader nodes: up to 8 LoRAs, each with "
        "its own enable toggle and model/clip strengths, accumulated into a single HOOKS output "
        "via comfy.hooks.create_hook_lora + HookGroup.clone_and_combine, instead of patching a "
        "MODEL/CLIP directly. An optional prev_hooks input chains in hooks from an upstream node "
        "(another PowerLoraHooks, or a native Create Hook LoRA/Model as LoRA) -- same convention "
        "as ComfyUI's own hook-creating nodes, useful once you need more than 8 LoRAs at once. "
        "Optional positive/negative CONDITIONING inputs get the accumulated "
        "hooks applied (additively, via comfy.hooks.set_hooks_for_conditioning) and are passed "
        "through -- KSampler resolves hooks found on a conditioning against the model "
        "automatically at sample time (comfy/samplers.py's calc_cond_batch), so no separate "
        "'apply hooks to model' node is needed. This also means positive and negative can each "
        "carry a different set of LoRAs in the same sampling run, which plain MODEL/CLIP "
        "patching can't do without loading the model twice. A positive/negative output whose "
        "matching input wasn't connected is blocked (ExecutionBlocker) rather than silently "
        "passing None downstream. Idea (a compact, multi-lora node with per-row enable/strength) "
        "credited to rgthree-comfy's Power Lora Loader -- see readme.md; this node is an "
        "independent implementation built around ComfyUI's native hooks API instead of that "
        "node's direct model/clip patching and fully dynamic custom-drawn rows.")

    def _load_lora(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        cached = self._loaded.get(lora_path)
        if cached is not None:
            return cached
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        self._loaded[lora_path] = lora_sd
        return lora_sd

    def build_hooks(self, positive=None, negative=None, prev_hooks=None, **kwargs):
        # Same convention as CreateHookLora: start from prev_hooks itself (never
        # mutated in place -- clone_and_combine below always returns a new
        # object) rather than an empty group, so chained PowerLoraHooks nodes
        # (or a native Create Hook LoRA feeding into this one) accumulate.
        hooks = prev_hooks if prev_hooks is not None else comfy.hooks.HookGroup()
        for i in range(1, self.NUM_SLOTS + 1):
            if not kwargs.get(f"lora_{i}_enabled", True):
                continue
            lora_name = kwargs.get(f"lora_{i}", _NO_LORA)
            if not lora_name or lora_name == _NO_LORA:
                continue
            strength_model = kwargs.get(f"lora_{i}_strength_model", 1.0)
            strength_clip = kwargs.get(f"lora_{i}_strength_clip", 1.0)
            if strength_model == 0 and strength_clip == 0:
                continue
            lora_sd = self._load_lora(lora_name)
            row_hooks = comfy.hooks.create_hook_lora(
                lora=lora_sd, strength_model=strength_model, strength_clip=strength_clip)
            hooks = hooks.clone_and_combine(row_hooks)

        def _apply(cond):
            if cond is not None:
                return comfy.hooks.set_hooks_for_conditioning(cond, hooks)
            return ExecutionBlocker(None) if _HAS_EXECUTION_BLOCKER else None

        return (hooks, _apply(positive), _apply(negative))
