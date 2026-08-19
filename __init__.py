# Import the classes from your separate files
from .dynamic_image_switch import DynamicImageRouter
from .model_switches import DualModelDPDTSwitch
from .multi_input_sampler_switch import MultiInputSamplerSwitch
from .image_hub import UniversalImageHub
from .ltx_latent_resizer import LTXLatentResizer
from .random_text_loader import RandomLineFromFile  # <-- 1. Import your new wildcard node
from .minimax_h3_extra_nodes import (
    MiniMaxH3UnifiedToVideo,
    MiniMaxH3ConcatAVLatent,
    MiniMaxH3ResolutionSelector,
    MiniMaxH3AudioLock,
    MiniMaxH3MultimodalChat,
)

# Map the internal Python class to the "ID" ComfyUI uses
NODE_CLASS_MAPPINGS = {
    "DynamicImageRouter": DynamicImageRouter,
    "DualModelDPDTSwitch": DualModelDPDTSwitch,
    "MultiInputSamplerSwitch": MultiInputSamplerSwitch,
    "UniversalImageHub": UniversalImageHub,
    "LTXLatentResizer": LTXLatentResizer,
    "RandomLineFromFile": RandomLineFromFile,  # <-- 2. Register the mapping ID
    # Best-effort MiniMax H3 nodes -- see minimax_h3_extra_nodes.py's
    # module docstring for what's genuinely real vs. reverse-engineered here.
    "MiniMaxH3UnifiedToVideo": MiniMaxH3UnifiedToVideo,
    "MiniMaxH3ConcatAVLatent": MiniMaxH3ConcatAVLatent,
    "MiniMaxH3ResolutionSelector": MiniMaxH3ResolutionSelector,
    "MiniMaxH3AudioLock": MiniMaxH3AudioLock,
    "MiniMaxH3MultimodalChat": MiniMaxH3MultimodalChat,
}

# Map the "ID" to the friendly name that appears in the right-click menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicImageRouter": "Dynamic Image Router",
    "DualModelDPDTSwitch": "Model Switch",
    "MultiInputSamplerSwitch": "Sampler Switch",
    "UniversalImageHub": "Image Hub",
    "LTXLatentResizer": "LTX Latent Resizer (Divisible by 32)",
    "RandomLineFromFile": "Random Line From File (Wildcard)",  # <-- 3. Register the menu label
    "MiniMaxH3UnifiedToVideo": "MiniMax H3 Unified To Video (best-effort)",
    "MiniMaxH3ConcatAVLatent": "MiniMax H3 Concat AV Latent (best-effort)",
    "MiniMaxH3ResolutionSelector": "MiniMax H3 Resolution Selector (best-effort)",
    "MiniMaxH3AudioLock": "MiniMax H3 Audio Lock (best-effort, unverified)",
    "MiniMaxH3MultimodalChat": "MiniMax H3 Multimodal Chat (local LLM)",
}

# Tell ComfyUI where the JavaScript for the 'Paste' and 'Refresh' buttons lives
WEB_DIRECTORY = "./web"

# Standard export for ComfyUI custom node packages
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]