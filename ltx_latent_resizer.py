import torch
import torch.nn.functional as F

_RESIZE_METHODS = ["bilinear", "bicubic", "nearest", "area"]

# Each video model's causal VAE has its own temporal-compression constraint on
# frame count, expressed as "count % multiple == remainder". These are the
# published constraints for the model families this pack's other nodes touch;
# "Custom" lets you enter your own multiple/remainder for anything else.
_FRAME_PRESETS = {
    "none (spatial resize only)": None,
    "LTX-Video (8k+1 frames)": (8, 1),
    "Hunyuan Video (4k+1 frames)": (4, 1),
    "WAN 2.1 / 2.2 (4k+1 frames)": (4, 1),
    "Custom": "custom",
}
_FRAME_PRESET_NAMES = list(_FRAME_PRESETS.keys())


def _trim_frame_count(n, multiple, remainder):
    """Largest count <= n satisfying count % multiple == remainder. Only ever
    trims (never pads) -- there's no way to invent frames that aren't there."""
    if n <= remainder:
        return n
    return remainder + ((n - remainder) // multiple) * multiple


class LTXLatentResizer:
    """Rescales an IMAGE batch by scale_factor, rounds both dimensions up to a
    multiple of `multiple_of`, and (optionally) trims the batch dimension down
    to a valid frame count for a chosen video model's VAE. Despite the name,
    this runs on pixel-space IMAGE tensors, not LATENT tensors: it's meant to
    sit right before a VAEEncode, not after one."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05}),
            },
            "optional": {
                "multiple_of": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                                 "tooltip": "32 fits LTX and most video VAEs; change it if yours needs otherwise."}),
                "resize_method": (_RESIZE_METHODS, {"default": "bilinear"}),
                "video_preset": (_FRAME_PRESET_NAMES, {"default": _FRAME_PRESET_NAMES[0],
                                 "tooltip": "Trims the batch (frame) dimension down to the nearest valid "
                                            "count for the chosen model's VAE, e.g. LTX needs 8k+1 frames. "
                                            "'none' leaves the frame count untouched (spatial resize only, "
                                            "the original behavior of this node)."}),
                "custom_frame_multiple": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                                           "tooltip": "Only used when video_preset is 'Custom'."}),
                "custom_frame_remainder": ("INT", {"default": 1, "min": 0, "max": 63, "step": 1,
                                            "tooltip": "Only used when video_preset is 'Custom'. Must be "
                                                       "less than custom_frame_multiple."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height", "frame_count")
    FUNCTION = "calculate_and_resize"
    CATEGORY = "LTXVideo/Utils"
    DESCRIPTION = ("Resizes an IMAGE by scale_factor and rounds both dimensions up to a "
                   "multiple of `multiple_of`. Run this before VAEEncode -- it operates on "
                   "pixels, not latents, despite the node's name. Optionally also trims the "
                   "batch/frame dimension down to a valid count for LTX-Video, Hunyuan Video, "
                   "WAN, or a custom multiple-of-N+remainder formula via `video_preset`.")

    def calculate_and_resize(self, image, scale_factor, multiple_of=32, resize_method="bilinear",
                              video_preset="none (spatial resize only)",
                              custom_frame_multiple=8, custom_frame_remainder=1):
        # ComfyUI Image tensors are formatted as: [B, H, W, C]
        _, original_height, original_width, _ = image.shape

        # Apply the scaling factor
        scaled_width = original_width * scale_factor
        scaled_height = original_height * scale_factor

        # Ceiling division trick to find the next nearest integer divisible by multiple_of
        m = max(1, multiple_of)
        final_width = int(((scaled_width + m - 1) // m) * m)
        final_height = int(((scaled_height + m - 1) // m) * m)

        # Enforce safe minimum bounds
        final_width = max(m, final_width)
        final_height = max(m, final_height)

        # PyTorch interpolate expects [B, C, H, W] -> shuffle channels
        img_permuted = image.permute(0, 3, 1, 2)

        # align_corners is only a valid kwarg for bilinear/bicubic -- nearest/area reject it
        interp_kwargs = {"align_corners": False} if resize_method in ("bilinear", "bicubic") else {}
        img_resized = F.interpolate(
            img_permuted,
            size=(final_height, final_width),
            mode=resize_method,
            **interp_kwargs,
        )

        # Reshuffle back to standard ComfyUI layout [B, H, W, C]. Clamp because
        # bicubic (unlike bilinear) can overshoot slightly past [0,1] at sharp edges.
        output_image = img_resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        preset = _FRAME_PRESETS[video_preset]
        if preset is not None:
            if preset == "custom":
                mult = max(1, custom_frame_multiple)
                rem = max(0, custom_frame_remainder) % mult
            else:
                mult, rem = preset
            frame_count_in = output_image.shape[0]
            frame_count = _trim_frame_count(frame_count_in, mult, rem)
            if frame_count < frame_count_in:
                output_image = output_image[:frame_count]
        else:
            frame_count = output_image.shape[0]

        return (output_image, final_width, final_height, frame_count)
