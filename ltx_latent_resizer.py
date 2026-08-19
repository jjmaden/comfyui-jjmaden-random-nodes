import torch
import torch.nn.functional as F

_RESIZE_METHODS = ["bilinear", "bicubic", "nearest", "area"]


class LTXLatentResizer:
    """Rescales an IMAGE batch by scale_factor and rounds both dimensions up
    to a multiple of `multiple_of` (32 by default -- what LTX and most other
    video VAEs require). Despite the name, this runs on pixel-space IMAGE
    tensors, not LATENT tensors: it's meant to sit right before a VAEEncode,
    not after one."""

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
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "calculate_and_resize"
    CATEGORY = "LTXVideo/Utils"
    DESCRIPTION = ("Resizes an IMAGE by scale_factor and rounds both dimensions up to a "
                   "multiple of `multiple_of`. Run this before VAEEncode -- it operates on "
                   "pixels, not latents, despite the node's name.")

    def calculate_and_resize(self, image, scale_factor, multiple_of=32, resize_method="bilinear"):
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

        return (output_image, final_width, final_height)
