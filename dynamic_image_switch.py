import torch

# ExecutionBlocker is ComfyUI's real mechanism for "this output is invalid,
# don't run whatever's downstream of it" -- confirmed present in
# comfy_execution/graph_utils.py, and its own docstring literally describes
# this node's use case ("a node with multiple possible outputs, some of
# which are invalid and should not be used"). Using it means an unselected
# path doesn't just receive an empty image -- the downstream nodes on that
# path never execute at all, so something that does image[0] instead of
# image[:1] won't crash on a path you didn't pick.
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


class DynamicImageRouter:
    """
    A stable router for ComfyUI images.
    Routes an input image batch to exactly one of 10 output paths. The other
    9 paths block their own downstream branch from executing (via
    ExecutionBlocker) rather than just handing it an empty image.
    """

    NUM_PATHS = 10

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "select_path": ("INT", {"default": 1, "min": 1, "max": s.NUM_PATHS, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * NUM_PATHS
    RETURN_NAMES = tuple(f"out_{i}" for i in range(1, NUM_PATHS + 1))
    FUNCTION = "route_images"
    CATEGORY = "utils"
    DESCRIPTION = ("Routes one image batch to exactly one of 10 numbered outputs, selected by "
                   "select_path. Unselected outputs use ExecutionBlocker so nothing downstream "
                   "of them runs at all -- if your ComfyUI version predates ExecutionBlocker, "
                   "this falls back to handing them an empty (0-batch) image instead.")

    def route_images(self, images, select_path):
        actual_index = max(0, min(select_path - 1, self.NUM_PATHS - 1))

        if _HAS_EXECUTION_BLOCKER:
            # Sharing one ExecutionBlocker instance across slots is fine -- it
            # blocks downstream execution outright, there's no tensor to alias.
            outputs = [ExecutionBlocker(None) for _ in range(self.NUM_PATHS)]
        else:
            # Fallback for older ComfyUI without ExecutionBlocker: a genuinely
            # empty batch per slot (0 images, 64x64 so any node that still needs
            # H/W doesn't divide by zero). Separate tensor instances -- sharing
            # one would mean an in-place write on one unselected branch's tensor
            # is visible on every other unselected branch too.
            outputs = [torch.empty((0, 64, 64, 3), dtype=torch.float32) for _ in range(self.NUM_PATHS)]

        outputs[actual_index] = images
        return tuple(outputs)
