class DualModelDPDTSwitch:
    """
    A DPDT-style switch for ComfyUI.
    Safely toggles between two pairs of Models (High/Low) without string
    configuration mismatches.

    Uses ComfyUI's lazy-input mechanism (`"lazy": True` + check_lazy_status)
    so only the SELECTED pair's upstream loaders actually run. Previously all
    four inputs were plain (non-lazy) required MODEL sockets, which meant
    ComfyUI executed every node feeding all four of them on every run --
    both the T2V and I2V GGUF sets loaded regardless of which one `switch`
    picked, defeating the point of a switch for a pair of large checkpoints.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Cleaned up naming layout to protect against logic discrepancies
                "switch": (["Set_A (T2V GGUF)", "Set_B (I2V GGUF)"], {"default": "Set_A (T2V GGUF)"}),
                "set_a_high": ("MODEL", {"lazy": True}),
                "set_a_low": ("MODEL", {"lazy": True}),
                "set_b_high": ("MODEL", {"lazy": True}),
                "set_b_low": ("MODEL", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("MODEL_HIGH", "MODEL_LOW")
    FUNCTION = "toggle_models"
    CATEGORY = "utils"
    DESCRIPTION = ("Toggles between two MODEL pairs (e.g. a T2V GGUF set and an I2V GGUF set). "
                   "Both pairs must still be wired -- this is a lazy switch, not an optional one "
                   "-- but only the selected pair's upstream loaders actually execute/load.")

    def check_lazy_status(self, switch, set_a_high=None, set_a_low=None, set_b_high=None, set_b_low=None):
        values = {"set_a_high": set_a_high, "set_a_low": set_a_low,
                  "set_b_high": set_b_high, "set_b_low": set_b_low}
        needed = ["set_a_high", "set_a_low"] if switch == "Set_A (T2V GGUF)" else ["set_b_high", "set_b_low"]
        # Only ask ComfyUI to evaluate the branch we're actually going to use --
        # this is what skips loading the other model set entirely.
        return [name for name in needed if values[name] is None]

    def toggle_models(self, switch, set_a_high=None, set_a_low=None, set_b_high=None, set_b_low=None):
        # Match string explicitly to the chosen menu option array
        if switch == "Set_A (T2V GGUF)":
            if set_a_high is None or set_a_low is None:
                raise ValueError("DualModelDPDTSwitch: 'Set_A (T2V GGUF)' is selected but "
                                  "set_a_high/set_a_low aren't both connected.")
            return (set_a_high, set_a_low)
        else:
            if set_b_high is None or set_b_low is None:
                raise ValueError("DualModelDPDTSwitch: 'Set_B (I2V GGUF)' is selected but "
                                  "set_b_high/set_b_low aren't both connected.")
            return (set_b_high, set_b_low)
