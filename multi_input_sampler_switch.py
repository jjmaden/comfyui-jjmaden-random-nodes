class MultiInputSamplerSwitch:
    """
    A robust 5-Way Multiplexer for Sampler inputs.
    Selects one set of (Positive, Negative, Latent) to pass through safely,
    falling back to set_1 for anything the selected set doesn't have wired.

    Sets 2-5 are lazy (`"lazy": True`), so only the selected set's upstream
    CONDITIONING/LATENT actually gets computed -- previously all 5 sets ran
    on every queue regardless of select_set. set_1 stays required/non-lazy
    since it's also the fallback for a partially-wired selection.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select_set": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "set_1_pos": ("CONDITIONING",),
                "set_1_neg": ("CONDITIONING",),
                "set_1_latent": ("LATENT",),
            },
            "optional": {
                "set_2_pos": ("CONDITIONING", {"lazy": True}),
                "set_2_neg": ("CONDITIONING", {"lazy": True}),
                "set_2_latent": ("LATENT", {"lazy": True}),
                "set_3_pos": ("CONDITIONING", {"lazy": True}),
                "set_3_neg": ("CONDITIONING", {"lazy": True}),
                "set_3_latent": ("LATENT", {"lazy": True}),
                "set_4_pos": ("CONDITIONING", {"lazy": True}),
                "set_4_neg": ("CONDITIONING", {"lazy": True}),
                "set_4_latent": ("LATENT", {"lazy": True}),
                "set_5_pos": ("CONDITIONING", {"lazy": True}),
                "set_5_neg": ("CONDITIONING", {"lazy": True}),
                "set_5_latent": ("LATENT", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE", "LATENT")
    FUNCTION = "switch_inputs"
    CATEGORY = "utils"
    DESCRIPTION = ("Picks one of 5 (positive, negative, latent) sets. Unselected sets 2-5 are "
                   "lazy and won't be computed at all. If the selected set (2-5) is missing any "
                   "of its three inputs, this falls back to set_1 for whatever's missing.")

    def check_lazy_status(self, select_set, set_1_pos, set_1_neg, set_1_latent, **kwargs):
        if select_set == 1:
            return []
        prefix = f"set_{select_set}_"
        needed = [f"{prefix}pos", f"{prefix}neg", f"{prefix}latent"]
        return [name for name in needed if kwargs.get(name) is None]

    def switch_inputs(self, select_set, set_1_pos, set_1_neg, set_1_latent, **kwargs):
        if select_set == 1:
            return (set_1_pos, set_1_neg, set_1_latent)

        prefix = f"set_{select_set}_"
        pos = kwargs.get(f"{prefix}pos")
        neg = kwargs.get(f"{prefix}neg")
        lat = kwargs.get(f"{prefix}latent")

        if pos is None or neg is None or lat is None:
            print(f"[MultiInputSamplerSwitch] select_set={select_set} isn't fully wired "
                  f"(pos/neg/latent) -- falling back to set_1 for whatever's missing.")

        pos = pos if pos is not None else set_1_pos
        neg = neg if neg is not None else set_1_neg
        lat = lat if lat is not None else set_1_latent

        return (pos, neg, lat)
