import os
import random

from ._file_utils import mtime_or_nan
from ._file_utils import strip_wrapping as _strip_wrapping

# Cache of {resolved_path: (mtime, lines)} so a large wildcard file isn't
# re-read and re-split on every single execution -- only when its mtime
# actually changes (the same signal IS_CHANGED already tracks below).
_line_cache = {}


def _load_lines(path):
    mtime = mtime_or_nan(path)
    cached = _line_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f
                     if line.strip() and not line.strip().startswith('#')]
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"RandomLineFromFile: '{path}' isn't valid UTF-8 text ({exc}). Re-save it as "
            f"UTF-8 -- a file saved as UTF-16 (a common Notepad default on Windows) or "
            f"containing stray non-UTF-8 bytes will hit this."
        ) from exc
    except OSError as exc:
        raise ValueError(f"RandomLineFromFile: couldn't read '{path}': {exc}") from exc

    _line_cache[path] = (mtime, lines)
    return lines


class RandomLineFromFile:
    """
    The ultimate wildcard loader. Decouples prompt randomization
    from KSampler randomization, allowing all 4 matrix variations natively.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_path": ("STRING", {"multiline": False, "placeholder": "Path to your text file"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # Connect to KSampler
                "prompt_mode": (["Follow KSampler/Seed Field", "Use Dedicated Prompt Seed", "True Independent Random"],),
                "prompt_fixed_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # Locks prompt separately
                "lora_trigger": ("STRING", {"multiline": False, "placeholder": "Text to replace [trigger]"}),
                "prefix_text": ("STRING", {"multiline": True, "placeholder": "Text to add to the beginning"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("random_line", "seed_output")
    FUNCTION = "get_random_line"
    CATEGORY = "My_Custom_Nodes"
    DESCRIPTION = ("Picks one random line from a text file using its own seed, independent of "
                   "the KSampler's. Uses a private random.Random instance rather than reseeding "
                   "Python's global random module, so it won't affect -- or be affected by -- "
                   "any other node in the same ComfyUI process that also calls random.*. Blank "
                   "lines and lines starting with '#' are skipped, so you can comment out entries "
                   "in your wildcard file. The parsed line list is cached by the file's mtime, so "
                   "large wildcard files aren't re-read from disk on every single execution.")

    @classmethod
    def VALIDATE_INPUTS(cls, text_path, seed, prompt_mode, prompt_fixed_seed, lora_trigger, prefix_text):
        if not _strip_wrapping(text_path):
            return "RandomLineFromFile: text_path is empty."
        return True

    def get_random_line(self, text_path, seed, prompt_mode, prompt_fixed_seed, lora_trigger, prefix_text):
        cleaned_path = _strip_wrapping(text_path)

        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(f"RandomLineFromFile: '{cleaned_path}' does not exist.")

        lines = _load_lines(cleaned_path)

        if not lines:
            raise ValueError(f"RandomLineFromFile: '{cleaned_path}' has no non-empty lines.")

        # A private RNG instance -- reseeding the global `random` module (the
        # original bug here) would silently change the behavior of every
        # other node in this process that also calls random.*, and vice versa.
        rng = random.Random()

        # --- The Ultimate Matrix Execution ---
        if prompt_mode == "Follow KSampler/Seed Field":
            # Prompt changes whenever the main KSampler seed changes
            rng.seed(seed)
        elif prompt_mode == "Use Dedicated Prompt Seed":
            # Prompt locks onto this specific seed, ignoring the KSampler seed completely
            rng.seed(prompt_fixed_seed)
        else:
            # True Independent Random: Prompt randomizes completely on its own system entropy
            rng.seed(os.urandom(16))

        random_line = rng.choice(lines)

        if "[trigger]" in random_line:
            random_line = random_line.replace("[trigger]", lora_trigger)

        final_line = f"{prefix_text} {random_line}" if prefix_text else random_line

        return (final_line, seed)

    @classmethod
    def IS_CHANGED(s, text_path, seed, prompt_mode, prompt_fixed_seed, lora_trigger, prefix_text):
        if prompt_mode == "True Independent Random":
            return float("NaN")

        # Track the wildcard file's own mtime too -- without this, editing the
        # .txt file's contents without touching the seed served a stale cached
        # line, which fights the exact hand-editing workflow this node's own
        # readme walks users through.
        cleaned_path = _strip_wrapping(text_path)
        mtime = mtime_or_nan(cleaned_path)  # nan (missing/unreadable) lets get_random_line raise the real error

        if prompt_mode == "Use Dedicated Prompt Seed":
            return f"{seed}-{prompt_fixed_seed}-{mtime}"  # Tracks both seed states + file content
        return f"{seed}-{mtime}"
