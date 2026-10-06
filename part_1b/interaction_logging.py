# TODO: Match the per-turn record assembled in pipelines.py: utterance, dialog act,
# TODO: extracted slots, system action, dialogue state, response and configuration.
# TODO: Wire this helper into the pipeline, or remove it if entries stay inline.

def create_interaction_log_entry(interaction_data: dict) -> dict:
    """Normalize the per-turn fields currently assembled by the interaction pipeline."""
    pass


# TODO: Serialize the pipeline's list of per-turn records as JSON at the supplied path.
# TODO: Ensure nested dialogue state and configuration values are JSON-serializable.

import json
import math
from copy import deepcopy
from pathlib import Path


def create_interaction_log_entry(interaction_data):
    """Copy the turn before the state changes again."""
    return deepcopy(interaction_data)


def make_json_value(value):
    """Make numpy values and missing numbers usable in JSON."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[str(key)] = make_json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.append(make_json_value(item))
        return result
    if hasattr(value, "item"):
        return make_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Cannot save this value in the log: " + type(value).__name__)


def save_interaction_log(log_entries, output_path):
    """Save the conversation to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = make_json_value(log_entries)
    with path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
