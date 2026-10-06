import json
import math
from copy import deepcopy
from pathlib import Path


def create_interaction_log_entry(interaction_data):
    return deepcopy(interaction_data)


def make_json_value(value):
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
    return value


def save_interaction_log(log_entries, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = make_json_value(log_entries)
    with path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
