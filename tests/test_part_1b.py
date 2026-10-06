import json
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest

from part_1b.config import get_runtime_config
from part_1b.interaction_logging import create_interaction_log_entry, save_interaction_log
from part_1b.reasoning import apply_reasoning
from part_1b.restaurant_lookup import find_restaurants
from part_1b.slot_extraction import extract_slot_details, extract_slots, match_fallback
from part_1b import pipelines

ROOT = Path(__file__).resolve().parents[1]


def test_config_options():
    assert get_runtime_config()["fallback"] == "levenshtein"
    assert get_runtime_config("embeddings", False)["reasoning_transparency"] is False
    with pytest.raises(ValueError):
        get_runtime_config("other")


def test_multiple_slots_and_word_boundaries():
    config = get_runtime_config()
    assert extract_slots("Cheap Italian food in the center", config) == {
        "food": "italian", "area": "centre", "pricerange": "cheap"}
    assert extract_slots("north american food in the south", config) == {
        "food": "north american", "area": "south"}
    assert extract_slots("inexpensive food", config)["pricerange"] == "cheap"
    assert extract_slots("eastern", config) == {}


def test_any_is_different_from_missing():
    config = get_runtime_config()
    assert extract_slots("", config) == {}
    assert extract_slots("any", config) == {}
    assert extract_slots("I don't care", config, {"current_state": "4_ask_price"}) == {"pricerange": "dontcare"}
    assert extract_slots("any food, any area, any price", config) == {
        "food": "dontcare", "area": "dontcare", "pricerange": "dontcare"}


def test_spelling_needs_confirmation():
    config = get_runtime_config()
    result = extract_slot_details("spenish food in the centre", config)
    assert result["slots"] == {"area": "centre"}
    assert result["pending"] == [{"slot": "food", "given": "spenish", "value": "spanish"}]
    assert "food" not in extract_slots("spenish food", config)
    assert match_fallback("zzzzzzzzz", "food", config) is None
    assert match_fallback("french", "food", config) == "french"


def test_unknown_food_and_negation():
    config = get_runtime_config()
    result = extract_slot_details("I want Swedish food", config)
    assert not result["slots"] and not result["pending"]
    assert result["unrecognized"] == [{"slot": "food", "given": "swedish"}]
    result = extract_slot_details("not italian but chinese", config)
    assert result["slots"] == {"food": "chinese"}
    assert result["excluded"] == {"food": ["italian"]}
    result = extract_slots("not romantic and no assigned seats", config)
    assert result == {"romantic": False, "assigned_seats": False}


def test_all_rules_and_both_conflicts():
    row = {"food": "romanian", "pricerange": "cheap", "food quality": "good",
           "crowdedness": "busy", "length of stay": "long"}
    result = apply_reasoning(row, {"romantic": True})
    assert [r["id"] for r in result["rules"]] == [1, 2, 3, 4, 5, 6]
    assert result["properties"] == {"touristic": False, "assigned_seats": True,
                                     "children": False, "romantic": False}
    assert {c["property"] for c in result["conflicts"]} == {"romantic", "touristic"}
    assert result["matches_requirements"] is False
    assert apply_reasoning(row, {"romantic": False})["matches_requirements"] is True


def test_children_default_and_unknown_values():
    result = apply_reasoning({"length of stay": "short"}, {"children": True})
    assert result["properties"]["children"] is True
    assert result["defaults"] and not result["rules"]
    assert result["matches_requirements"] is True
    assert apply_reasoning({}, {"children": True})["matches_requirements"] is False
    assert apply_reasoning({}, {"romantic": False})["matches_requirements"] is False
    assert apply_reasoning({}, {})["matches_requirements"] is True


def test_lookup_on_real_data():
    data = pd.read_csv(ROOT / "data/restaurant_info_extended.csv", keep_default_na=False)
    matches = find_restaurants({"food": "indian", "area": "north", "pricerange": "moderate"}, data, [])
    assert {r["restaurantname"] for r in matches} == {"meghna", "the nirala", "graffiti", "tandoori palace"}
    excluded = find_restaurants({"food": "indian", "area": "north", "pricerange": "moderate"}, data, ["MEGHNA"])
    assert len(excluded) == 3 and all(r["restaurantname"] != "meghna" for r in excluded)
    assert find_restaurants({"food": "indian", "area": "west", "pricerange": "moderate"}, data, []) == []
    assert len(find_restaurants({"area": "dontcare"}, data, [])) == len(data)
    assert list(data["restaurantname"]) == [r["restaurantname"] for r in find_restaurants({}, data, [])]


def test_lookup_missing_values():
    data = pd.DataFrame([{"restaurantname": "a", "food": "italian", "area": np.nan,
                          "pricerange": "cheap", "phone": np.nan}])
    assert find_restaurants({"area": "north"}, data, []) == []
    assert find_restaurants({"area": "dontcare"}, data, [])[0]["phone"] == ""


def test_log_keeps_old_state_and_saves_json(tmp_path):
    state = {"requirements": {"food": "italian"}, "matches": [{"name": "a"}]}
    entry = create_interaction_log_entry({"dialogue_state": state, "score": np.float64(0.5), "missing": np.nan})
    state["requirements"]["food"] = "chinese"
    state["matches"].clear()
    path = tmp_path / "nested" / "log.json"
    save_interaction_log([entry], path)
    saved = json.loads(path.read_text())[0]
    assert saved["dialogue_state"]["requirements"]["food"] == "italian"
    assert saved["dialogue_state"]["matches"] == [{"name": "a"}]
    assert saved["score"] == 0.5 and saved["missing"] is None


def test_reference_file_coverage():
    dialogs = pipelines.read_reference_dialogs(ROOT / "data/reference_dialogs.txt")
    checks = json.loads((ROOT / "part_1b/reference_checks.json").read_text())
    assert set(dialogs) == set(range(1, 21))
    for number, utterances in dialogs.items():
        assert utterances == [item["utterance"] for item in checks[str(number)]]


def test_reference_checker_rejects_empty_response():
    state = pipelines.create_initial_state()
    errors = pipelines.check_reference_turn(state, "", {"action": "1_welcome"})
    assert "The system response is empty." in errors


def test_pipeline_filters_before_selection_and_keeps_config(monkeypatch):
    monkeypatch.setattr(pipelines, "predict_dialog_act", lambda model, text: "inform")
    data = pd.read_csv(ROOT / "data/restaurant_info_extended.csv", keep_default_na=False)
    state = pipelines.create_initial_state()
    state["requirements"] = {"food": "turkish", "area": "dontcare", "pricerange": "moderate"}
    state["additional_requirements_asked"] = True
    state, response, entry = pipelines.process_turn(state, "romantic", get_runtime_config(), data, random.Random(12))
    assert state["current_recommendation"]["restaurantname"] == "sesame restaurant and bar"
    assert state["reasoning"]["properties"]["romantic"] is True
    assert entry["configuration"]["reasoning_transparency"] is True


def test_changed_area_refreshes_candidates(monkeypatch):
    monkeypatch.setattr(pipelines, "predict_dialog_act", lambda model, text: "reqalts")
    data = pd.read_csv(ROOT / "data/restaurant_info_extended.csv", keep_default_na=False)
    state = pipelines.create_initial_state()
    state["requirements"] = {"food": "indian", "area": "north", "pricerange": "moderate"}
    state["additional_requirements_asked"] = True
    state["matches"] = find_restaurants(state["requirements"], data, [])
    state, _, _ = pipelines.process_turn(state, "another one in the west", get_runtime_config(), data, random.Random(12))
    assert state["requirements"]["area"] == "west"
    assert state["matches"] == []


def test_transparency_does_not_change_selection(monkeypatch):
    monkeypatch.setattr(pipelines, "predict_dialog_act", lambda model, text: "inform")
    data = pd.read_csv(ROOT / "data/restaurant_info_extended.csv", keep_default_na=False)
    initial = pipelines.create_initial_state()
    initial["additional_requirements_asked"] = True
    names = []
    for visible in (True, False):
        state, _, _ = pipelines.process_turn(deepcopy(initial), "any food any area any price",
                                            get_runtime_config(reasoning_transparency=visible), data, random.Random(12))
        names.append(state["current_recommendation"]["restaurantname"])
    assert names[0] == names[1]
