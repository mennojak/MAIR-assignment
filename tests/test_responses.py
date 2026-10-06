from copy import deepcopy

from part_1b.reasoning import apply_reasoning
from part_1b.response_generation import generate_response
from part_1b.state import create_initial_state


def restaurant_state():
    state = create_initial_state()
    state["current_recommendation"] = {
        "restaurantname": "example restaurant", "food": "italian", "pricerange": "cheap",
        "area": "centre", "phone": "01223 123456", "addr": "10 example street",
        "postcode": "CB1 1AA", "food quality": "good", "crowdedness": "busy",
        "length of stay": "long",
    }
    state["requirements"] = {"food": "italian", "area": "centre", "pricerange": "cheap"}
    state["reasoning"] = apply_reasoning(state["current_recommendation"], {})
    state["configuration"] = {"reasoning_transparency": True}
    return state


def test_recommendation_and_hidden_reasoning():
    state = restaurant_state()
    before = deepcopy(state)
    shown = generate_response("7_suggest_restaurant", state)
    assert state == before
    assert all(word in shown for word in ["example restaurant", "italian", "cheap", "centre"])
    assert "because" in shown
    state["configuration"]["reasoning_transparency"] = False
    hidden = generate_response("7_suggest_restaurant", state)
    assert all(word in hidden for word in ["example restaurant", "italian", "cheap", "centre"])
    assert "because" not in hidden and "rules" not in hidden and "busy" not in hidden


def test_conflict_has_a_clear_final_conclusion():
    state = restaurant_state()
    state["additional_requirements"]["romantic"] = False
    text = generate_response("7_suggest_restaurant", state)
    assert "busy" in text and "long stay" in text
    assert "negative conclusion priority" in text
    assert text.endswith("not romantic.")


def test_children_default_is_described_as_assumption():
    state = restaurant_state()
    state["current_recommendation"]["length of stay"] = "short"
    state["additional_requirements"]["children"] = True
    state["reasoning"] = apply_reasoning(state["current_recommendation"], {"children": True})
    assert "assume" in generate_response("7_suggest_restaurant", state)


def test_request_can_ask_for_several_fields():
    state = restaurant_state()
    state["last_utterance"] = "Can I get the phone number, address and postal code?"
    text = generate_response("8_give_info", state)
    assert "01223 123456" in text and "10 example street" in text and "CB1 1AA" in text
    state["current_recommendation"]["postcode"] = ""
    state["last_utterance"] = "post code"
    assert "postcode" in generate_response("8_give_info", state)
    assert "unknown" in generate_response("8_give_info", state)
    state["last_utterance"] = "address"
    state["current_recommendation"]["addr"] = None
    assert "unknown" in generate_response("8_give_info", state)


def test_other_information_and_unspecified_request():
    state = restaurant_state()
    for question, answer in [("price range", "cheap"), ("area", "centre"), ("food type", "italian")]:
        state["last_utterance"] = question
        assert answer in generate_response("8_give_info", state)
    state["last_utterance"] = "tell me something"
    assert "Would you like" in generate_response("8_give_info", state)


def test_no_match_uses_current_preferences():
    state = create_initial_state()
    state["requirements"] = {"food": "indian", "area": "west", "pricerange": "dontcare"}
    text = generate_response("5_no_match", state)
    assert "food=indian" in text and "area=west" in text
    assert "dontcare" not in text and "pricerange" not in text
    assert "change a preference" in text


def test_exhausted_options_are_not_confused_with_changed_area():
    state = restaurant_state()
    state["dialog_act"] = "reqalts"
    assert "all the options" in generate_response("5_no_match", state)
    state["requirements"]["area"] = "west"
    text = generate_response("5_no_match", state)
    assert "area=west" in text and "all the options" not in text


def test_no_match_explains_rejected_candidate_and_honors_switch():
    state = restaurant_state()
    state["additional_requirements"]["romantic"] = True
    state["rejected_candidates"] = [{"restaurantname": "example restaurant", "reasoning": state["reasoning"]}]
    state["current_recommendation"] = None
    text = generate_response("5_no_match", state)
    assert "busy" in text and "negative conclusion priority" in text
    state["configuration"]["reasoning_transparency"] = False
    text = generate_response("5_no_match", state)
    assert "busy" not in text and "rules" not in text


def test_missing_recommendation_and_unknown_preference():
    state = create_initial_state()
    assert "haven't recommended" in generate_response("8_give_info", state)
    assert "don't have" in generate_response("7_suggest_restaurant", state)
    state["slot_details"] = {"unrecognized": [{"slot": "food", "given": "swedish"}]}
    text = generate_response("2_ask_food", state)
    assert "not familiar with swedish" in text and "What kind of food" in text


def test_no_confirmation_prompt_without_a_confirmation_state():
    state = create_initial_state()
    state["slot_details"] = {"pending": [{"slot": "food", "given": "spenish", "value": "spanish"}]}
    assert generate_response("2_ask_food", state) == "What kind of food would you like?"
