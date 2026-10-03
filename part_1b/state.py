REQUIRED_SLOTS = ("food", "area", "pricerange")
ADDITIONAL_REQUIREMENTS = ("touristic", "assigned_seats", "children", "romantic")


def create_initial_state() -> dict:
    return {
        "current_state": "1_welcome",
        "requirements": {slot: None for slot in REQUIRED_SLOTS},
        "additional_requirements": {slot: None for slot in ADDITIONAL_REQUIREMENTS},
        "additional_requirements_asked": False,
        "matches": [],
        "previous_recommendations": [],
        "current_recommendation": None,
    }


def update_state_fields(state, updates) -> dict:
    if updates.get("dialog_act") == "restart":
        state = create_initial_state()

    updated_state = state

    updated_state["requirements"] = {
        slot: state.get("requirements", {}).get(slot)
        for slot in REQUIRED_SLOTS
    }

    updated_state["additional_requirements"] = {
        slot: state.get("additional_requirements", {}).get(slot)
        for slot in ADDITIONAL_REQUIREMENTS
    }

    updated_state["matches"] = list(state.get("matches", []))

    updated_state["previous_recommendations"] = list(state.get("previous_recommendations", []))

    for name, value in updates.get("slots", {}).items():
        if name in REQUIRED_SLOTS:
            updated_state["requirements"][name] = value
        elif name in ADDITIONAL_REQUIREMENTS:
            updated_state["additional_requirements"][name] = value

    if state.get("current_state") == "6_ask_additional_requirements":
        updated_state["additional_requirements_asked"] = True

    for name, value in updates.items():
        if name != "slots":
            updated_state[name] = value

    return updated_state


def transition_state(state: dict, dialog_act: str) -> dict:
    dialog_act = dialog_act.lower()

    if dialog_act in ("bye", "thankyou"):
        state["current_state"] = "9_goodbye"
        return state

    if dialog_act == "request" and state.get("current_recommendation") is not None:
        state["current_state"] = "8_give_info"
        return state

    if state.get("current_state") == "8_give_info":
        state["current_state"] = "7_suggest_restaurant"

    if dialog_act == "reqalts":
        state["current_state"] = "7_suggest_restaurant" if state.get("matches") else "5_no_match"
        return state

    required_values = state.get("requirements", {})
    if required_values.get("food") is None:
        state["current_state"] = "2_ask_food"
        return state
    if required_values.get("area") is None:
        state["current_state"] = "3_ask_area"
        return state
    if required_values.get("pricerange") is None:
        state["current_state"] = "4_ask_price"
        return state

    matches = state.get("matches", [])
    if not matches:
        state["current_state"] = "5_no_match"
        return state

    if matches and not state.get("additional_requirements_asked", False):
        state["current_state"] = "6_ask_additional_requirements"
        return state

    state["current_state"] = "7_suggest_restaurant"
    return state