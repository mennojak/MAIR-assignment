# TODO: add confirmation replies when the state machine handles them.

import re


PROPERTY_NAMES = {
    "touristic": ("not touristic", "touristic"),
    "assigned_seats": ("without assigned seats", "with assigned seats"),
    "children": ("not suitable for children", "suitable for children"),
    "romantic": ("not romantic", "romantic"),
}


def describe_preferences(state):
    parts = []
    for slot in ("food", "area", "pricerange"):
        value = state.get("requirements", {}).get(slot)
        if value not in (None, "", "dontcare"):
            parts.append(slot + "=" + str(value))
    for slot, value in state.get("additional_requirements", {}).items():
        if slot in PROPERTY_NAMES and isinstance(value, bool):
            parts.append(PROPERTY_NAMES[slot][value])
    if not parts:
        return "your preferences"
    return ", ".join(parts)


def explain_reasoning(reasoning, requirements):
    wanted = []
    for slot, value in requirements.items():
        if isinstance(value, bool):
            wanted.append(slot)
    sentences = []
    for rule in reasoning.get("rules", []):
        if wanted and rule["property"] not in wanted:
            continue
        sentences.append(rule["reason"])
    if not wanted or "children" in wanted:
        sentences.extend(reasoning.get("defaults", []))
    for conflict in reasoning.get("conflicts", []):
        slot = conflict["property"]
        if wanted and slot not in wanted:
            continue
        conclusion = PROPERTY_NAMES[slot][0]
        sentences.append("These rules disagree. We give the negative conclusion priority, "
                         "so we treat this restaurant as " + conclusion + ".")
    for slot in wanted:
        if slot not in PROPERTY_NAMES:
            continue
        value = reasoning.get("properties", {}).get(slot)
        if value is None:
            sentences.append("The available rules do not tell us whether this restaurant is "
                             + PROPERTY_NAMES[slot][1] + ".")
    return " ".join(sentences)


def generate_response(system_action, state):
    restaurant = state.get("current_recommendation") or {}
    additional = state.get("additional_requirements", {})
    show_reasoning = state.get("configuration", {}).get("reasoning_transparency", True)

    if system_action == "1_welcome":
        return "Hello, welcome to the restaurant recommendation system. How may I help you?"

    questions = {
        "2_ask_food": "What kind of food would you like?",
        "3_ask_area": "What part of town would you prefer?",
        "4_ask_price": "What price range are you looking for?",
    }
    if system_action in questions:
        unknown = state.get("slot_details", {}).get("unrecognized", [])
        if unknown:
            values = []
            for item in unknown:
                values.append(item["given"])
            given = ", ".join(values)
            return "Sorry, I am not familiar with " + given + ". " + questions[system_action]
        return questions[system_action]

    if system_action == "5_no_match":
        if state.get("dialog_act") == "request" and not restaurant:
            return "I haven't recommended a restaurant yet. Would you like to change a preference?"

        # Check whether the user changed any preferences.
        same_preferences = bool(restaurant)
        for slot, value in state.get("requirements", {}).items():
            if value not in (None, "", "dontcare") and restaurant.get(slot) != value:
                same_preferences = False
        properties = state.get("reasoning", {}).get("properties", {})
        for slot, value in additional.items():
            if not isinstance(value, bool):
                continue
            if properties.get(slot) is not value:
                same_preferences = False
        if same_preferences and state.get("dialog_act") in ("reqalts", "reqmore"):
            return "I'm afraid that's all the options I have. Would you like to change a preference?"

        preferences = describe_preferences(state)
        response = "Sorry, I couldn't find a remaining restaurant matching " + preferences + "."
        rejected = state.get("rejected_candidates", [])
        if rejected:
            response = "I couldn't find a remaining restaurant that meets " + preferences + "."
            if show_reasoning:
                # Showing one rejected restaurant is enough here.
                example = rejected[0]
                explanation = explain_reasoning(example["reasoning"], additional)
                if explanation:
                    response += " For example, for " + example["restaurantname"] + ": " + explanation
        return response + " Would you like to change a preference?"

    if system_action == "6_ask_additional_requirements":
        return ("I found some restaurants matching your preferences. Do you have any additional "
                "requirements, such as being romantic, touristic, suitable for children, "
                "or having assigned seats?")

    if system_action == "7_suggest_restaurant":
        if not restaurant:
            return "I don't have a restaurant to recommend yet. Would you like to change a preference?"
        response = "I recommend " + restaurant["restaurantname"] + "."
        if restaurant.get("food"):
            response += " It serves " + restaurant["food"] + " food."
        if restaurant.get("pricerange"):
            response += " It is in the " + restaurant["pricerange"] + " price range."
        if restaurant.get("area"):
            response += " It is in the " + restaurant["area"] + " part of town."
        if show_reasoning:
            explanation = explain_reasoning(state.get("reasoning", {}), additional)
            if explanation:
                response += " " + explanation
        return response

    if system_action == "8_give_info":
        if not restaurant:
            return "I haven't recommended a restaurant yet."
        text = state.get("last_utterance", "").lower()
        name = restaurant["restaurantname"]
        replies = []
        if re.search(r"\b(phone|telephone|number|contact)\b", text):
            phone = restaurant.get("phone") or "unknown"
            replies.append("The phone number of " + name + " is " + phone + ".")
        if re.search(r"\b(address|addr|located|location|where)\b", text):
            address = restaurant.get("addr")
            if address:
                replies.append(name + " is located at " + address + ".")
            else:
                replies.append("The address of " + name + " is unknown.")
        if re.search(r"\b(post\s*code|postal\s*code|zip\s*code|code)\b", text):
            postcode = restaurant.get("postcode") or "unknown"
            replies.append("The postcode of " + name + " is " + postcode + ".")
        if re.search(r"\b(price|cost|expensive|cheap|moderate)\b", text):
            value = restaurant.get("pricerange") or "unknown"
            replies.append("The price range of " + name + " is " + value + ".")
        if re.search(r"\b(area|part of town)\b", text):
            value = restaurant.get("area") or "unknown"
            replies.append("The area of " + name + " is " + value + ".")
        if re.search(r"\b(food|cuisine)\b", text):
            value = restaurant.get("food") or "unknown"
            replies.append("The food type of " + name + " is " + value + ".")
        if replies:
            return " ".join(replies)
        return "Would you like the phone number, address or postcode of " + name + "?"

    if system_action == "9_goodbye":
        return "Goodbye!"

    return "Sorry, I didn't understand that. Could you say it another way?"
