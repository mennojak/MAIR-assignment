from part_1b.state import DialogueState, REQUIRED_SLOTS


def requested_additional_requirement_explanations(state: DialogueState):
    if not state.additional_requirements_asked or state.reasoning is None:
        return []

    requested = {name for name, value in state.additional_requirements.items() if value is not None}
    properties = state.reasoning.get("properties", {})
    explanations = [
        rule["reason"] for rule in state.reasoning.get("rules", [])
        if rule["property"] in requested
        and rule["value"] is properties.get(rule["property"])
    ]

    return explanations


def generate_response(system_action: str, state: DialogueState, config: dict):
    if system_action == "2_ask_food":
        return "What kind of food would you like?"

    if system_action == "3_ask_area":
        return "What part of town would you prefer?"

    if system_action == "4_ask_price":
        return "What price range are you looking for?"

    if system_action == "5_confirm_value":
        confirmation = state.pending_confirmation
        return f"Sorry, I don't know {confirmation['original']}. Did you mean {confirmation['suggestion']}?"

    if system_action == "6_no_match":
        preferences = []
        for slot in REQUIRED_SLOTS:
            value = state.requirements[slot]
            if value is not None and value != "dontcare":
                preferences.append(f"{slot}={value}")

        for requirement, value in state.additional_requirements.items():
            if value is not None:
                preferences.append(f"{requirement}={value}")

        if state.dialog_act == "reqmore":
            return "I'm afraid that's all the options I have. Would you like to change a preference?"

        if config["reasoning_transparency"]:
            response = "Sorry, I couldn't find any restaurant matching " + ", ".join(preferences)
            return response
        return "Sorry, I couldn't find any restaurant matching your preferences"

    if system_action == "7_ask_additional_requirements":
        return "I found some restaurants matching your preferences. Do you have any additional requirements?"

    if system_action == "8_suggest_restaurant":
        restaurant = state.current_recommendation

        restaurant_name = restaurant["restaurantname"]
        response = f"I recommend {restaurant_name}."

        if config["reasoning_transparency"]:
            food = restaurant.get("food")
            area = restaurant.get("area")
            price = restaurant.get("pricerange")

            # Price and cuisine are never unknown in the data, area is in some cases.
            additional_information = f"This is a {price} {food} restaurant"
            if area:
                if area == "unknown":
                    additional_information += "; its area is unknown"
                else:
                    additional_information += f" in the {area}"
            additional_information += "."
            response += " " + additional_information

            explanations = requested_additional_requirement_explanations(state)
            if explanations:
                response += " " + " ".join(explanations)

        return response

    if system_action == "9_give_info":
        restaurant = state.current_recommendation

        utterance = state.last_utterance.lower()
        restaurant_name = restaurant["restaurantname"]

        if "number" in utterance:
            phone_number = restaurant.get("phone")
            return f"The phone number of {restaurant_name} is {phone_number}."

        if "address" in utterance:
            address = restaurant.get("addr")
            return f"{restaurant_name} is located at {address}."

        if "code" in utterance:
            postcode = restaurant.get("postcode")
            return f"The postcode of {restaurant_name} is {postcode}."

        return f"Would you like the phone number, address or postcode of {restaurant_name}?"

    if system_action == "10_goodbye":
        return "Goodbye!"

    # State 1. Welcome is the default system action
    return "Hello, welcome to the restaurant recommendation system. How may I help you?"
