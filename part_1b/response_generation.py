# The pipeline passes current_state as system_action and the full dialogue state here.
# TODO: Keep response wording consistent with the numbered FSM actions below.
# TODO: Use config["reasoning_transparency"] to choose between concise and detailed responses.
# TODO: Use the selected restaurant's attributes and state.reasoning in the recommendation.
# TODO: Honor reasoning_transparency: show the inference chain when enabled and omit it
# when disabled, without changing the selected restaurant or recommendation.
# TODO: Implement no-match and restaurant-info responses using the current requirements,
# requested information, remaining candidates, and applicable FSM behavior.
# TODO: Add confirmation or contradiction responses only with matching FSM handling.

from part_1b.state import DialogueState


def generate_response(
    system_action: str,
    state: DialogueState,
    config: dict,
) -> str:
    """Return the response for a numbered FSM action using the current dialogue state."""

    if system_action == "1_welcome":
        return "Hello, welcome to the restaurant recommendation system. How may I help you?"

    if system_action == "2_ask_food":
        return "What kind of food would you like?"

    if system_action == "3_ask_area":
        return "What part of town would you prefer?"

    if system_action == "4_ask_price":
        return "What price range are you looking for?"

    if system_action == "5_no_match":
        # TODO: Include the active preferences and offer an appropriate way to revise them.
        #   Example from reference_dialog.txt: "Sorry, I couldn't find any restaurant matching food=indian, area=west, pricerange=moderate"
        return ""

    if system_action == "6_ask_additional_requirements":
        return "I found some restaurants matching your preferences. Do you have any additional requirements?"

    if system_action == "7_suggest_restaurant":
        # TODO: Describe the selected restaurant and optionally explain matching inferences.
        # When the config reasoning_transparency option is enabled, include the reasoning chain in the response.
        #   Example from reference_dialog.txt: "I recommend Zizzi, which is a moderate restaurant in the west part of town. It allows a long stay."
        # When it is disabled, omit the reasoning chain from the response.
        #   Example from reference_dialog.txt: "I recommend Zizzi"
        return ""

    if system_action == "8_give_info":
        # TODO: Identify whether the user asked for the phone, address, or postcode,
        # then return that field from the current recommendation.
        return ""

    if system_action == "9_goodbye":
        return "Goodbye!"

    return "How can I help you?"
