import pandas as pd
from part_1a.user_interaction import predict_dialog_act

from part_1b.state import REQUIRED_SLOTS, DialogueState
from part_1b.slot_extraction import extract_slots
from part_1b.restaurant_lookup import find_restaurants
from part_1b.reasoning import apply_reasoning
from part_1b.response_generation import generate_response
from part_1b.interaction_logging import save_interaction_log


# TODO: Apply reasoning to every candidate before selection, filter by requested additional
#       requirements, and keep the explanation available to response generation. Belongs to the reasoning.py tasks
def run_interaction_pipeline(config: dict) -> None:
    """Run the complete Part 1b restaurant recommendation dialogue."""
    print("\nStarting the restaurant interaction pipeline for Part 1b...\n")

    restaurant_info_df = pd.read_csv("data/restaurant_info_extended.csv")

    model_path = "models/model_frozen_embeddings_grouped_SVM"
    # Run first prediction to load the model avoiding the first response load time
    predict_dialog_act(model_path, "hello")

    # State 1. Welcome
    state = DialogueState()

    interaction_log = []

    print("Tell the system goodbye or thank you to quit (or alternatively type 'exit').\n")
    print("----------------------------------------\n")
    print("Hello, welcome to the restaurant recommendation system. How may I help you?")


    # Interaction loop which keeps asking questions till the user chooses to exit. Each turn we do:
        # Update state with user input
        # Decide next state
        # Generate system response
    while True:
        utterance = input("> ")
        cleaned_utterance = utterance.strip()

        if cleaned_utterance.lower() == "exit":
            break

        if not cleaned_utterance:
            print("please type something")
            continue

        dialog_act = predict_dialog_act(model_path, utterance)
        print("(TEMPORARY PRINT) dialog act:", dialog_act)

        extracted_slots = extract_slots(utterance, config)
        state.update_from_user_input(utterance, dialog_act, extracted_slots)

        # States 2-4 ask for food, area, then price when a value is still missing.
        requirements = state.requirements
        has_all_required_values = all(requirements.get(slot) is not None for slot in REQUIRED_SLOTS)

        # State 5 handles no matches. State 6 asks for additional requirements before
        # recommendation; the FSM does this even when lookup returns only one candidate.
        if dialog_act not in ("request", "reqalts", "bye", "thankyou"):
            if has_all_required_values:
                state.matches = find_restaurants(
                    {slot: requirements[slot] for slot in REQUIRED_SLOTS},
                    restaurant_info_df,
                    state.previous_recommendations,
                )
            else:
                state.matches = []

        # Transition to the next state based on the current state and dialog act
        state.transition(dialog_act)
        system_action = state.current_state

        # State 7. Suggest a restaurant.
        if system_action == "7_suggest_restaurant":
            matches = state.matches
            if matches:
                restaurant = matches[0]
                remaining_matches = matches[1:]
                reasoning = apply_reasoning(restaurant, state.additional_requirements)
                restaurant_name = restaurant.get("restaurantname", restaurant)

                state.matches = remaining_matches
                state.current_recommendation = restaurant
                state.previous_recommendations.append(restaurant_name)
                state.reasoning = reasoning

        # State 8. Give information about the current recommendation
        if system_action == "8_give_info":
            state.last_utterance = utterance

        response = generate_response(system_action, state, config)
        print(response)

        interaction_log.append(state.to_dict())

        # State 9. Goodbye
        if system_action == "9_goodbye":
            break

    datetime = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_interaction_log(interaction_log, config, f"part_1b/results/interaction_log_{datetime}.txt")

    print("\n\n----Finished the restaurant interaction pipeline for Part 1b----\n\n")



# TODO: Replay all 20 reference dialogs through the run_interaction_pipeline function, we manually compare the output to the reference_dialog.txt file.
def run_reference_dialog_tests_pipeline() -> None:
    """TODO: Evaluate the production dialog pipeline against all 20 reference dialogs."""
    print("\nStarting the reference dialog tests for Part 1b...\n")
