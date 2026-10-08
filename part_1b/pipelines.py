import pandas as pd
from copy import deepcopy
from pathlib import Path

from part_1a.user_interaction import predict_dialog_act
from part_1b.state import REQUIRED_SLOTS, DialogueState
from part_1b.slot_extraction import extract_slots
from part_1b.restaurant_lookup import find_restaurants
from part_1b.response_generation import generate_response
from part_1b.interaction_logging import save_interaction_log, save_reference_dialog_log
from part_1b.config import get_runtime_config


def process_user_utterance(utterance: str, config: dict, state: DialogueState):
    """Process one user turn and return the response and data to log."""
    dialog_act = predict_dialog_act(config["model_path"], utterance)

    if state.pending_confirmation is not None or dialog_act not in ("inform", "reqalts"):
        extraction = None
    else:
        extraction = extract_slots(utterance, config, state.current_state)

    state.update_from_user_input(utterance, dialog_act, extraction)

    requirements = state.requirements
    has_all_required_values = all(
        requirements.get(slot) is not None for slot in REQUIRED_SLOTS
    )

    if (
        state.pending_confirmation is None
        and dialog_act not in ("restart", "request", "reqalts", "bye", "thankyou")
    ):
        if has_all_required_values:
            state.matches = find_restaurants(
                {slot: requirements[slot] for slot in REQUIRED_SLOTS},
                state.additional_requirements,
                state.previous_recommendations,
            )
        else:
            state.matches = []

    state.transition(dialog_act)
    system_action = state.current_state

    if system_action == "8_suggest_restaurant":
        if state.matches:
            restaurant = state.matches[0]
            state.matches = state.matches[1:]
            state.current_recommendation = restaurant
            state.previous_recommendations.append(restaurant["restaurantname"])
            state.reasoning = restaurant.get("reasoning")

    if system_action == "9_give_info":
        state.last_utterance = utterance

    response = generate_response(system_action, state, config)
    turn = {
        "user_utterance": utterance,
        "dialog_act": dialog_act,
        "state": deepcopy(state.__dict__),
        "response": response,
    }
    return dialog_act, response, turn


def run_interaction_pipeline(config: dict) -> None:
    """Run the complete Part 1b restaurant recommendation dialogue."""
    print("\nStarting the restaurant interaction pipeline for Part 1b...\n")

    # Run first prediction to load the model avoiding the first response load time
    predict_dialog_act(config["model_path"], "hello")

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

        dialog_act, response, turn = process_user_utterance(utterance, config, state)
        print("(TEMPORARY PRINT) dialog act:", dialog_act)

        print(response)

        interaction_log.append(turn)

        # State 10. Goodbye
        if state.current_state == "10_goodbye":
            break

    datetime = pd.Timestamp.now().strftime("%m-%d_%H-%M")
    save_interaction_log(interaction_log, config, f"interaction_log_{datetime}.txt")

    print("\n\n----Finished the restaurant interaction pipeline for Part 1b----\n\n")



def run_reference_dialog_tests_pipeline():
    print("\nStarting the reference dialog tests for Part 1b...\n")

    reference_path = Path("data/reference_dialogs.txt")
    dialogs = []
    current_dialog = {"reference": [], "utterances": []}
    inside_dialog = False

    # Keep the original transcript and its user utterances together per dialog.
    for line in reference_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("--- Dialog"):
            if inside_dialog:
                dialogs.append(current_dialog)
            current_dialog = {"reference": [], "utterances": []}
            inside_dialog = True
            current_dialog["reference"].append(line.strip())
        elif inside_dialog and line.strip():
            current_dialog["reference"].append(line.strip())
            if line.strip().startswith("U:"):
                current_dialog["utterances"].append(line.strip()[2:].strip())

    if inside_dialog:
        dialogs.append(current_dialog)

    # We run the tests for both fallback methods,
    # because it was not specified in the rubric/project description (only stating: "System handles all 20 reference dialogs correctly").
    # The reasoning transparency feature is enabled for these tests instead of concise responses,
    # because this allows us to more easily check if the system is working correctly.
    for fallback in ("levenshtein", "embeddings"):
        config = get_runtime_config(reasoning_transparency=True, fallback=fallback)
        print(f"Running reference dialogs with {fallback} fallback...")

        tested_dialogs = []

        for dialog in dialogs:
            state = DialogueState()
            tested_turns = []

            for utterance in dialog["utterances"]:
                _, response, _ = process_user_utterance(utterance, config, state)
                tested_turns.append((utterance, response))

                if state.current_state == "10_goodbye":
                    break

            tested_dialogs.append((dialog["reference"], tested_turns))

        filename = f"reference_dialogs_{fallback}.txt"
        save_reference_dialog_log(tested_dialogs, filename, fallback)
        print(f"Saved reference replay to part_1b/logs/{filename}")
