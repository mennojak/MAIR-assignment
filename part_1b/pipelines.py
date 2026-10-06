import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from part_1a.user_interaction import predict_dialog_act
from part_1b.state import REQUIRED_SLOTS, create_initial_state, update_state_fields, transition_state
from part_1b.slot_extraction import extract_slot_details
from part_1b.restaurant_lookup import find_restaurants
from part_1b.reasoning import apply_reasoning
from part_1b.response_generation import generate_response
from part_1b.config import get_runtime_config
from part_1b.interaction_logging import create_interaction_log_entry, save_interaction_log


ROOT = Path(__file__).resolve().parents[1]


def process_turn(state, utterance, config, restaurant_data, rng):
    """Handle one turn. The tests use this too."""
    dialog_act = predict_dialog_act(config["model_path"], utterance)
    details = extract_slot_details(utterance, config, state)
    # A question about a restaurant isn't a new preference.
    slots = details["slots"]
    if dialog_act in ("request", "confirm", "bye", "thankyou", "repeat", "restart"):
        slots = {}
    updates = {"last_utterance": utterance, "dialog_act": dialog_act, "slots": slots}
    state = update_state_fields(state, updates)
    state["configuration"] = dict(config)
    state["slot_details"] = details
    requirements = state["requirements"]

    # Check the extra requirements before picking a restaurant.
    candidate_reasoning = {}
    rejected = []
    ready = True
    for slot in REQUIRED_SLOTS:
        if requirements.get(slot) is None:
            ready = False
    if ready:
        candidates = find_restaurants(requirements, restaurant_data,
                                      state.get("previous_recommendations", []))
        matches = []
        for restaurant in candidates:
            reasoning = apply_reasoning(restaurant, state["additional_requirements"])
            name = restaurant["restaurantname"]
            candidate_reasoning[name] = reasoning
            if reasoning["matches_requirements"]:
                matches.append(restaurant)
            else:
                rejected.append({"restaurantname": name, "reasoning": reasoning})
        state["matches"] = matches
    else:
        state["matches"] = []
    state["rejected_candidates"] = rejected

    state = transition_state(state, dialog_act)
    action = state["current_state"]
    if action == "7_suggest_restaurant" and state["matches"]:
        restaurant = rng.choice(state["matches"])
        state["matches"].remove(restaurant)
        name = restaurant["restaurantname"]
        state["current_recommendation"] = restaurant
        state["previous_recommendations"].append(name)
        state["reasoning"] = candidate_reasoning[name]

    response = generate_response(action, state)
    state["last_response"] = response
    entry = create_interaction_log_entry({
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "user_utterance": utterance, "dialog_act": dialog_act,
        "extracted_slots": slots, "slot_details": details,
        "system_action": action, "dialogue_state": state,
        "response": response, "configuration": config,
    })
    return state, response, entry


def run_interaction_pipeline(config=None):
    """Run the conversation and save what happened."""
    config = config or get_runtime_config()
    data = pd.read_csv(ROOT / "data/restaurant_info_extended.csv", keep_default_na=False)
    rng = random.Random(config["random_seed"])
    predict_dialog_act(config["model_path"], "hello")
    state = create_initial_state()
    log = []
    print(generate_response("1_welcome", state))
    print("Type exit to stop.")
    try:
        while True:
            utterance = input("> ").strip()
            if utterance.lower() == "exit":
                break
            if not utterance:
                print("Please type something.")
                continue
            state, response, entry = process_turn(state, utterance, config, data, rng)
            log.append(entry)
            print(response)
            if state["current_state"] == "9_goodbye":
                break
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        path = ROOT / "part_1b/results" / ("interaction_log_" + stamp + ".json")
        save_interaction_log(log, path)
        print("Log saved to", path)


def read_reference_dialogs(path):
    """Read the user lines from each reference dialog."""
    dialogs = {}
    number = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"--- Dialog (\d+) ---", line)
        if match:
            number = int(match.group(1))
            dialogs[number] = []
        elif line.startswith("U:") and number is not None:
            dialogs[number].append(line[2:].strip())
    return dialogs


def check_reference_turn(state, response, expected):
    """Check what the system did and said."""
    errors = []
    action = state["current_state"]
    if expected["action"] is not None and action != expected["action"]:
        errors.append("Expected " + expected["action"] + ", got " + action)
    if not response.strip():
        errors.append("The system response is empty.")
    if expected.get("pending"):
        pending = state.get("slot_details", {}).get("pending", [])
        if not any(item["value"] == expected["pending"] for item in pending):
            errors.append("The spelling suggestion is missing.")
    for slot, value in expected.get("slots", {}).items():
        actual = state["requirements"].get(slot, state["additional_requirements"].get(slot))
        if actual != value:
            errors.append(f"Expected {slot}={value}, got {actual}")

    restaurant = state.get("current_recommendation")
    if action == "7_suggest_restaurant":
        if not restaurant:
            errors.append("There is no selected restaurant.")
        else:
            for slot, value in state["requirements"].items():
                if value not in (None, "dontcare") and restaurant.get(slot) != value:
                    errors.append("The restaurant does not match " + slot)
            if restaurant["restaurantname"].lower() not in response.lower():
                errors.append("The response does not name the selected restaurant.")
            reasoning = state.get("reasoning", {})
            if not reasoning.get("matches_requirements"):
                errors.append("The restaurant does not meet the additional requirements.")
            if state.get("configuration", {}).get("reasoning_transparency"):
                for slot, wanted in state["additional_requirements"].items():
                    if isinstance(wanted, bool):
                        reasons = []
                        for rule in reasoning.get("rules", []):
                            if rule["property"] == slot:
                                reasons.append(rule["reason"])
                        if slot == "children":
                            reasons += reasoning.get("defaults", [])
                        if reasons and not any(word in response.lower() for word in ("because", "since", "assume", "rule")):
                            errors.append("The response is missing the explanation for " + slot)
    field = expected.get("info")
    if field:
        value = str((restaurant or {}).get(field, ""))
        if not restaurant:
            errors.append("Information was requested without a current recommendation.")
        elif value and value.lower() not in response.lower():
            errors.append("The response does not contain the restaurant's " + field)
        elif not value and "unknown" not in response.lower():
            errors.append("Missing " + field + " should be reported as unknown.")
    if "response_contains" in expected:
        if expected["response_contains"].lower() not in response.lower():
            errors.append("Missing response content: " + expected["response_contains"])
    return errors


def run_reference_dialog_tests_pipeline(config=None, output_dir=None):
    """Run the 20 dialogs and save the failed checks."""
    config = config or get_runtime_config()
    output_dir = Path(output_dir) if output_dir else ROOT / "part_1b/results"
    dialogs = read_reference_dialogs(ROOT / "data/reference_dialogs.txt")
    checks = json.loads((ROOT / "part_1b/reference_checks.json").read_text(encoding="utf-8"))
    if set(dialogs) != set(range(1, 21)):
        raise ValueError("Expected reference dialogs 1 to 20.")
    data = pd.read_csv(ROOT / "data/restaurant_info_extended.csv", keep_default_na=False)
    rng = random.Random(config["random_seed"])
    report = []
    transcript = []
    for number, utterances in dialogs.items():
        expectations = checks[str(number)]
        if len(utterances) != len(expectations):
            raise ValueError("The reference checks do not match dialog " + str(number))
        state = create_initial_state()
        failures = []
        for index, (utterance, expected) in enumerate(zip(utterances, expectations), start=1):
            if utterance != expected["utterance"]:
                raise ValueError("Reference text changed; review the checks for dialog " + str(number))
            try:
                state, response, entry = process_turn(state, utterance, config, data, rng)
                entry.update({"dialog": number, "turn": index, "added_turn": False})
                transcript.append(entry)
                # Some examples skip the extra question. Add a 'no' and log it.
                if expected.get("allow_extra_question") and state["current_state"] == "6_ask_additional_requirements":
                    state, response, entry = process_turn(state, "no", config, data, rng)
                    entry.update({"dialog": number, "turn": index, "added_turn": True})
                    transcript.append(entry)
                errors = check_reference_turn(state, response, expected)
            except Exception as error:
                errors = [type(error).__name__ + ": " + str(error)]
            if errors:
                failures.append({"turn": index, "utterance": utterance, "errors": errors})
        report.append({"dialog": number, "passed": not failures, "failures": failures})

    passed = 0
    for item in report:
        if item["passed"]:
            passed += 1
    lines = [f"Reference dialogs: {passed}/20 passed",
             "Uses the actual Part 1a classifier and the terminal turn function.",
             "Extra 'no' turns are recorded where examples omit the additional question.",
             "Dialog 17 expects no romantic match under our negative-wins policy.",
             "Dialog 18 accepts Curry King under our short-stay children default.", ""]
    for item in report:
        result = "PASS" if item["passed"] else "FAIL"
        lines.append(f"Dialog {item['dialog']}: {result}")
        for failure in item["failures"]:
            lines.append(f"  Turn {failure['turn']}: {failure['utterance']}")
            for error in failure["errors"]:
                lines.append("    " + error)
    mode = config["fallback"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"reference_summary_{mode}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_interaction_log(report, output_dir / f"reference_checks_{mode}.json")
    save_interaction_log(transcript, output_dir / f"reference_turns_{mode}.json")
    print(lines[0])
    print("Results saved to", output_dir)
    return report


def compare_slot_extractors():
    """Try both methods on a few hand-written examples."""
    examples = [
        ("cheap italian food in the centre", None, {"food": "italian", "area": "centre", "pricerange": "cheap"}),
        ("inexpensive food in the center", None, {"pricerange": "cheap", "area": "centre"}),
        ("spenish food", None, {"food": "spanish"}),
        ("italan food", None, {"food": "italian"}),
        ("nort", "3_ask_area", {"area": "north"}),
        ("cheep", "4_ask_price", {"pricerange": "cheap"}),
        ("any", "2_ask_food", {"food": "dontcare"}),
        ("any area any price", None, {"area": "dontcare", "pricerange": "dontcare"}),
        ("not romantic", None, {"romantic": False}),
        ("assigned seats", None, {"assigned_seats": True}),
        ("suitable for children", None, {"children": True}),
        ("Swedish food", None, {}),
        ("sdfhsdjkhjdhsdf", None, {}),
        ("fancy", "4_ask_price", {"pricerange": "expensive"}),
        ("costly", "4_ask_price", {"pricerange": "expensive"}),
        ("northern", "3_ask_area", {"area": "north"}),
        ("southern", "3_ask_area", {"area": "south"}),
    ]
    results = []
    lines = ["Slot comparison on 17 hand-written examples (not a held-out test).",
             "Recovered values include proposed corrections, which still require confirmation.", ""]
    for mode in ("levenshtein", "embeddings"):
        config = get_runtime_config(mode)
        correct = 0
        for utterance, question, expected in examples:
            details = extract_slot_details(utterance, config, {"current_state": question})
            recovered = dict(details["slots"])
            for suggestion in details["pending"]:
                recovered[suggestion["slot"]] = suggestion["value"]
            passed = recovered == expected
            if passed:
                correct += 1
            results.append({"mode": mode, "utterance": utterance, "question": question,
                            "expected": expected, "details": details, "passed": passed,
                            "configuration": config})
            if not passed:
                lines.append(f"{mode}: {utterance!r}: expected {expected}, found {recovered}")
        lines.append(f"{mode}: {correct}/{len(examples)} correct (including rejected unknown inputs)")
    output = ROOT / "part_1b/results"
    save_interaction_log(results, output / "slot_comparison.json")
    (output / "slot_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--test", action="store_true")
    modes.add_argument("--compare-slots", action="store_true")
    parser.add_argument("--fallback", choices=["levenshtein", "embeddings"], default="levenshtein")
    parser.add_argument("--hide-reasoning", action="store_true")
    args = parser.parse_args()
    options = get_runtime_config(args.fallback, not args.hide_reasoning)
    if args.compare_slots:
        compare_slot_extractors()
    elif args.test:
        report = run_reference_dialog_tests_pipeline(options)
        raise SystemExit(0 if all(item["passed"] for item in report) else 1)
    else:
        run_interaction_pipeline(options)
