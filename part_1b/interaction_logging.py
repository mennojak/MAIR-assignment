from pathlib import Path

def save_interaction_log(turns: list[dict], config: dict, filename: str):
    output_path = Path(f"part_1b/logs/{filename}")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Configuration\n")
        for name, value in config.items():
            file.write(f"{name}: {value}\n")

        current_dialog = None
        for turn_idx, turn in enumerate(turns, start=1):
            dialog_number = turn.get("dialog_number")
            if dialog_number is not None and dialog_number != current_dialog:
                current_dialog = dialog_number
                file.write(f"\n\n========== Dialog {dialog_number} ==========\n")

            file.write("\n----------------------------------------\n")
            turn_number = turn.get("turn_number", turn_idx)
            file.write(f"Turn {turn_number}\n")
            file.write(f"User utterance: {turn['user_utterance']}\n")
            if "dialog_act" in turn:
                file.write(f"Dialog act: {turn['dialog_act']}\n")
            file.write("State:\n")
            for name, value in turn["state"].items():
                file.write(f"  {name}: {value}\n")
            file.write(f"Response: {turn['response']}\n")


def save_reference_dialog_log(dialogs, filename: str, fallback: str):
    output_path = Path(f"part_1b/tests/{filename}")

    with output_path.open("w", encoding="utf-8") as file:
        for dialog_number, (reference, tested_turns) in enumerate(dialogs, start=1):
            file.write(f"--- Dialog {dialog_number}: Reference ---\n\n")
            file.write("\n".join(reference))
            file.write(f"\n\n{'-' * 80}\n")
            file.write(f"--- Dialog {dialog_number}: Test ({fallback}) ---\n\n")
            file.write("S: Hello, welcome to the restaurant recommendation system. How may I help you?\n")

            for utterance, response in tested_turns:
                file.write(f"U: {utterance}\n")
                file.write(f"S: {response}\n")

            file.write(f"\n\n\n\n\n{'=' * 80}\n\n\n\n\n")
