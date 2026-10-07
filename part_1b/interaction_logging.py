from pathlib import Path

def save_interaction_log(turns: list[dict], config: dict, filename: str):
    file_path = f"part_1b/logs/{filename}"
    output_path = Path(file_path)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Configuration\n")
        for name, value in config.items():
            file.write(f"{name}: {value}\n")

        for turn_idx, turn in enumerate(turns, start=1):
            file.write("\n----------------------------------------\n")
            file.write(f"Turn {turn_idx}\n")
            file.write(f"User utterance: {turn['user_utterance']}\n")
            file.write("State:\n")
            for name, value in turn["state"].items():
                file.write(f"  {name}: {value}\n")
            file.write(f"Response: {turn['response']}\n")
