# TODO: Match the per-turn record assembled in pipelines.py: utterance, dialog act,
# TODO: extracted slots, system action, dialogue state, response and configuration.
# TODO: Wire this helper into the pipeline, or remove it if entries stay inline.

def create_interaction_log_entry(interaction_data: dict) -> dict:
    """Normalize the per-turn fields currently assembled by the interaction pipeline."""
    pass


# TODO: Serialize the pipeline's list of per-turn records as JSON at the supplied path.
# TODO: Ensure nested dialogue state and configuration values are JSON-serializable.

def save_interaction_log(log_entries: list[dict], output_path: str) -> None:
    """Save the per-turn records assembled by the interaction pipeline."""
    pass
