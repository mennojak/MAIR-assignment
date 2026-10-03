# TODO: Return the options consumed by slot_extraction.extract_slots and logged per session.
# TODO: Configure keyword-first extraction, the Levenshtein/embedding fallback choice,
# TODO: and any matching thresholds.
# TODO: Include reasoning_transparency as a runtime boolean for the Part 2 A/B variants; 
#       Show vs. hide the inference chain behind a recommendation (e.g., "I recommend Zizzi because it allows a long stay" vs. just "I recommend Zizzi").
# TODO: both variants must use the same state machine, classifier, lookup and reasoning.
# TODO: Move dialog-act model selection here if it needs to be configurable at runtime.
# TODO: Keep runtime options separate from dialogue state and FSM logic.

def get_runtime_config() -> dict:
    """Return options passed to slot extraction and recorded in interaction logs."""
    pass
