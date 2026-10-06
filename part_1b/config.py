# TODO: Return the options consumed by slot_extraction.extract_slots and logged per session.
# TODO: Configure keyword-first extraction, the Levenshtein/embedding fallback choice,
# TODO: and any matching thresholds.
# TODO: Include reasoning_transparency as a runtime boolean for the Part 2 A/B variants; 
#       Show vs. hide the inference chain behind a recommendation (e.g., "I recommend Zizzi because it allows a long stay" vs. just "I recommend Zizzi").
# TODO: both variants must use the same state machine, classifier, lookup and reasoning.
# TODO: Move dialog-act model selection here if it needs to be configurable at runtime.
# TODO: Keep runtime options separate from dialogue state and FSM logic.

def get_runtime_config(fallback="levenshtein", reasoning_transparency=True):
    # Defaults used by the terminal and tests.
    if fallback not in ("levenshtein", "embeddings"):
        raise ValueError("Choose levenshtein or embeddings for the fallback.")
    if not isinstance(reasoning_transparency, bool):
        raise ValueError("reasoning_transparency must be True or False.")

    return {
        "fallback": fallback,
        "max_edit_distance": 2,
        "max_edit_ratio": 0.25,
        "similarity_threshold": 0.85,
        "similarity_margin": 0.03,
        "reasoning_transparency": reasoning_transparency,
        "model_path": "models/model_frozen_embeddings_grouped_SVM",
        "random_seed": 12,
    }
