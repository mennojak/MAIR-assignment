def get_runtime_config(reasoning_transparency: bool, fallback: str = "levenshtein"):
    if fallback not in ("levenshtein", "embeddings"):
        raise ValueError("Choose 'levenshtein' or 'embeddings' for the fallback.")

    return {
        "reasoning_transparency": reasoning_transparency,
        "fallback": fallback,
        "max_edit_distance": 2,
        "max_edit_ratio": 0.25,
        "similarity_threshold": 0.85,
        "similarity_margin": 0.03,
        "model_path": "models/model_frozen_embeddings_grouped_SVM", # Best performing model from part 1a
        "random_seed": 12,
    }
