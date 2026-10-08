import re
import numpy as np
import pandas as pd
from part_1b.state import REQUIRED_SLOTS
from part_1a.frozen_embedding_models import make_embeddings
from Levenshtein import distance


ASKED_SLOTS = {
    "2_ask_food": "food",
    "3_ask_area": "area",
    "4_ask_price": "pricerange",
}


def clean_utterance(utterance):
    text = utterance.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Some aliases of values in the references that are different in the CSV so we normalize those.
    text = re.sub(r"\bcenter\b", "centre", text)
    text = re.sub(r"\bmoderately priced\b", "moderate", text)
    return text


def get_possible_restaurant_values():
    restaurants = pd.read_csv("data/restaurant_info_extended.csv")
    return {slot: restaurants[slot].dropna().str.strip().str.lower().unique().tolist() for slot in REQUIRED_SLOTS}


def find_slot_values(text, possible_values):
    slots = {}

    for slot, values in possible_values.items():
        for value in values:
            pattern = r"\b" + re.escape(value) + r"\b"
            if re.search(pattern, text):
                slots[slot] = value
                break

    return slots


def add_dontcare_values(text, slots, asked_slot):
    phrases = {
        "food": ("any food", "any cuisine"),
        "area": ("any area", "any location", "any part of town"),
        "pricerange": ("any price", "any price range"),
    }

    for slot, slot_phrases in phrases.items():
        if any(phrase in text for phrase in slot_phrases):
            slots[slot] = "dontcare"

    # When a user is asked a specific slot they can simply reply like this without the slot name present.
    if asked_slot and (
        text == "any"
        or "dont care" in text
        or "do not care" in text
        or "doesnt matter" in text
    ):
        slots[asked_slot] = "dontcare"


def add_additional_requirements(text, slots):
    phrases = {
        "touristic": ("touristic",),
        "assigned_seats": ("assigned seats", "assigned seating"),
        "children": ("suitable for children",),
        "romantic": ("romantic",),
    }

    for additional_requirement, possible_phrase_wordings in phrases.items():
        if any(wording in text for wording in possible_phrase_wordings):
            slots[additional_requirement] = True


def get_fallback_slot(text, slots, asked_slot):
    if asked_slot and asked_slot not in slots:
        return asked_slot

    # At the start of a dialogue no slot has been asked yet. 
    # This checks if there was a typo in the food value, which otherwise could be missed.
    if "food" not in slots and re.search(r"\b(food|cuisine|serves?)\b", text):
        return "food"

    return None


def get_fallback_candidate(text, slot):
    patterns = {
        "food": r"\b([a-z]+) food\b",
        "area": r"\bin (?:the )?([a-z]+)\b",
        "pricerange": r"\b([a-z]+) priced?\b",
    }

    match = re.search(patterns[slot], text)
    if match:
        return match.group(1)

    return text


def get_fallback_match(candidate, slot, config):
    terms = get_possible_restaurant_values()[slot]

    if config["fallback"] == "levenshtein":
        matches = sorted((distance(candidate, term), term) for term in terms)
        best_distance, best_value = matches[0]
        ratio = best_distance / max(len(candidate), len(best_value))

        if best_distance <= config["max_edit_distance"] and ratio <= config["max_edit_ratio"]:
            return best_value
        return None

    # Embedding based fallback
    vectors = make_embeddings([candidate, *terms])
    candidate_vector = vectors[0]
    term_vectors = vectors[1:]
    scores = np.dot(term_vectors, candidate_vector) / (
        np.linalg.norm(term_vectors, axis=1) * np.linalg.norm(candidate_vector)
    )
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    second_best = float(np.partition(scores, -2)[-2])

    if best_score >= config["similarity_threshold"] and best_score - second_best >= config["similarity_margin"]:
        return terms[best_index]

    return None


def extract_slots(utterance, config, current_state=None):
    text = clean_utterance(utterance)
    possible_values = get_possible_restaurant_values()
    slots = find_slot_values(text, possible_values)
    asked_slot = ASKED_SLOTS.get(current_state)

    add_dontcare_values(text, slots, asked_slot)
    add_additional_requirements(text, slots)

    fallback_slot = get_fallback_slot(text, slots, asked_slot)
    if fallback_slot is None:
        return {"slots": slots, "confirmation": None}

    candidate = get_fallback_candidate(text, fallback_slot)
    suggestion = get_fallback_match(candidate, fallback_slot, config)
    if suggestion is None:
        return {"slots": slots, "confirmation": None}

    return {
        "slots": slots,
        "confirmation": {
            "slot": fallback_slot,
            "original": candidate,
            "suggestion": suggestion,
        },
    }
