# TODO: connect spelling suggestions to the yes/no confirmation flow.

import csv
import re
from functools import lru_cache
from pathlib import Path


ASKED_SLOTS = {"2_ask_food": "food", "3_ask_area": "area", "4_ask_price": "pricerange"}
ALIASES = {"center": "centre", "moderately priced": "moderate",
           "moderately": "moderate", "moderate price": "moderate",
           "inexpensive": "cheap"}
ADDITIONAL_WORDS = {
    "touristic": ["touristic", "touristy", "popular with tourists"],
    "assigned_seats": ["assigned seats", "assigned seating"],
    "children": ["suitable for children", "child friendly", "kid friendly", "children", "kids"],
    "romantic": ["romantic"],
}
STOP_WORDS = set("i im am a an the want would like looking for restaurant restaurants food "
                "in of town part area location price range priced some please that serves serve "
                "serving it should be and with how about what can have find me to eat "
                "any fine is okay ok then yes no none nothing else dont care matter "
                "doesnt do not prefer looking something hello hi thanks thank you bye "
                "another one more actually but instead are they there".split())


def clean_utterance(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def load_ontology():
    path = Path(__file__).resolve().parents[1] / "data" / "restaurant_info.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ontology = {}
    for slot in ("food", "area", "pricerange"):
        values = set()
        for row in rows:
            value = row[slot].strip()
            if value:
                values.add(value.lower())
        ontology[slot] = sorted(values)
    return ontology


def is_negative(text, start):
    before = text[:start]
    parts = re.split(r"\b(?:but|and|instead)\b", before)
    before = parts[-1].split()
    for word in before[-4:]:
        if word in ("not", "no", "dont", "avoid", "without"):
            return True
    return False


@lru_cache(maxsize=4)
def ontology_embeddings(slot, terms):
    from part_1a.frozen_embedding_models import make_embeddings
    return make_embeddings(list(terms))


def match_fallback(given, slot, config):
    terms = load_ontology()[slot]
    if len(given) < 4:
        return None
    if config["fallback"] == "levenshtein":
        from Levenshtein import distance
        scores = []
        for term in terms:
            scores.append((distance(given, term), term))
        scores.sort()
        score, best = scores[0]
        if len(scores) > 1 and score == scores[1][0]:
            return None
        ratio = score / max(len(given), len(best))
        if score <= config["max_edit_distance"] and ratio <= config["max_edit_ratio"]:
            return best
        return None

    import numpy as np
    from part_1a.frozen_embedding_models import make_embeddings
    vectors = ontology_embeddings(slot, tuple(terms))
    query = make_embeddings([given])[0]
    query_length = np.linalg.norm(query)
    scores = []
    for vector in vectors:
        length = np.linalg.norm(vector) * query_length
        if length < 1e-12:
            length = 1e-12
        score = np.dot(vector, query) / length
        scores.append(float(score))

    best = max(scores)
    index = scores.index(best)
    ordered = sorted(scores, reverse=True)
    if best < config["similarity_threshold"]:
        return None
    if best - ordered[1] < config["similarity_margin"]:
        return None
    return terms[index]


def extract_slot_details(utterance, config, state=None):
    text = clean_utterance(utterance)
    state = state or {}
    asked = ASKED_SLOTS.get(state.get("current_state"))
    ontology = load_ontology()
    slots = {}
    pending = []
    unrecognized = []
    excluded = {}
    used = set()

    for alias, value in ALIASES.items():
        text = re.sub(r"\b" + re.escape(alias) + r"\b", value, text)

    # Check 'north american' before checking 'north'.
    keywords = []
    for slot, values in ontology.items():
        for value in values:
            keywords.append((value, slot))
    keywords.sort(key=lambda item: len(item[0]), reverse=True)
    for value, slot in keywords:
        for match in re.finditer(r"\b" + re.escape(value) + r"\b", text):
            overlap = False
            for position in range(match.start(), match.end()):
                if position in used:
                    overlap = True
                    break
            if overlap:
                continue
            used.update(range(match.start(), match.end()))
            if is_negative(text, match.start()):
                if slot not in excluded:
                    excluded[slot] = []
                excluded[slot].append(value)
            else:
                # For 'east, actually west', keep west.
                old = slots.get(slot)
                if old is None or text.rfind(value) > text.rfind(old):
                    slots[slot] = value

    any_words = {
        "food": ["any food", "any cuisine", "any kind of food", "any kind of cuisine",
                 "any type of food", "any type of cuisine"],
        "area": ["any part of town", "any area", "any location"],
        "pricerange": ["any price"],
    }
    padded = " " + text + " "
    for slot, phrases in any_words.items():
        for phrase in phrases:
            if " " + phrase + " " in padded:
                slots[slot] = "dontcare"
                break

    names = {"food": ["food", "cuisine"], "area": ["area", "location", "part of town"],
             "pricerange": ["price", "price range"]}
    for slot, phrases in names.items():
        for phrase in phrases:
            for prefix in ["dont care", "no preference"]:
                for middle in [" ", " about ", " for ", " for the ", " about the "]:
                    if " " + prefix + middle + phrase + " " in padded:
                        slots[slot] = "dontcare"

    answer = text
    if answer.startswith("i "):
        answer = answer[2:]
    if answer.endswith(" is fine"):
        answer = answer[:-8]
    if asked and answer in ["dont care", "do not care", "have no preference",
                            "no preference", "any", "anything", "whatever",
                            "it doesnt matter", "doesnt matter"]:
        slots[asked] = "dontcare"

    for slot, phrases in ADDITIONAL_WORDS.items():
        for phrase in phrases:
            match = re.search(r"\b" + re.escape(phrase) + r"\b", text)
            if match:
                slots[slot] = not is_negative(text, match.start())
                used.update(range(match.start(), match.end()))
                break

    # Try the words we haven't matched yet.
    words = []
    for match in re.finditer(r"\b[a-z]+\b", text):
        if match.group() in STOP_WORDS:
            continue
        if match.start() in used:
            continue
        if is_negative(text, match.start()):
            continue
        words.append(match.group())
    candidates = list(words)
    if 1 < len(words) <= 3:
        candidates.insert(0, " ".join(words))

    food_mentioned = bool(re.search(r"\b(?:food|cuisine|serve|serves|serving)\b", text))
    area_mentioned = bool(re.search(r"\b(?:in|area|location|town)\b", text))
    price_mentioned = bool(re.search(r"\b(?:price|priced|cost)\b", text))
    for slot in ("food", "area", "pricerange"):
        if slot in slots or slot in excluded:
            continue
        # Use the last question to help with short answers.
        relevant = asked == slot or (not asked and len(text.split()) <= 2)
        if slot == "food" and food_mentioned:
            relevant = True
        if slot == "area" and area_mentioned:
            relevant = True
        if slot == "pricerange" and price_mentioned:
            relevant = True
        if not relevant:
            continue
        found = False
        for candidate in candidates:
            best = match_fallback(candidate, slot, config)
            if best is not None:
                # Wait for confirmation before accepting a guess.
                pending.append({"slot": slot, "given": candidate, "value": best})
                found = True
                break
        if not found:
            if candidates and (asked == slot or (slot == "food" and food_mentioned)):
                unrecognized.append({"slot": slot, "given": " ".join(words)})

    return {"slots": slots, "pending": pending, "unrecognized": unrecognized, "excluded": excluded}


def extract_slots(utterance, config, state=None):
    return extract_slot_details(utterance, config, state)["slots"]
