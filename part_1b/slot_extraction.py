# The pipeline passes the utterance and runtime configuration, then merges the returned
# mapping into the fixed requirements and additional_requirements state fields.
# TODO: Return only the supported keys: food, area, pricerange, touristic,
# TODO: assigned_seats, children and romantic; support multiple values per utterance.
# TODO: Handle don't-care phrasing and use the configured fallback strategy where needed.
# TODO: Add confirmation handling only alongside an explicit confirmation state/flow.

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
    """Clean the text before looking for words."""
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def load_ontology():
    """Read the possible foods, areas and prices."""
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
    """Check for words such as "not" before this value."""
    before = re.split(r"\b(?:but|and|instead)\b", text[:start])[-1].split()
    for word in before[-4:]:
        if word in ("not", "no", "dont", "avoid", "without"):
            return True
    return False


@lru_cache(maxsize=4)
def ontology_embeddings(slot, terms):
    """Get the vectors using our Part 1a code."""
    from part_1a.frozen_embedding_models import make_embeddings
    return make_embeddings(list(terms))


def match_fallback(given, slot, config):
    """Try a close match; leave it unknown if unsure."""
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
    lengths = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    scores = (vectors @ query) / np.maximum(lengths, 1e-12)
    order = np.argsort(scores)[::-1]
    first, second = int(order[0]), int(order[1])
    gap = scores[first] - scores[second]
    if scores[first] >= config["similarity_threshold"] and gap >= config["similarity_margin"]:
        return terms[first]
    return None


def extract_slot_details(utterance, config, state=None):
    """Find preferences and keep guesses separately."""
    text = clean_utterance(utterance)
    state = state or {}
    asked = ASKED_SLOTS.get(state.get("current_state"))
    ontology = load_ontology()
    slots = {}
    pending = []
    unrecognized = []
    excluded = {}
    used = []

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
            for start, end in used:
                if match.start() < end and match.end() > start:
                    overlap = True
                    break
            if overlap:
                continue
            used.append(match.span())
            if is_negative(text, match.start()):
                if slot not in excluded:
                    excluded[slot] = []
                excluded[slot].append(value)
            else:
                # For 'east, actually west', keep west.
                old = slots.get(slot)
                if old is None or text.rfind(value) > text.rfind(old):
                    slots[slot] = value

    any_patterns = {
        "food": r"\bany (?:kind of |type of )?(?:food|cuisine)\b",
        "area": r"\bany (?:part of town|area|location)\b",
        "pricerange": r"\bany price(?: range)?\b",
    }
    for slot, pattern in any_patterns.items():
        if re.search(pattern, text):
            slots[slot] = "dontcare"
    for slot, words in {"food": "food|cuisine", "area": "area|location|part of town",
                        "pricerange": "price|price range"}.items():
        if re.search(r"\b(?:dont care|no preference)(?: about| for| for the| about the)? (?:" + words + r")\b", text):
            slots[slot] = "dontcare"
    if asked and re.fullmatch(r"(?:i )?(?:dont care|do not care|have no preference|no preference|any|anything|whatever|it doesnt matter|doesnt matter)(?: is fine)?", text):
        slots[asked] = "dontcare"

    for slot, phrases in ADDITIONAL_WORDS.items():
        for phrase in phrases:
            match = re.search(r"\b" + re.escape(phrase) + r"\b", text)
            if match:
                slots[slot] = not is_negative(text, match.start())
                used.append(match.span())
                break

    # Try the words we haven't matched yet.
    words = []
    for match in re.finditer(r"\b[a-z]+\b", text):
        if match.group() in STOP_WORDS:
            continue
        already_used = False
        for start, end in used:
            if start <= match.start() < end:
                already_used = True
                break
        if not already_used and not is_negative(text, match.start()):
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
        relevant = relevant or (slot == "food" and food_mentioned)
        relevant = relevant or (slot == "area" and area_mentioned)
        relevant = relevant or (slot == "pricerange" and price_mentioned)
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
    """Return the values that do not need confirmation."""
    return extract_slot_details(utterance, config, state)["slots"]
