# TODO: Implement all six Part 1b rules against restaurant_info_extended.csv:
# TODO: cheap AND good food -> touristic=True
# TODO: Romanian -> touristic=False
# TODO: busy -> assigned_seats=True
# TODO: long stay -> children=False
# TODO: busy -> romantic=False
# TODO: long stay -> romantic=True
# TODO: Return derived values, the rule IDs/natural-language explanations that fired,
# TODO: and any conflicts. The pipeline must apply this to every lookup candidate and
# TODO: filter candidates against requested additional_requirements before choosing one.
# TODO: Choose and document an explicit contradiction policy; never silently discard a
# TODO: conflicting conclusion. Response generation must honor reasoning_transparency.

def apply_reasoning(restaurant, additional_requirements):
    """Check the six rules. If they disagree, False wins."""
    properties = {"touristic": None, "assigned_seats": None,
                  "children": None, "romantic": None}
    rules = []
    price = str(restaurant.get("pricerange", "")).strip().lower()
    food = str(restaurant.get("food", "")).strip().lower()
    quality = str(restaurant.get("food quality", "")).strip().lower()
    busy = str(restaurant.get("crowdedness", "")).strip().lower() == "busy"
    stay = str(restaurant.get("length of stay", "")).strip().lower()

    if price == "cheap" and quality == "good":
        rules.append({"id": 1, "property": "touristic", "value": True,
                      "reason": "It is touristic because it is cheap and has good food."})
    if food == "romanian":
        rules.append({"id": 2, "property": "touristic", "value": False,
                      "reason": "Under rule 2, Romanian food is considered not touristic."})
    if busy:
        rules.append({"id": 3, "property": "assigned_seats", "value": True,
                      "reason": "It has assigned seats because it is busy."})
    if stay == "long":
        rules.append({"id": 4, "property": "children", "value": False,
                      "reason": "It is not suitable for children because it allows a long stay."})
    if busy:
        rules.append({"id": 5, "property": "romantic", "value": False,
                      "reason": "It is not romantic because it is busy."})
    if stay == "long":
        rules.append({"id": 6, "property": "romantic", "value": True,
                      "reason": "It is romantic because it allows a long stay."})

    conflicts = []
    explanations = []
    for name in properties:
        values = []
        rule_ids = []
        reasons = []
        for rule in rules:
            if rule["property"] == name:
                values.append(rule["value"])
                rule_ids.append(rule["id"])
                reasons.append(rule["reason"])

        if False in values:
            properties[name] = False
        elif True in values:
            properties[name] = True
        if True in values and False in values:
            conflicts.append({"property": name, "rule_ids": rule_ids,
                              "chosen_value": False})
            explanations.append("Rules disagree about " + name +
                                "; we give the negative conclusion priority.")
        explanations.extend(reasons)

    defaults = []
    # Our group's assumption. A missing stay value is still unknown.
    if stay == "short":
        properties["children"] = True
        defaults.append("We assume restaurants with a short stay are suitable for children.")

    unmet = []
    for name, wanted in additional_requirements.items():
        if name not in properties:
            raise ValueError("Unknown additional requirement: " + name)
        if wanted is None or wanted == "dontcare":
            continue
        if not isinstance(wanted, bool):
            raise ValueError("Additional requirements must be True, False or None.")
        if properties[name] is not wanted:
            unmet.append(name)

    return {"properties": properties, "rules": rules, "conflicts": conflicts,
            "explanations": explanations, "defaults": defaults,
            "matches_requirements": not unmet, "unmet_requirements": unmet,
            "conflict_policy": "negative_wins"}

