# Rules from the project description:
# 1. cheap AND good food -> touristic=True
# 2. Romanian -> touristic=False
# 3. busy -> assigned_seats=True
# 4. long stay -> children=False
# 5. busy -> romantic=False
# 6. long stay -> romantic=True

def apply_reasoning(restaurant: dict, additional_requirements: dict) -> dict:
    """Infer additional properties and check whether the restaurant matches them."""
    properties = {"touristic": None, "assigned_seats": None, "children": None, "romantic": None}
    rules = []

    price = str(restaurant.get("pricerange", "")).strip().lower()
    food = str(restaurant.get("food", "")).strip().lower()
    quality = str(restaurant.get("food quality", "")).strip().lower()
    crowdedness = str(restaurant.get("crowdedness", "")).strip().lower()
    stay = str(restaurant.get("length of stay", "")).strip().lower()

    # Rule 1
    if price == "cheap" and quality == "good":
        properties["touristic"] = True
        reason = "It is touristic because it is cheap and has good food."
        rules.append({
            "id": 1,
            "property": "touristic",
            "value": True,
            "reason": reason,
        })

    # Rule 2
    if food == "romanian":
        properties["touristic"] = False
        reason = "Romanian food is considered not touristic."
        rules.append({
            "id": 2,
            "property": "touristic",
            "value": False,
            "reason": reason,
        })

    # Rule 3 and 5
    if crowdedness == "busy":
        properties["assigned_seats"] = True
        properties["romantic"] = False
        reason = "It has assigned seats because it is busy."
        rules.append({
            "id": 3,
            "property": "assigned_seats",
            "value": True,
            "reason": reason,
        })

        reason = "It is not romantic because it is busy."
        rules.append({
            "id": 5,
            "property": "romantic",
            "value": False,
            "reason": reason,
        })

    # Rule 4 and 6
    if stay == "long":
        properties["children"] = False
        reason = "It is not suitable for children because it allows a long stay."
        rules.append({
            "id": 4,
            "property": "children",
            "value": False,
            "reason": reason,
        })

        reason = "It is romantic because it allows a long stay."
        rules.append({
            "id": 6,
            "property": "romantic",
            "value": True,
            "reason": reason,
        })
        if crowdedness != "busy":
            properties["romantic"] = True

    # Rule 6 takes priority when both rules apply, matching the reference dialogs.
    if crowdedness == "busy" and stay == "long":
        properties["romantic"] = True
        rules.append({
            "id": 7,
            "property": "romantic",
            "value": True,
            "reason": (
                "It is also busy, which normally suggests it is not romantic, "
                "but we prioritize the long-stay rule."
            ),
        })

    unmet_requirements = []
    for property, user_request in additional_requirements.items():
        if user_request is None:
            continue

        if properties[property] is not user_request:
            unmet_requirements.append(property)

    return {
        "properties": properties,
        "rules": rules,
        "matches_requirements": not unmet_requirements,
        "unmet_requirements": unmet_requirements,
    }