def find_restaurants(requirements, restaurant_data, recommended_restaurants):
    # Find matches, then remove the restaurants already suggested.
    matches = restaurant_data.copy()
    for slot in ("food", "area", "pricerange"):
        value = requirements.get(slot)
        if value is None:
            continue
        value = str(value).strip().lower()
        if value in ("", "any", "dontcare"):
            continue
        if slot == "area" and value == "center":
            value = "centre"
        column = matches[slot].fillna("").astype(str).str.strip().str.lower()
        matches = matches[column == value]

    names = set()
    for name in recommended_restaurants:
        names.add(name.strip().lower())
    column = matches["restaurantname"].fillna("").astype(str).str.strip().str.lower()
    matches = matches[~column.isin(names)]
    return matches.fillna("").to_dict("records")
