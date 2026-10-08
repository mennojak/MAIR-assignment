import pandas as pd

from part_1b.reasoning import apply_reasoning
from part_1b.state import REQUIRED_SLOTS


def find_restaurants(requirements: dict, additional_requirements: dict, recommended_restaurants: list[str]):
    restaurants_df = pd.read_csv("data/restaurant_info_extended.csv")
    matches = restaurants_df.copy()

    for slot in REQUIRED_SLOTS:
        value = requirements.get(slot)
        if value is None:
            continue

        value = str(value).strip().lower()
        if value in ("", "any", "dontcare", "don't care"):
            continue
        if slot == "area" and value == "center":
            value = "centre"

        column = matches[slot].fillna("").astype(str).str.strip().str.lower()
        matches = matches[column == value]

    recommended_names = set()
    for name in recommended_restaurants:
        recommended_names.add(str(name).strip().lower())

    restaurant_names = (
        matches["restaurantname"].fillna("").astype(str).str.strip().str.lower()
    )
    matches = matches[~restaurant_names.isin(recommended_names)]

    restaurants = matches.fillna("unknown").to_dict("records")
    matching_restaurants = []

    for restaurant in restaurants:
        reasoning = apply_reasoning(restaurant, additional_requirements)
        if reasoning["matches_requirements"]:
            restaurant["reasoning"] = reasoning
            matching_restaurants.append(restaurant)

    return matching_restaurants