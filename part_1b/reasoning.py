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

def apply_reasoning(restaurant, additional_requirements: dict) -> dict:
    """Return inferred properties, explanations and requirement-match information.

    The caller still needs to apply this to all candidates and select/filter them.
    """
    pass
