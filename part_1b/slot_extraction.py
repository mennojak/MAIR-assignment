# Return {"slots": {...}, "confirmation": None} when no fallback needs confirmation.
# Otherwise include {"slot": ..., "original": ..., "suggestion": ...} in "confirmation".
# The state update method stores the suggestion and waits for an affirm/negate dialog act.
# TODO: Support food, area, pricerange, touristic, assigned_seats, children and romantic.
# TODO: Return None for an additional requirement the user doesn't care about.
# TODO: Use booleans for requested additional requirements and use the configured
#       fallback strategy where needed.

def extract_slots(utterance: str, config: dict) -> dict:
    """Return exact slots and an optional value that needs user confirmation."""
    pass
