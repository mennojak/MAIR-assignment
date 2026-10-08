# The pipeline passes the utterance and runtime configuration, then merges the returned
# mapping into the fixed requirements and additional_requirements state fields.
# TODO: Return only the supported keys: food, area, pricerange, touristic,
# TODO: assigned_seats, children and romantic; support multiple values per utterance.
# TODO: Return None for an additional requirement the user doesn't care about.
# TODO: Use booleans for requested additional requirements and use the configured
#       fallback strategy where needed.
# TODO: Add confirmation handling only alongside an explicit confirmation state/flow.

def extract_slots(utterance: str, config: dict) -> dict:
    """Return recognized fixed-schema slot values from one user utterance."""
    pass
