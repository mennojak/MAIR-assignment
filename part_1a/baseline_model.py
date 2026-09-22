import re

BASELINE_RULES = [
    ("affirm", re.compile(r"(?:yes|correct|right|yea)")),
    ("confirm", re.compile(r"(?:is\s+it)")),
    ("deny", re.compile(r"(?:dont\s+want)")),
    ("hello", re.compile(r"\b(?:hi|hello)")),
    # "inform" is what we set as default, so we don't need a rule for it
    ("negate", re.compile(r"^no\s+")),
    ("null", re.compile(r"(?:sil|noise|unintelligible|cough|uh)")),
    ("repeat", re.compile(r"(?:repeat|again)")),
    ("reqalts", re.compile(r"(?:how\s+about|what\s+about|anything\s+else)")),
    ("reqmore", re.compile(r"more")),
    ("request", re.compile(r"(?:what|address|phone\s+number|post\s+code|price\s+range|area)")),
    ("restart", re.compile(r"(?:start|reset)")),
    ("thankyou", re.compile(r"thank\s*you")),
    ("ack", re.compile(r"\b(?:okay|kay|ok)")), # Put last (instead of alpahetical) since "okay" is often what other dialog acts start with, now it doesn't interfere
    ("bye", re.compile(r"(?:good\s*bye|goodbye)")),   # bye is often in the "thankyou" act, so the bye rule needs to be below it
]

def run_inference_baseline_model(utterances):
    predictions = []

    for utterance in utterances:
        # If we cannot determine another act through the rules we assume the user wants to be informed, 
        # since it's the main goal of the recommendation system and also the biggest class.
        prediction = "inform"

        for dialog_act, pattern in BASELINE_RULES:
            if pattern.search(utterance):
                prediction = dialog_act
                break

        predictions.append(prediction)

    return predictions