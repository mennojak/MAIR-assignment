import os
import pandas as pd


DIFFICULT_CASES = [
    # Ambiguous wording cases: 
    # If the user meant "how about" or "what about" in terms of "tell me about" (i.e. inform)
    # Inspiration is taken from cases like "what about turkish" as seen in part_1a/results/original_evaluation.txt
    # The model wrongly predicts the request act in those cases.
    ("inform", "how about some french food"),
    ("inform", "how about some vietnamese food"),
    ("inform", "what about turkish food"),

    # Negation cases:
    # Inspired by deny cases like "i dont want pizza" from the dataset 
    ("deny", "I do not want more french options"),
    ("deny", "I do not want korean"),
    ("deny", "I do not want scandanavian"),
]

def create_difficult_cases_datasets():
    difficult_cases = pd.DataFrame(DIFFICULT_CASES, columns=["dialog_act", "utterance"])

    print("\nDifficult cases:")
    for i, case in difficult_cases.iterrows():
        print(
            f'- {"Ambiguous -" if i > 4 else "Negation -"}'
            f'true={case["dialog_act"]}: "{case["utterance"]}"'
        )
    print("------------------------------\n")

    output_path = "data/processed/difficult_cases.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    difficult_cases.to_csv(output_path, index=False)