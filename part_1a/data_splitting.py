import pandas as pd
import numpy as np

def load_data_and_create_splits(held_out = False):
    rows = []

    filepath = "data/dialog_acts_test.dat" if held_out else "data/dialog_acts.dat"

    with open(filepath) as file:
        for line in file:
            dialog_act, utterance = line.split(maxsplit=1)
            rows.append({
                'dialog_act': dialog_act.lower(),
                'utterance': utterance.lower(),
            })

    df = pd.DataFrame(rows)

    if held_out:
        df.to_csv(f"data/processed/held_out_test.csv", index=False)
        return
    
    dialog_acts = df['dialog_act'].unique()

    train_original, test_original = create_stratified_split(df, dialog_acts)
    train_original.to_csv(f'data/processed/train_original.csv', index=False)
    test_original.to_csv(f'data/processed/test_original.csv', index=False)

    train_grouped, test_grouped = create_stratified_split(df, dialog_acts, grouped=True)
    train_grouped.to_csv(f'data/processed/train_grouped.csv', index=False)
    test_grouped.to_csv(f'data/processed/test_grouped.csv', index=False)


def create_stratified_split(df, dialog_acts, grouped=False):
    random_state = np.random.RandomState(12)
    train_idx = []
    test_idx = []

    for act in dialog_acts:
        rows_with_specific_act = df[df['dialog_act'] == act].copy()

        # Split the unique utterances into train and test sets, assigning duplicates to the same split
        if grouped: 
            unique_utterances = rows_with_specific_act['utterance'].drop_duplicates().tolist()
            random_state.shuffle(unique_utterances)

            split_point = round(len(unique_utterances) * 0.15)   
            test_utterances = set(unique_utterances[:split_point]) 

            for idx, row in rows_with_specific_act.iterrows():
                if row['utterance'] in test_utterances:
                    test_idx.append(idx)
                else:   
                    train_idx.append(idx)
        else:
            act_row_idx = rows_with_specific_act.index.tolist()
            random_state.shuffle(act_row_idx)

            split_point = round(len(act_row_idx) * 0.15)
            test_idx.extend(act_row_idx[:split_point])
            train_idx.extend(act_row_idx[split_point:])

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    save_name = "grouped" if grouped else "original"

    train_df.to_csv(f'data/processed/train_{save_name}.csv', index=False)
    test_df.to_csv(f'data/processed/test_{save_name}.csv', index=False)

    return train_df, test_df