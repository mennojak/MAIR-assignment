from sklearn.metrics import recall_score

def evaluate_model_results(df, filename, difficult_cases=False):
    true_labels = df['dialog_act']
    model_prediction_columns = [column for column in df.columns if column.startswith('pred_')]

    results = []
    act_accurracies = ['\nActs by accuracy:']
    misclassified_for_all = []

    for prediction_column in model_prediction_columns:
        predictions = df[prediction_column]

        accuracy = (true_labels == predictions).mean()
        # The nicer balanced_accuracy_score from sklearn library gives a user warning, so used recall_score instead with macro to get the same result
        balanced_accuracy = recall_score(
            true_labels,
            predictions,
            labels=true_labels.unique(),
            average='macro',
        )

        model_name = prediction_column.replace('pred_', '')

        results.append(
            f'{model_name}: accuracy={accuracy:.4f}, '
            f'balanced_accuracy={balanced_accuracy:.4f}'
        )

        correct_predictions = true_labels == predictions
        act_accuracy = correct_predictions.groupby(true_labels).mean().sort_values()
        acts_ordered_by_accuracy = ', '.join(f'{act} ({accuracy:.4f})' for act, accuracy in act_accuracy.items())
        act_accurracies.append(f'{model_name}: {acts_ordered_by_accuracy}')

    predictions = df[model_prediction_columns]
    wrong_in_every_model = predictions.ne(true_labels, axis=0).all(axis=1)
    shared_errors = df.loc[wrong_in_every_model, ['utterance', 'dialog_act'] + model_prediction_columns]    

    # For difficult cases we want to see every utterance, normally only the shared errors.
    cases_to_report = df if difficult_cases else shared_errors
    text_heading = ("Difficult-case predictions" if difficult_cases else "Utterances misclassified by all systems")

    misclassified_for_all.append(f'\n{text_heading} ({len(cases_to_report)}):')
    for _, row in cases_to_report.iterrows():
        predictions_text = ', '.join(
            f'{column.replace("pred_", "")}={row[column]}'
            for column in model_prediction_columns
        )
        misclassified_for_all.append(
            f'- "{row["utterance"]}" '
            f'(true={row["dialog_act"]}; {predictions_text})\n'
        )

    results_print = '\n'.join(results)
    print(f"------------------------\nResults for {filename}:\n{results_print}")

    results_text = '\n'.join(results + act_accurracies + misclassified_for_all)

    results_path = f'part_1a/results/{filename}_evaluation.txt'

    with open(results_path, 'w') as file:
        file.write(results_text + '\n')