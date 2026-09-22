from sklearn.metrics import recall_score

def evaluate_model_results(df, filename):
    true_labels = df['dialog_act']
    model_prediction_columns = [column for column in df.columns if column.startswith('pred_')]

    results = []

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

    results_text = '\n'.join(results)
    print(f"------------------------\nResults for {filename}:\n{results_text}")

    results_path = f'part_1a/results/{filename}_evaluation.txt'

    with open(results_path, 'w') as file:
        file.write(results_text + '\n')