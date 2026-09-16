def run_training_pipeline() -> None:
    """
    Split the training dataset and train all required Part 1a models.
    """
    # TODO: Load dialog_acts.dat.
    # TODO: Create both stratified original and grouped train/test splits.
    # TODO: Train two BoW classifiers on both split variants.
    # TODO: Train the same two classifiers on frozen DistilBERT embeddings for both splits.
    # TODO: Save trained artifacts and metadata.
    print("TODO: run_training_pipeline")


def run_evaluation_pipeline() -> None:
    """
    Evaluate all required model/split combinations and save the results.
    """ 
    # TODO: Load the corresponding test sets.
    # TODO: Run inference for all 10 required standard combinations.
    # TODO: Calculate accuracy, balanced accuracy, and optional extra metrics.
    # TODO: Save the results in a table.
    print("TODO: run_evaluation_pipeline")


def run_heldout_pipeline() -> None:
    """
    Evaluate all models on the held-out test set.
    """
    # TODO: Load dialog_acts_test.dat, without splitting it!
    # TODO: Load selected trained model(s) and their preprocessing artifacts.
    # TODO: Run inference and calculate accuracy and balanced accuracy once labels are available.
    # TODO: Print/save results in the location documented by the README.
    print("TODO: run_heldout_pipeline")


def run_difficult_cases_pipeline() -> None:
    """
    Run the two manually constructed difficult-case test sets through all models.
    """
    # TODO: Load the two manually curated difficult-case datasets.
    # TODO: Run inference for all models.
    # TODO: Save predictions and aggregate results.
    print("TODO: run_difficult_cases_pipeline")


def run_interaction_pipeline() -> None:
    """
    Run the interactive utterance-classification loop on the best classifier model.
    """
    # TODO: Load the best classifier selected for Part 1b integration.
    # TODO: Start the prompt loop.
    # TODO: Lowercase and preprocess user input consistently with training.
    # TODO: Stop cleanly on explicit exit commands and/or the bye dialog act.
    print("TODO: run_interaction_pipeline")
