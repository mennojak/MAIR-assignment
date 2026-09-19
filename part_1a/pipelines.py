import os

import pandas as pd

from part_1a.data_splitting import load_data_and_create_splits
from part_1a.bow_models import train_bow_classifier, run_inference_bow_model
from part_1a.difficult_cases import create_difficult_cases_datasets
from part_1a.evaluation import evaluate_model_results
from part_1a.frozen_embedding_models import train_frozen_embeddings_classifier, run_inference_frozen_embeddings_model
from part_1a.baseline_model import run_inference_baseline_model
from part_1a.user_interaction import interactive_classification_loop

def run_training_pipeline() -> None:
    """
    Split the training dataset and train all required Part 1a models.
    """
    print("\nStarting the training pipeline for Part 1a...\n")

    load_data_and_create_splits(held_out = False)

    df_original_train = pd.read_csv("data/processed/train_original.csv")
    df_grouped_train = pd.read_csv("data/processed/train_grouped.csv")

    train_bow_classifier(df_original_train)
    train_bow_classifier(df_grouped_train, grouped=True)

    train_frozen_embeddings_classifier(df_original_train)
    train_frozen_embeddings_classifier(df_grouped_train, grouped=True)

    print("Finished the training pipeline for Part 1a")


def run_evaluation_pipeline() -> None:
    """
    Evaluate all required model/split combinations and save the results.
    """ 
    print("\nStarting the evaluation pipeline for Part 1a...\n")

    df_original_test = pd.read_csv("data/processed/test_original.csv")
    utterances = df_original_test["utterance"].tolist()

    # TODO: Change model names once we know the names of them, for now placeholders
    df_original_test["pred_baseline"] = run_inference_baseline_model(utterances)
    df_original_test["pred_BoW_LR"] = run_inference_bow_model("models/model_BoW_LR", utterances)
    df_original_test["pred_BoW_SVM"] = run_inference_bow_model("models/model_BoW_SVM", utterances)
    df_original_test["pred_frozen_embeddings_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_LR", utterances)
    df_original_test["pred_frozen_embeddings_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_SVM", utterances)

    evaluate_model_results(df_original_test, "original")

    df_grouped_test = pd.read_csv("data/processed/test_grouped.csv")
    utterances = df_grouped_test["utterance"].tolist()

    # TODO: Change model names once we know the names of them, for now placeholders
    df_grouped_test["pred_baseline"] = run_inference_baseline_model(utterances)
    df_grouped_test["pred_BoW_LR"] = run_inference_bow_model("models/model_BoW_LR", utterances)
    df_grouped_test["pred_BoW_SVM"] = run_inference_bow_model("models/model_BoW_SVM", utterances)
    df_grouped_test["pred_frozen_embeddings_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_LR", utterances)
    df_grouped_test["pred_frozen_embeddings_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_SVM", utterances)

    evaluate_model_results(df_grouped_test, "grouped")

    print("Finished the evaluation pipeline for Part 1a")


def run_held_out_pipeline() -> None:
    """
    Evaluate all models on the held-out test set.
    """
    print("\nStarting the held out testing pipeline for Part 1a...\n")

    if not os.path.exists("data/dialog_acts_test.dat"):
        print("dialog_acts_test.dat file is not found under the data folder. Add this file to be able to run this pipeline")

    load_data_and_create_splits(held_out = True)

    df_held_out_test = pd.read_csv("data/processed/dialog_acts_test.dat")
    utterances = df_held_out_test["utterance"].tolist()

    # TODO: Change model names once we know the names of them, for now placeholders
    df_held_out_test["pred_baseline"] = run_inference_baseline_model(utterances)
    df_held_out_test["pred_BoW_original_LR"] = run_inference_bow_model("models/model_BoW_original_LR", utterances)
    df_held_out_test["pred_BoW_grouped_LR"] = run_inference_bow_model("models/model_BoW_grouped_LR", utterances)
    df_held_out_test["pred_BoW_original_SVM"] = run_inference_bow_model("models/model_BoW_original_SVM", utterances)
    df_held_out_test["pred_BoW_grouped_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_held_out_test["pred_frozen_embeddings_original_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_LR", utterances)
    df_held_out_test["pred_frozen_embeddings_grouped_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_LR", utterances)
    df_held_out_test["pred_frozen_embeddings_original_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_SVM", utterances)
    df_held_out_test["pred_frozen_embeddings_grouped_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_SVM", utterances)

    evaluate_model_results(df_held_out_test, "held_out")

    print("Finished the held out testing pipeline for Part 1a")


def run_difficult_cases_pipeline() -> None:
    """
    Run the two manually constructed difficult-case test sets through all models.
    """
    print("\nStarting the difficult cases pipeline for Part 1a...\n")

    create_difficult_cases_datasets()

    df_difficult_cases_1 = pd.read_csv("data/processed/difficult_cases_1.csv")
    utterances = df_difficult_cases_1["utterance"].tolist()

    # TODO: Change model names once we know the names of them, for now placeholders
    df_difficult_cases_1["pred_baseline"] = run_inference_baseline_model(utterances)
    df_difficult_cases_1["pred_BoW_original_LR"] = run_inference_bow_model("models/model_BoW_original_LR", utterances)
    df_difficult_cases_1["pred_BoW_grouped_LR"] = run_inference_bow_model("models/model_BoW_grouped_LR", utterances)
    df_difficult_cases_1["pred_BoW_original_SVM"] = run_inference_bow_model("models/model_BoW_original_SVM", utterances)
    df_difficult_cases_1["pred_BoW_grouped_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_difficult_cases_1["pred_frozen_embeddings_original_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_LR", utterances)
    df_difficult_cases_1["pred_frozen_embeddings_grouped_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_LR", utterances)
    df_difficult_cases_1["pred_frozen_embeddings_original_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_SVM", utterances)
    df_difficult_cases_1["pred_frozen_embeddings_grouped_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_SVM", utterances)

    evaluate_model_results(df_difficult_cases_1, "difficult_cases_1")

    df_difficult_cases_2 = pd.read_csv("data/processed/difficult_cases_2.csv")
    utterances = df_difficult_cases_2["utterance"].tolist()

    # TODO: Change model names once we know the names of them, for now placeholders
    df_difficult_cases_2["pred_baseline"] = run_inference_baseline_model(utterances)
    df_difficult_cases_2["pred_BoW_original_LR"] = run_inference_bow_model("models/model_BoW_original_LR", utterances)
    df_difficult_cases_2["pred_BoW_grouped_LR"] = run_inference_bow_model("models/model_BoW_grouped_LR", utterances)
    df_difficult_cases_2["pred_BoW_original_SVM"] = run_inference_bow_model("models/model_BoW_original_SVM", utterances)
    df_difficult_cases_2["pred_BoW_grouped_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_difficult_cases_2["pred_frozen_embeddings_original_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_LR", utterances)
    df_difficult_cases_2["pred_frozen_embeddings_grouped_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_LR", utterances)
    df_difficult_cases_2["pred_frozen_embeddings_original_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_SVM", utterances)
    df_difficult_cases_2["pred_frozen_embeddings_grouped_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_SVM", utterances)

    evaluate_model_results(df_difficult_cases_2, "difficult_cases_2")

    print("Finished the difficult cases pipeline for Part 1a")


def run_interaction_pipeline() -> None:
    """
    Run the interactive classification loop on the best classifier.
    """
    print("\nStarting the user interaction pipeline for Part 1a...\n")

    # TODO: Change this to the best model once we know which one it is.
    best_model_path = "models/model_frozen_embeddings_grouped_LR"
    print(f"Using the best model: {best_model_path}")

    interactive_classification_loop(best_model_path)

    print("Finished the user interaction pipeline for Part 1a")
