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

    df_original_train = pd.read_csv("data/processed/train_original.csv", keep_default_na=False)
    df_grouped_train = pd.read_csv("data/processed/train_grouped.csv", keep_default_na=False)

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

    df_original_test = pd.read_csv("data/processed/test_original.csv", keep_default_na=False)
    utterances = df_original_test["utterance"].tolist()

    df_original_test["pred_baseline"] = run_inference_baseline_model(utterances)
    df_original_test["pred_BoW_LR"] = run_inference_bow_model("models/model_BoW_original_LR", utterances)
    df_original_test["pred_BoW_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_original_test["pred_frozen_embeddings_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_LR", utterances)
    df_original_test["pred_frozen_embeddings_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_SVM", utterances)
    
    evaluate_model_results(df_original_test, "original")

    df_grouped_test = pd.read_csv("data/processed/test_grouped.csv", keep_default_na=False)
    utterances = df_grouped_test["utterance"].tolist()

    df_grouped_test["pred_baseline"] = run_inference_baseline_model(utterances)
    df_grouped_test["pred_BoW_LR"] = run_inference_bow_model("models/model_BoW_grouped_LR", utterances)
    df_grouped_test["pred_BoW_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_grouped_test["pred_frozen_embeddings_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_LR", utterances)
    df_grouped_test["pred_frozen_embeddings_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_SVM", utterances)
    
    evaluate_model_results(df_grouped_test, "grouped")

    print("\nFinished the evaluation pipeline for Part 1a")


def run_held_out_pipeline() -> None:
    """
    Evaluate all models on the held-out test set.
    """
    print("\nStarting the held out testing pipeline for Part 1a...\n")

    if not os.path.exists("data/dialog_acts_test.dat"):
        print("dialog_acts_test.dat file is not found under the data folder. Add this file to be able to run this pipeline")

    load_data_and_create_splits(held_out = True)

    df_held_out_test = pd.read_csv("data/processed/held_out_test.csv", keep_default_na=False)
    utterances = df_held_out_test["utterance"].tolist()

    df_held_out_test["pred_baseline"] = run_inference_baseline_model(utterances)
    df_held_out_test["pred_BoW_original_LR"] = run_inference_bow_model("models/model_BoW_original_LR", utterances)
    df_held_out_test["pred_BoW_grouped_LR"] = run_inference_bow_model("models/model_BoW_grouped_LR", utterances)
    df_held_out_test["pred_BoW_original_SVM"] = run_inference_bow_model("models/model_BoW_original_SVM", utterances)
    df_held_out_test["pred_BoW_grouped_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_held_out_test["pred_frozen_embeddings_original_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_LR", utterances)
    df_held_out_test["pred_frozen_embeddings_grouped_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_LR", utterances)
    df_held_out_test["pred_frozen_embeddings_original_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_SVM", utterances)
    df_held_out_test["pred_frozen_embeddings_grouped_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_SVM", utterances)

    evaluate_model_results(df_held_out_test, "held_out")

    print("Finished the held out testing pipeline for Part 1a")


def run_difficult_cases_pipeline() -> None:
    """
    Run the two manually constructed difficult-case test sets through all models.
    """
    print("\nStarting the difficult cases pipeline for Part 1a...\n")

    create_difficult_cases_datasets()

    df_difficult_cases = pd.read_csv("data/processed/difficult_cases.csv", keep_default_na=False)
    utterances = df_difficult_cases["utterance"].tolist()

    df_difficult_cases["pred_baseline"] = run_inference_baseline_model(utterances)
    df_difficult_cases["pred_BoW_original_LR"] = run_inference_bow_model("models/model_BoW_original_LR", utterances)
    df_difficult_cases["pred_BoW_grouped_LR"] = run_inference_bow_model("models/model_BoW_grouped_LR", utterances)
    df_difficult_cases["pred_BoW_original_SVM"] = run_inference_bow_model("models/model_BoW_original_SVM", utterances)
    df_difficult_cases["pred_BoW_grouped_SVM"] = run_inference_bow_model("models/model_BoW_grouped_SVM", utterances)
    df_difficult_cases["pred_frozen_embeddings_original_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_LR", utterances)
    df_difficult_cases["pred_frozen_embeddings_grouped_LR"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_LR", utterances)
    df_difficult_cases["pred_frozen_embeddings_original_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_original_SVM", utterances)
    df_difficult_cases["pred_frozen_embeddings_grouped_SVM"] = run_inference_frozen_embeddings_model("models/model_frozen_embeddings_grouped_SVM", utterances)

    evaluate_model_results(df_difficult_cases, "difficult_cases", difficult_cases=True)

    print("Finished the difficult cases pipeline for Part 1a")


def run_interaction_pipeline() -> None:
    """
    Run the interactive classification loop on the best classifier.
    """
    print("\nStarting the user interaction pipeline for Part 1a...\n")

    print(f"Choose the model you want to use (best model = 5):")
    print(f"1. Baseline")
    print(f"2. BoW Logistical Regression - original split")
    print(f"3. BoW Logistical Regression - grouped split")
    print(f"4. BoW SVM - original split")
    print(f"5. BoW SVM - grouped split")
    print(f"6. Frozen embeddings Logistical Regression - original split")
    print(f"7. Frozen embeddings Logistical Regression - grouped split")
    print(f"8. Frozen embeddings SVM - original split")
    print(f"9. Frozen embeddings SVM - grouped split")

    chosen_model = input("Choose a model (1-9): ")

    match chosen_model:
        case "1":
            chosen_model = "baseline"
        case "2":
            chosen_model = "models/model_BoW_original_LR"
        case "3":
            chosen_model = "models/model_BoW_grouped_LR"
        case "4":
            chosen_model = "models/model_BoW_original_SVM"
        case "5":
            chosen_model = "models/model_BoW_grouped_SVM"
        case "6":
            chosen_model = "models/model_frozen_embeddings_original_LR"
        case "7":
            chosen_model = "models/model_frozen_embeddings_grouped_LR"
        case "8":
            chosen_model = "models/model_frozen_embeddings_original_SVM"
        case "9":
            chosen_model = "models/model_frozen_embeddings_grouped_SVM"
        case _:
            print("Invalid choice")
            return

    interactive_classification_loop(chosen_model)

    print("Finished the user interaction pipeline for Part 1a")
