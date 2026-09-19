# TODO: Create the BoW representation using the training utterances
# TODO: Make sure the BoW representation is only fitted on the training set
# Save the embeddings in the models folder (for original or grouped), for easy use in the inference function

# TODO: Decide how unknown words in the test set are handled

# TODO: Create the classifiers that will be used with BoW
# TODO: Train the classifier (either for original or grouped training data split) and save them for later use.

# TODO: Make a function that takes a trained model and test utterances and predicts a dialog act for every utterance

# TODO: Save the trained models and BoW representation so they can be reused later in the folder part_1a/models
# TODO: Load the saved models and BoW representation for evaluation/inference

def train_bow_classifier(df_train, grouped=False):
    print(df_train.describe())
    print("TODO: implement this function")

def run_inference_bow_model(model_file_path, utterances):
    predictions = []
    return predictions