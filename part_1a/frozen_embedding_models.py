# TODO: Load the pretrained DistilBERT tokenizer and model

# TODO: Freeze DistilBERT so its weights are not changed during training
# TODO: Put DistilBERT in evaluation mode

# TODO: Convert every utterance into a fixed-size DistilBERT representation
# TODO: Decide how the DistilBERT output will be pooled into one vector per utterance

# TODO: Create the same two classifier types used for the BoW models
# TODO: Train both classifiers using the frozen DistilBERT representations

# TODO: Make a function that converts new utterances into DistilBERT representations and predicts their dialog acts

# TODO: Save the trained classifiers and the information needed to recreate the DistilBERT representations
# TODO: Load the saved models for later evaluation