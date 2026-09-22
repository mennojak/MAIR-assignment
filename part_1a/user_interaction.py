import os
from part_1a.baseline_model import run_inference_baseline_model
from part_1a.bow_models import clean_text, run_inference_bow_model
from part_1a.frozen_embedding_models import run_inference_frozen_embeddings_model


def predict_dialog_act(model_path, text):
    text = clean_text(text)
    if 'BoW' in model_path:
        result = run_inference_bow_model(model_path, [text])
    elif 'frozen_embeddings' in model_path:
        result = run_inference_frozen_embeddings_model(model_path, [text])
    else:
        result = run_inference_baseline_model([text])
    if len(result) == 0:
        return 'unknown'
    return result[0]


def interactive_classification_loop(model_path):
    if 'model_' in model_path and not os.path.exists('part_1a/' + model_path + '.joblib'):
        print('model', model_path, 'not found, run the training pipeline first (option 1)')
        return

    print("\nType a sentence to classify. To quit, type exit or stop or use ctrl+c or ctrl+d.")
    # the first prediction is slow because distilbert has to be loaded
    if 'frozen_embeddings' in model_path:
        print("\nloading the model...\n\n")
        predict_dialog_act(model_path, "hello")

    while True:
        try:
            text = input("Type your sentence: ")
        except (KeyboardInterrupt, EOFError):
            # ctrl+c or ctrl+d
            print()
            break
        if text.strip().lower() == 'exit' or text.strip().lower() == 'quit' or text.strip().lower() == 'stop':
            break
        if clean_text(text) == '':
            print("please type something")
            continue
        act = predict_dialog_act(model_path, text)
        print("dialog act:", act)
