# INFOMAIR Restaurant Recommendation Dialog System

This repository contains Part 1a and Part 1b of the project.

- `data/` - received datasets
    - `processed/` - processed datasets for the experiments 
- `part_1a/` - dialog-act classification
    - `models/` - trained models and embeddings
    - `results/` - results from all experiments are saved here
    - `pipelines.py` - orchestrator for all Part 1a pipelines
    - all other files implement helper functions the orchestrator uses for the different pipelines.
- `part_1b/` - dialog-management extension built on top of Part 1a
    - `logs/` - interaction logs from users interacting with the dialog system
    - `tests/` - reference-dialog test results
    - `pipelines.py` - orchestrates part 1b pipelines (interactive dialogue and reference-dialog tests)
    - all other files implement helper functions the orchestrator uses for the different pipelines.
- `main.py` - Pipeline selector (and the file you run)

Part 1a is built first. Part 1b reuses the selected Part 1a classifier and adds dialog management, slot extraction, restaurant lookup, reasoning, configurable behavior and interaction logging.

## Installation guide
Run these commands from the repository folder:

```text
python -m venv .venv
```

Activate the environment and install the requirements:

```text
# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then start the system:

```text
python main.py
```

The first run that uses DistilBERT embeddings may download the pretrained model.

## Part 1a
This first part of the project asked us to create/train multiple text classifiers that take a user input and predict 1 of 15 possible dialog acts. It includes a baseline model with hardcoded rules and multiple versions of a Logistical Regression model and an SVM (Support Vector Machine) model. 

The two machine learning models will be trained on both BoW (bag-of-words) and Frozen DistilBERT embeddings. Additionally the given dataset is also split in two different ways, once randomly (called `original`) and the other focussed on handling duplicates (called `grouped`). Which gives a total of 9 different model/data split combinations (1 of which remains the baseline model) that have been evaluated to find the best classifier. 

Additionally, a held-out dataset for testing, `dialog_acts_test.dat`, is meant to be provided by the teachers for another round of evaluations. All evaluations are found under the `part_1a/results` folder and used for the report. Re-running all the different pipelines should yield the same results, because we implemented a set seed `12` (instead of a random one).

Running `main.py` provides the different options for pipelines (i.e. different parts/deliverables of the project) that can be run. For Part 1a these include:
1. Training pipeline
2. Evaluation pipeline
3. Held-out test set pipeline
    - Add the `dialog_acts_test.dat` file under the `data` folder before running. 
    - The results are saved in `part_1a/results/held_out_evaluation.txt`
4. Difficult cases pipeline
5. Interactive classification pipeline

To reproduce the Part 1a evaluation results, run the training pipeline first if the trained model files or processed data are not present, and then run the evaluation pipeline. The results are written to `part_1a/results/original_evaluation.txt` and `part_1a/results/grouped_evaluation.txt`. The held-out test pipeline requires the teacher-provided `data/dialog_acts_test.dat` file.

## Results part 1a
The accuracy and balanced accuracy for the original and grouped test splits are shown below.

| Model | Original accuracy | Original balanced accuracy | Grouped accuracy | Grouped balanced accuracy |
|---|---:|---:|---:|---:|
| Baseline | 0.9205 | 0.7082 | 0.8727 | 0.6920 |
| BoW Logistic Regression | 0.9753 | 0.7030 | 0.8977 | 0.5994 |
| BoW SVM | 0.9814 | 0.8796 | 0.8952 | 0.6974 |
| Frozen embeddings Logistic Regression | 0.9836 | 0.8785 | 0.9105 | 0.7416 |
| Frozen embeddings SVM | 0.9858 | 0.9010 | 0.9169 | 0.7707 |

We chose grouped frozen embeddings SVM as our best model to use as the classifier in part 1b.

The difficult cases evaluation shows that all models struggling with questions phrased as “How about [cuisine] food” and with denial utterances.

## Part 1b
Part 1b uses the grouped frozen embeddings SVM classifier from Part 1a to run a restaurant recommendation dialogue. It extracts food, area, and price preferences, looks up matching restaurants, and can explain additional requirements.

The Part 1b options in `main.py` are:

6. Restaurant recommendations with concise responses
7. Restaurant recommendations with detailed responses (this is our reasoning transparancy feature)
8. Reference dialogue tests

For options 6 and 7, choose a fallback method when prompted:
- `levenshtein` checks for spelling errors in a preference.
- `embeddings` uses semantic similarity to match a preference.

The detailed response option enables reasoning transparency. This is the runtime toggle used to compare concise and detailed responses.

The restaurant recommendation pipelines (6 and 7) will only quit when the user ends the conversation by saying things like "thank you" or "goodbye", alternatively `ctrl+c` can always be used to completely stop the program. The log of the interaction will be saved under the `part_1b/logs` folder.

Option 8 tests the dialogs in `data/reference_dialogs.txt` with both fallback methods and detailed responses. The generated transcripts are saved to `part_1b/tests/reference_dialogs_levenshtein.txt` and `part_1b/tests/reference_dialogs_embeddings.txt`.

## Results part 1b 
The reference tests showed our system performing as intended and passing all 20 reference dialogs. There are however a few points to note:
- Dialog 3: 
    - In the provided dataset this restaurant does not have a postal code, which our system correctly denotes. Thus it differs from the reference dialog, but we don't see this as a fail.
- dialog 9, 10, 14, 15 and 20: 
    - Works correctly, just provided a different (but still valid) recommendation compared to the reference dialog.
- dialog 12 and 13:
    - With levenshtein fallback both pass, but the embeddings fallback both fail, as that method seemingly unable to notice the typo.

All other dialogs are manually checked and found passing or similar enough to pass. 