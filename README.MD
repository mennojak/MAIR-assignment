# INFOMAIR Restaurant Recommendation Dialog System

This repository contains Part 1a and Part 1b of the project.

- `data/` - received datasets
    - `processed/` - processed split datasets for part 1a model training 
- `part1a/` - dialog-act classification
    - `models/` - trained models and embeddings
    - `results/` - results from all experiments
    - `pipelines.py` - orchastrator for all part 1a pipelines
    - all other files implement helper functions the orchastrator uses for the different pipelines.
- `part1b/` - dialog-management extension built on top of Part 1a
- `main.py` - top-level pipeline selector

Part 1a is built first. Part 1b reuses the selected Part 1a classifier and adds dialog management, slot extraction, restaurant lookup, reasoning, configurable behavior and interaction logging.

## Installation guide
* Create your own python environment (e.g. venv)
* Install the requirements `pip install requirements.txt` 

## Part 1a
This first part of the project asked us to create/train multiple text classifiers that take a user input and predict 1 of 15 possible dialog acts. It includes a baseline model with hardcoded rules and multiple versions of a Logistical Regression model and an SVM (Support Vector Machine) model. 

The two machine learning models will be trained on both BoW (Bag of Words) and Frozen distilbert embeddings. Additionally the given dataset is also split in two different ways, once randomly (called `original`) and the other focussed on handling duplicates (called `grouped`). Which gives a total of 9 different model/data split combinations (1 of which remains the baseline model) that have been evaluated to find the best classifier. 

Additionally a held out dataset for testing `dialog_acts_test.dat` is meant to be provided by the teachers for another round of evaluations. All evalutations are found under the `part_1a/results` folder and used for the report. Re-running all the different pipelines should yield the exact same results, because we made use of the seed `12` (instead of a random one).

Running `main.py` provides the different options for pipelines (i.e. different parts/deliverables of the project) that can be run. For part 1a these include:
1. Training pipeline
2. Evaluation pipeline
3. Held-out test set pipeline
    - Add the `dialog_acts_test.dat` file under the `data` folder before running.
4. Difficult cases pipeline
5. Interactive classification pipeline

## Part 1b
TODO