import os
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def clean_text(text):
    # lowercase and remove punctuation
    if type(text) != str:
        text = ''
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub('[^a-z0-9 ]', ' ', text)
    # remove double spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    text = text.strip()
    return text


def train_bow_classifier(df_train, grouped=False):
    if grouped:
        split = 'grouped'
    else:
        split = 'original'

    utterances = []
    for u in df_train['utterance']:
        utterances.append(clean_text(u))
    labels = df_train['dialog_act']

    # the vectorizer is only fitted on the training set
    # words in the test set that are not in the vocabulary are ignored by transform
    vectorizer = CountVectorizer(token_pattern=r"\S+")
    X_train = vectorizer.fit_transform(utterances)
    print(split, 'vocabulary size:', len(vectorizer.vocabulary_))

    if not os.path.exists('part_1a/models'):
        os.makedirs('part_1a/models')
    joblib.dump(vectorizer, 'part_1a/models/bow_vectorizer_' + split + '.joblib')

    # logistic regression
    lr = LogisticRegression(max_iter=2000, random_state=12)
    lr.fit(X_train, labels)
    joblib.dump({'model': lr, 'split': split}, 'part_1a/models/model_BoW_' + split + '_LR.joblib')
    print('trained model_BoW_' + split + '_LR')

    # svm
    svm = LinearSVC(dual=True, max_iter=10000, random_state=12)
    svm.fit(X_train, labels)
    joblib.dump({'model': svm, 'split': split}, 'part_1a/models/model_BoW_' + split + '_SVM.joblib')
    print('trained model_BoW_' + split + '_SVM\n')


def run_inference_bow_model(model_file_path, utterances):
    # model_file_path is something like models/model_BoW_grouped_LR
    saved = joblib.load('part_1a/' + model_file_path + '.joblib')
    model = saved['model']
    vectorizer = joblib.load('part_1a/models/bow_vectorizer_' + saved['split'] + '.joblib')

    predictions = []
    for u in utterances:
        u = clean_text(u)
        if u == '':
            # nothing left after cleaning, for example only punctuation
            predictions.append('null')
        else:
            X = vectorizer.transform([u])
            pred = model.predict(X)[0]
            predictions.append(pred)
    return predictions
