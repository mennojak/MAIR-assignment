import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from part_1a.bow_models import clean_text

tokenizer = None
bert = None


def load_bert():
    global tokenizer, bert
    if bert == None:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        bert = AutoModel.from_pretrained('distilbert-base-uncased')
        bert.eval()
        # freeze the weights
        for p in bert.parameters():
            p.requires_grad = False


def make_embeddings(utterances):
    load_bert()
    all_embeddings = []
    # do it in batches of 64, otherwise it is very slow
    for i in range(0, len(utterances), 64):
        batch = utterances[i:i+64]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt')
        with torch.no_grad():
            output = bert(**tokens)
        hidden = output.last_hidden_state
        # mean pooling over the tokens, the mask is used so the padding does not count
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        summed = (hidden * mask).sum(1)
        count = mask.sum(1)
        mean = summed / count
        all_embeddings.append(mean.numpy())
    return np.concatenate(all_embeddings)


def train_frozen_embeddings_classifier(df_train, grouped=False):
    if grouped:
        split = 'grouped'
    else:
        split = 'original'

    utterances = []
    for u in df_train['utterance']:
        utterances.append(clean_text(u))
    labels = df_train['dialog_act']

    if not os.path.exists('part_1a/models'):
        os.makedirs('part_1a/models')

    # the embeddings are saved because making them takes long
    embeddings_file = 'part_1a/models/embeddings_' + split + '.npy'
    if os.path.exists(embeddings_file):
        X_train = np.load(embeddings_file)
        print('loaded saved embeddings for', split)
    else:
        print('making embeddings for', split, 'this can take a few minutes')
        X_train = make_embeddings(utterances)
        np.save(embeddings_file, X_train)

    # same two classifiers as for bow
    lr = LogisticRegression(max_iter=2000, random_state=12)
    lr.fit(X_train, labels)
    joblib.dump({'model': lr, 'bert': 'distilbert-base-uncased', 'pooling': 'mean', 'max_length': 64},
                'part_1a/models/model_frozen_embeddings_' + split + '_LR.joblib')
    print('trained model_frozen_embeddings_' + split + '_LR')

    svm = LinearSVC(dual=True, max_iter=10000, random_state=12)
    svm.fit(X_train, labels)
    joblib.dump({'model': svm, 'bert': 'distilbert-base-uncased', 'pooling': 'mean', 'max_length': 64},
                'part_1a/models/model_frozen_embeddings_' + split + '_SVM.joblib')
    print('trained model_frozen_embeddings_' + split + '_SVM')


def run_inference_frozen_embeddings_model(model, utterances):
    saved = joblib.load('part_1a/' + model + '.joblib')
    clf = saved['model']

    predictions = []
    for u in utterances:
        u = clean_text(u)
        if u == '':
            predictions.append('null')
        else:
            emb = make_embeddings([u])
            pred = clf.predict(emb)[0]
            predictions.append(pred)
    return predictions