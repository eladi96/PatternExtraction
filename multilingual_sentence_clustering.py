import ast
import pickle
from collections import defaultdict
import tatoeba
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model
from os.path import join
from minisom import MiniSom
from sklearn.metrics.pairwise import cosine_similarity
from constants import *
import csv
import os
import pandas as pd


def preprocessing():
    model = load_model(join(MODELS_DIR, 'pretrained_model.h5'))
    embedder = keras.backend.function([model.layers[0].input, model.layers[1].input], model.layers[-3].output)
    MAX_SEQUENCE_ENG = model.layers[0].input_shape[0][1]
    MAX_SEQUENCE_JPN = model.layers[1].input_shape[0][1]

    # Read english sentences
    eng_sents = tatoeba.read_sentences(tatoeba.ENG_SENT)
    jpn_sents = tatoeba.read_sentences(tatoeba.JPN_SENT)

    with open(join(MODELS_DIR, ENG_TOKENIZER), 'rb') as handle:
        print("Loading english tokenizer...", end=" ")
        eng_tokenizer = pickle.load(handle)
        print("Done.")

    with open(join(MODELS_DIR, JPN_TOKENIZER), 'rb') as handle:
        print("Loading japanese tokenizer...", end=" ")
        jpn_tokenizer = pickle.load(handle)
        print("Done.")

    with open(join(TATOEBA_PATH, 'embedded_sentences.tsv'), 'w') as output:

        # Keep only the sentences that have japanese translation
        with open(join(TATOEBA_PATH, ENG_JPN_LINKS), 'r') as file:
            reader = csv.reader(file, delimiter='\t')

            for count, row in enumerate(reader):
                print("\rWriting row ", count, end="")
                eng_sent = eng_sents.pop(row[0], None)
                jpn_sent = jpn_sents.get(row[1])
                if eng_sent is not None:
                    eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences([eng_sent]), maxlen=MAX_SEQUENCE_ENG)
                    jpn_seq = pad_sequences(jpn_tokenizer.texts_to_sequences([jpn_sent]), maxlen=MAX_SEQUENCE_JPN)
                    embedding = embedder([eng_seq, jpn_seq])
                    output.write(
                        str(count) + "\t" + eng_sent + "\t" + jpn_sent + "\t" + str(embedding[0].tolist()) + "\n")


def winmap(vectors, items):
    wm = defaultdict(list)
    tot = len(vectors)
    for item, vector in enumerate(vectors):
        print("\rCreating winmap... %d%% " % np.floor((item / tot) * 100), end="")
        wm[som.winner(vector)].append(items.iloc[item])
    print("Done.")
    return wm


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    df = pd.read_csv(join(TATOEBA_PATH, 'prova.tsv'), sep='\t',
                     names=['index', 'eng', 'jpn', 'embedding'])

    embeddings = df['embedding'].apply(lambda x: ast.literal_eval(x))
    data = embeddings.tolist()
    sentences = df[['eng', 'jpn']]
    del df

    dimension = int(np.sqrt(5 * np.sqrt(len(data))))
    som = MiniSom(dimension, dimension, 2*EMBEDDING_DIM, sigma=0.3, learning_rate=0.5, activation_distance='cosine')
    som.random_weights_init(data)
    som.train(data, 500, verbose=True)

    sent_map = winmap(data, sentences)
    data = None
    print(sent_map)

    # test = input()
    # while test != "":
    #     test = embedder(pad_sequences(tokenizer.texts_to_sequences([test]), maxlen=MAX_SEQUENCE_LENGTH, padding='pre'))
    #     similar = winmap[som.winner(test[0])]
    #     scores = dict()
    #     for sent in similar:
    #         embedded = embedder(
    #             pad_sequences(tokenizer.texts_to_sequences([sent]), maxlen=MAX_SEQUENCE_LENGTH, padding='pre'))
    #         scores[sent] = cosine_similarity(test[0].reshape(1, -1), embedded[0].reshape(1, -1))
    #     scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    #     for sent, score in scores.items():
    #         print(sent, score)
    #     test = input()
