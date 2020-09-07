import os
import csv
import keras
import pickle
import tatoeba
import numpy as np
from constants import *
from os.path import join
from minisom import MiniSom
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Recall, Precision
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity


def load_data(model):

    if model in ['baseline', 'eng', 'multilingual']:

        with open(join(SOM_DIR, model + '_som.pickle'), 'rb') as handle:
            som = pickle.load(handle)

        with open(join(SOM_DIR, model + '_winmap.pickle'), 'rb') as handle:
            winmap = pickle.load(handle)

        with open(join(SOM_DIR, model + '_embeddings.pickle'), 'rb') as handle:
            embeddings = pickle.load(handle)

        print(model, "model loaded.")
        return som, winmap, embeddings
    else:
        print("Incorrect model name.")


def preprocessing():

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

    baseline_model = load_model(join(MODELS_DIR, 'baseline_model.h5'),
                                custom_objects={'Recall': Recall, 'Precision': Precision})
    baseline_embedder = keras.backend.function(baseline_model.layers[0].input, baseline_model.layers[-3].output)

    eng_model = load_model(join(MODELS_DIR, 'eng_model.h5'), custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_embedder = keras.backend.function(eng_model.layers[0].input, eng_model.layers[-3].output)

    multilingual_model = load_model(join(MODELS_DIR, 'combined_model.h5'),
                                    custom_objects={'Recall': Recall, 'Precision': Precision})
    multilingual_embedder = keras.backend.function(
        [multilingual_model.layers[0].input, multilingual_model.layers[1].input], multilingual_model.layers[-3].output)

    jpn_model = load_model(join(MODELS_DIR, 'jpn_model.h5'), custom_objects={'Recall': Recall, 'Precision': Precision})
    jpn_embedder = keras.backend.function(jpn_model.layers[0].input, jpn_model.layers[-3].output)

    MAX_SEQUENCE_ENG = eng_model.layers[0].input_shape[0][1]
    MAX_SEQUENCE_JPN = jpn_model.layers[0].input_shape[0][1]

    sentences = list()
    baseline_embeddings = list()
    eng_embeddings = list()
    multilingual_embeddings = list()
    jpn_embeddings = list()

    # Keep only the sentences that have japanese translation
    with open(join(TATOEBA_DIR, ENG_JPN_LINKS), 'r') as file:
        reader = csv.reader(file, delimiter='\t')

        for count, row in enumerate(reader):
            print("\rSentence ", count, end="")
            eng_sent = eng_sents.pop(row[0], None)
            jpn_sent = jpn_sents.get(row[1])
            if eng_sent is not None:
                sentences.append((eng_sent, jpn_sent))
                eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences([eng_sent]), maxlen=MAX_SEQUENCE_ENG)
                jpn_seq = pad_sequences(jpn_tokenizer.texts_to_sequences([jpn_sent]), maxlen=MAX_SEQUENCE_JPN)
                baseline_embeddings.append(baseline_embedder(eng_seq)[0].tolist())
                eng_embeddings.append(eng_embedder(eng_seq)[0].tolist())
                multilingual_embeddings.append(multilingual_embedder([eng_seq, jpn_seq])[0].tolist())
                jpn_embeddings.append(jpn_embedder(jpn_seq)[0].tolist())
        print("Done!")

    with open(join(SOM_DIR, 'sentences.pickle'), 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'baseline_embeddings.pickle'), 'wb') as handle:
        pickle.dump(baseline_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'eng_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'multilingual_embeddings.pickle'), 'wb') as handle:
        pickle.dump(multilingual_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'jpn_embeddings.pickle'), 'wb') as handle:
        pickle.dump(jpn_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_baseline_som():

    with open(join(SOM_DIR, 'baseline_embeddings.pickle'), 'rb') as handle:
        baseline_embeddings = pickle.load(handle)

    tot = len(baseline_embeddings)
    dimension = int(np.sqrt(5 * np.sqrt(tot)))

    som = MiniSom(dimension, dimension, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5,
                  activation_distance='cosine')
    som.random_weights_init(baseline_embeddings)
    som.train_batch(baseline_embeddings, tot, verbose=True)
    with open(join(SOM_DIR, 'baseline_som.pickle'), 'wb') as handle:
        pickle.dump(som, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("SOM saved to " + join(MODELS_DIR, 'baseline_som.pickle'))

    winmap = [[list() for _ in range(dimension)] for _ in range(dimension)]
    for idx, vector in enumerate(baseline_embeddings):
        print("\rCreating winmap... %d%% " % np.floor((idx / tot) * 100), end="")
        x, y = som.winner(vector)
        winmap[x][y].append(idx)
    print("Done.")
    with open(join(SOM_DIR, 'baseline_winmap.pickle'), 'wb') as handle:
        pickle.dump(winmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Winmap saved to " + join(MODELS_DIR, 'baseline_winmap.pickle'))


def build_eng_som():

    with open(join(SOM_DIR, 'eng_embeddings.pickle'), 'rb') as handle:
        eng_embeddings = pickle.load(handle)

    tot = len(eng_embeddings)
    dimension = int(np.sqrt(5 * np.sqrt(tot)))

    som = MiniSom(dimension, dimension, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5,
                  activation_distance='cosine')
    som.random_weights_init(eng_embeddings)
    som.train_batch(eng_embeddings, tot, verbose=True)
    with open(join(SOM_DIR, 'eng_som.pickle'), 'wb') as handle:
        pickle.dump(som, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("SOM saved to " + join(MODELS_DIR, 'eng_som.pickle'))

    winmap = [[list() for _ in range(dimension)] for _ in range(dimension)]
    for idx, vector in enumerate(eng_embeddings):
        print("\rCreating winmap... %d%% " % np.floor((idx / tot) * 100), end="")
        x, y = som.winner(vector)
        winmap[x][y].append(idx)
    print("Done.")
    with open(join(SOM_DIR, 'eng_winmap.pickle'), 'wb') as handle:
        pickle.dump(winmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Winmap saved to " + join(MODELS_DIR, 'eng_winmap.pickle'))


def build_multilingual_som():

    with open(join(SOM_DIR, 'multilingual_embeddings.pickle'), 'rb') as handle:
        multilingual_embeddings = pickle.load(handle)

    tot = len(multilingual_embeddings)
    dimension = int(np.sqrt(5 * np.sqrt(tot)))

    som = MiniSom(dimension, dimension, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5,
                  activation_distance='cosine')
    som.random_weights_init(multilingual_embeddings)
    som.train_batch(multilingual_embeddings, tot, verbose=True)
    with open(join(SOM_DIR, 'multilingual_som.pickle'), 'wb') as handle:
        pickle.dump(som, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("SOM saved to " + join(MODELS_DIR, 'multilingual_som.pickle'))

    winmap = [[list() for _ in range(dimension)] for _ in range(dimension)]
    for idx, vector in enumerate(multilingual_embeddings):
        print("\rCreating winmap... %d%% " % np.floor((idx / tot) * 100), end="")
        x, y = som.winner(vector)
        winmap[x][y].append(idx)
    print("Done.")
    with open(join(SOM_DIR, 'multilingual_winmap.pickle'), 'wb') as handle:
        pickle.dump(winmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Winmap saved to " + join(MODELS_DIR, 'multilingual_winmap.pickle'))


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    test_sents_idx = [31473, 34034, 12334, 1223, 45323, 12343, 5423]
    results = [dict() for _ in range(len(test_sents_idx))]

    with open(join(SOM_DIR, 'sentences.pickle'), 'rb') as hndl:
        sentences = pickle.load(hndl)

    indice = 12334
    test_sent = sentences[indice]
    print(test_sent)

    # for model in ['baseline', 'eng', 'multilingual']:
    #     sm, wm, emb = load_data(model)
    #     test_emb = emb[indice]
    #     x, y = sm.winner(test_emb)
    #     similar_idx = wm[x][y]
    #     scores = dict()
    #     for index in similar_idx:
    #         eng_similarity = cosine_similarity(np.array(test_emb).reshape(1, -1),
    #                                            np.array(emb[index]).reshape(1, -1))[0][0]
    #         scores[sentences[index]] = eng_similarity
    #     scores = {k: v for count, (k, v) in
    #               enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)) if
    #               count <= 3 and k is not test_sent}
    #
    #     for k, v in scores.items():
    #         print(k, v)

    model = 'eng'
    sm, wm, emb = load_data(model)
    with open(join(SOM_DIR, 'jpn_embeddings.pickle'), 'rb') as handle:
        jpn_emb = pickle.load(handle)
    test_emb = emb[indice]
    x, y = sm.winner(test_emb)
    similar_idx = wm[x][y]
    scores = dict()
    for index in similar_idx:
        eng_similarity = cosine_similarity(np.array(test_emb).reshape(1, -1),
                                           np.array(emb[index]).reshape(1, -1))[0][0]
        jpn_similarity = cosine_similarity(np.array(jpn_emb[indice]).reshape(1, -1),
                                           np.array(jpn_emb[index]).reshape(1, -1))[0][0]
        scores[sentences[index]] = eng_similarity
    scores = {k: v for count, (k, v) in
              enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)) if
              count <= 20 and k is not test_sent}

    for k, v in scores.items():
        print(k, v)
