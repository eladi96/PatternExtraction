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


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_data(model):
    if model in ['baseline', 'eng', 'eng_jpn', 'eng_ita']:

        with open(join(SOM_DIR, model + '_som.pickle'), 'rb') as handle:
            som = pickle.load(handle)

        with open(join(SOM_DIR, model + '_winmap.pickle'), 'rb') as handle:
            winmap = pickle.load(handle)

        with open(join(SOM_DIR, model + '_embeddings.pickle'), 'rb') as handle:
            embeddings = pickle.load(handle)

        print(Bcolors.BOLD + model, "model loaded." + Bcolors.ENDC)
        return som, winmap, embeddings
    else:
        print("Incorrect model name.")


def preprocessing():
    eng_sents = tatoeba.read_sentences(tatoeba.ENG_SENT)
    jpn_sents = tatoeba.read_sentences(tatoeba.JPN_SENT)
    ita_sents = tatoeba.read_sentences(tatoeba.ITA_SENT)

    with open(join("models", ENG_TOKENIZER), 'rb') as handle:
        print("Loading english tokenizer...", end=" ")
        eng_tokenizer = pickle.load(handle)
        print("Done.")

    with open(join("models", JPN_TOKENIZER), 'rb') as handle:
        print("Loading japanese tokenizer...", end=" ")
        jpn_tokenizer = pickle.load(handle)
        print("Done.")

    with open(join(MODELS_DIR, ITA_TOKENIZER), 'rb') as handle:
        print("Loading italian tokenizer...", end=" ")
        ita_tokenizer = pickle.load(handle)
        print("Done.")

    baseline_model = load_model(join(MODELS_DIR, 'baseline_model.h5'),
                                custom_objects={'Recall': Recall, 'Precision': Precision})
    baseline_embedder = keras.backend.function(baseline_model.layers[0].input, baseline_model.layers[-1].output)

    eng_model = load_model(join(MODELS_DIR, 'eng_model.h5'), custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_embedder = keras.backend.function(eng_model.layers[0].input, eng_model.layers[-3].output)

    eng_jpn_model = load_model(join(MODELS_DIR, 'eng_jpn_model.h5'),
                               custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_jpn_embedder = keras.backend.function(
        [eng_jpn_model.layers[0].input, eng_jpn_model.layers[1].input], eng_jpn_model.layers[-3].output)

    eng_ita_model = load_model(join(MODELS_DIR, 'eng_ita_model.h5'),
                               custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_ita_embedder = keras.backend.function(
        [eng_ita_model.layers[0].input, eng_ita_model.layers[1].input], eng_ita_model.layers[-3].output)

    MAX_SEQUENCE_ENG = eng_model.layers[0].input_shape[0][1]
    MAX_SEQUENCE_JPN = eng_jpn_model.layers[1].input_shape[0][1]
    MAX_SEQUENCE_ITA = eng_ita_model.layers[1].input_shape[0][1]

    sentences = list()
    baseline_embeddings = list()
    eng_embeddings = list()
    eng_jpn_embeddings = list()
    eng_ita_embeddings = list()

    # Keep only the sentences that have japanese and italian translation
    with open(join(TATOEBA_DIR, ENG_JPN_LINKS), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        eng_jpn = {row[0]: row[1] for row in reader}
    with open(join(TATOEBA_DIR, ENG_ITA_LINKS), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        eng_ita = {row[0]: row[1] for row in reader}

    intersection = eng_jpn.keys() & eng_ita.keys()

    # {eng_id:(jpn_id, ita_id)}
    sentences_id = {eng_id: (eng_jpn[eng_id], eng_ita[eng_id]) for eng_id in intersection}
    del intersection, eng_jpn, eng_ita

    for count, (key, value) in enumerate(sentences_id.items()):
        print("\rSentence ", count, end="")
        eng_sent = eng_sents.pop(key, None)
        jpn_sent = jpn_sents.get(value[0])
        ita_sent = ita_sents.get(value[1])
        if eng_sent is not None:
            sentences.append((eng_sent, jpn_sent, ita_sent))
            eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences([eng_sent]), maxlen=MAX_SEQUENCE_ENG)
            jpn_seq = pad_sequences(jpn_tokenizer.texts_to_sequences([jpn_sent]), maxlen=MAX_SEQUENCE_JPN)
            ita_seq = pad_sequences(ita_tokenizer.texts_to_sequences([ita_sent]), maxlen=MAX_SEQUENCE_ITA)
            baseline_embeddings.append(baseline_embedder(eng_seq)[0].tolist())
            eng_embeddings.append(eng_embedder(eng_seq)[0].tolist())
            eng_jpn_embeddings.append(eng_jpn_embedder([eng_seq, jpn_seq])[0].tolist())
            eng_ita_embeddings.append(eng_ita_embedder([eng_seq, ita_seq])[0].tolist())
    print("Done!")

    with open(join(SOM_DIR, 'sentences.pickle'), 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'baseline_embeddings.pickle'), 'wb') as handle:
        pickle.dump(baseline_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'eng_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'eng_jpn_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_jpn_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'eng_ita_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_ita_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_som(model):
    if model in ['baseline', 'eng', 'eng_jpn', 'eng_ita']:

        with open(join(SOM_DIR, model + '_embeddings.pickle'), 'rb') as handle:
            embeddings = pickle.load(handle)

        tot = len(embeddings)
        dimension = int(np.sqrt(5 * np.sqrt(tot)))

        som = MiniSom(dimension, dimension, NUM_LABELS, sigma=0.3, learning_rate=0.5,
                      activation_distance='cosine')
        som.random_weights_init(embeddings)
        som.train_batch(embeddings, tot, verbose=True)
        with open(join(SOM_DIR, model + '_som.pickle'), 'wb') as handle:
            pickle.dump(som, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("SOM saved to " + join(SOM_DIR, model + '_som.pickle'))

        winmap = [[list() for _ in range(dimension)] for _ in range(dimension)]
        for idx, vector in enumerate(embeddings):
            print("\rCreating winmap... %d%% " % np.floor((idx / tot) * 100), end="")
            # noinspection PyShadowingNames
            x, y = som.winner(vector)
            winmap[x][y].append(idx)
        print("Done.")
        with open(join(SOM_DIR, model + '_winmap.pickle'), 'wb') as handle:
            pickle.dump(winmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Winmap saved to " + join(SOM_DIR, model + '_winmap.pickle'))
    else:
        print("Incorrect model name.")


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    test_sents_idx = []
    results = [dict() for _ in range(len(test_sents_idx))]

    with open(join(SOM_DIR, 'sentences.pickle'), 'rb') as hndl:
        sents = pickle.load(hndl)

    test_idx = np.random.randint(len(sents))
    print(Bcolors.HEADER + str(test_idx) + " - " + sents[test_idx][0] + Bcolors.ENDC)
    for mdl in ['baseline', 'eng', 'eng_jpn', 'eng_ita']:
        sm, wm, emb = load_data(mdl)
        test_emb = emb[test_idx]
        x, y = sm.winner(test_emb)
        similar_idx = wm[x][y]
        scores = dict()
        for index in similar_idx:
            sem_similarity = cosine_similarity(np.array(test_emb).reshape(1, -1),
                                               np.array(emb[index]).reshape(1, -1))[0][0]
            scores[sents[index]] = sem_similarity
        scores = {k: v for count, (k, v) in
                  enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)) if
                  count <= 3 and k is not sents[test_idx]}

        for k, v in scores.items():
            print(k[0])
        print()
