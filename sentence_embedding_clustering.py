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


def plot_som(x, y):
    labels_to_colors = {
        "location": 'lightcoral',
        "request": 'brown',
        "weather": 'red',
        "duration": 'salmon',
        "language": 'coral',
        "idiom": 'chocolate',
        "desire": 'saddlebrown',
        "frequency": 'peru',
        "sports": 'darkorange',
        "advice": 'antiquewhite',
        "school": 'orange',
        "ability": 'gold',
        "dialogue": 'khaki',
        "comparison": 'olive',
        "romance": 'yellow',
        "time": 'olivedrab',
        "proverb": 'greenyellow',
        "offer": 'darkseagreen',
        "musical instrument": 'palegreen',
        "suggestion": 'darkgreen',
        "year": 'lime',
        "obligation": 'mediumaquamarine',
        "likes": 'aquamarine',
        "necessity": 'lightseagreen',
        "exclamative": 'turquoise',
        "age": 'darkblue',
        "mathematics": 'rebeccapurple',
        "ignorance": 'indigo',
        "purpose": 'darkviolet',
        "permission": 'purple',
        "gambling": 'fuchsia',
        "reason": 'hotpink',
        "restaurant": 'crimson',
        "possibility": 'lightpink',
        "courtroom": 'slategrey',
    }

    color = [labels_to_colors[label] for label in labels]
    plt.figure(figsize=(x, y))
    for i, (t, c, vec) in enumerate(zip(ids, color, embeddings)):
        winnin_position = som.winner(vec)
        plt.text(winnin_position[0],
                 winnin_position[1] + np.random.rand() * .9,
                 t,
                 color=c)

    plt.xticks(range(x))
    plt.yticks(range(y))
    plt.grid()
    plt.xlim([0, x])
    plt.ylim([0, y])
    plt.plot()
    plt.show()


def winmap(vectors, items):
    wm = defaultdict(list)
    tot = len(vectors)
    for item, vector in enumerate(vectors):
        print("\rCreating winmap... %d%% " % np.floor((item / tot) * 100), end="")
        wm[som.winner(vector)].append(items[item])
    print("Done.")
    return wm


if __name__ == '__main__':
    model = load_model(join(MODELS_DIR, 'eng_model.h5'))
    embedder = keras.backend.function(model.layers[0].input, model.layers[-2].output)
    MAX_SEQUENCE_LENGTH = model.layers[0].input_shape[0][1]

    with open(join(MODELS_DIR, ENG_TOKENIZER), 'rb') as handle:
        print("Loading tokenizer...", end=" ")
        tokenizer = pickle.load(handle)
        print("Done.")

    # Read english sentences
    sentences = tatoeba.read_sentences(tatoeba.ENG_SENT)

    # Keep only the sentences that have japanese translation
    with open(join(TATOEBA_PATH, ENG_JPN_LINKS), 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        couples = {row[0]: row[1] for row in reader}

    for key in list(sentences.keys()):
        if key not in couples.keys():
            sentences.pop(key)

    # data = tokenizer.texts_to_sequences(sentences)
    # data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    # data = embedder(data)
    #
    # som_x = 40
    # som_y = 40
    # som = MiniSom(som_x, som_y, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5, activation_distance='cosine')
    # som.random_weights_init(data)
    # som.train(data, 500, verbose=True)
    #
    # winmap = winmap(data, sentences)
    # data = None
    #
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
