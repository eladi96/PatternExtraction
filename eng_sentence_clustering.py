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


def load_data():
    with open(join(SOM_DIR, 'eng_som.pickle'), 'rb') as handle:
        net = pickle.load(handle)

    with open(join(SOM_DIR, 'eng_winmap.pickle'), 'rb') as handle:
        w = pickle.load(handle)

    with open(join(SOM_DIR, 'eng_sentences.pickle'), 'rb') as handle:
        s = pickle.load(handle)

    with open(join(SOM_DIR, 'eng_embeddings.pickle'), 'rb') as handle:
        e = pickle.load(handle)

    print("English model loaded.")

    return net, w, s, e


def preprocessing():
    model = load_model(join(MODELS_DIR, 'eng_model.h5'), custom_objects={'Recall': Recall, 'Precision': Precision})
    embedder = keras.backend.function(model.layers[0].input, model.layers[-3].output)
    MAX_SEQUENCE_ENG = model.layers[0].input_shape[0][1]

    eng_sents = tatoeba.read_sentences(tatoeba.ENG_SENT)

    with open(join(MODELS_DIR, ENG_TOKENIZER), 'rb') as handle:
        print("Loading english tokenizer...", end=" ")
        eng_tokenizer = pickle.load(handle)
        print("Done.")

    with open(join(SOM_DIR, 'eng_embedded_sentences.tsv'), 'w') as output:
        writer = csv.writer(output, delimiter='\t')

        # Keep only the sentences that have japanese translation
        with open(join(TATOEBA_DIR, ENG_JPN_LINKS), 'r') as file:
            reader = csv.reader(file, delimiter='\t')

            for count, row in enumerate(reader):
                print("\rWriting row ", count, end="")
                eng_sent = eng_sents.pop(row[0], None)
                if eng_sent is not None:
                    eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences([eng_sent]), maxlen=MAX_SEQUENCE_ENG)
                    embedding = embedder(eng_seq)
                    row = [eng_sent]
                    row.extend(embedding[0])
                    writer.writerow(row)
    print("Done!")


def build_som():
    sentences = list()
    embeddings = list()
    with open(join(SOM_DIR, 'eng_embedded_sentences.tsv'), 'r') as file:
        tot = sum(1 for _ in file)
        file.seek(0)
        reader = csv.reader(file, delimiter='\t')
        for count, row in enumerate(reader):
            print("\rPreparing the dataset... %d%% " % np.floor((count / tot) * 100), end="")
            sentences.append(row[0])
            embeddings.append(list(map(float, row[1:])))
    with open(join(SOM_DIR, 'eng_sentences.pickle'), 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'eng_embeddings.pickle'), 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")

    dimension = int(np.sqrt(5 * np.sqrt(len(embeddings))))
    som = MiniSom(dimension, dimension, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5,
                         activation_distance='cosine')
    som.random_weights_init(embeddings)
    som.train_batch(embeddings, len(embeddings), verbose=True)
    with open(join(SOM_DIR, 'eng_som.pickle'), 'wb') as handle:
        pickle.dump(som, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("SOM saved to " + join(SOM_DIR, 'eng_som.pickle'))

    winmap = [[list() for _ in range(dimension)] for _ in range(dimension)]
    tot = len(embeddings)
    for idx, vector in enumerate(embeddings):
        print("\rCreating winmap... %d%% " % np.floor((idx / tot) * 100), end="")
        x_c, y_c = som.winner(vector)
        winmap[x_c][y_c].append(idx)
    print("Done.")
    with open(join(SOM_DIR, 'eng_winmap.pickle'), 'wb') as handle:
        pickle.dump(winmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Winmap saved to " + join(SOM_DIR, 'eng_winmap.pickle'))
