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
from sequence_similarity import trigram_similarity


def load_data():
    with open(join(SOM_DIR, SOM), 'rb') as handle:
        net = pickle.load(handle)

    with open(join(SOM_DIR, WINMAP), 'rb') as handle:
        w = pickle.load(handle)

    with open(join(SOM_DIR, 'sentences.pickle'), 'rb') as handle:
        s = pickle.load(handle)

    with open(join(SOM_DIR, 'embeddings.pickle'), 'rb') as handle:
        e = pickle.load(handle)

    print("Multilingual model loaded.")

    return net, w, s, e


def preprocessing():
    model = load_model(join(MODELS_DIR, 'combined_model.h5'), custom_objects={'Recall': Recall, 'Precision': Precision})
    embedder = keras.backend.function([model.layers[0].input, model.layers[1].input], model.layers[-3].output)
    MAX_SEQUENCE_ENG = model.layers[0].input_shape[0][1]
    MAX_SEQUENCE_JPN = model.layers[1].input_shape[0][1]

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

    with open(join(SOM_DIR, 'embedded_sentences.tsv'), 'w') as output:
        writer = csv.writer(output, delimiter='\t')

        # Keep only the sentences that have japanese translation
        with open(join(TATOEBA_DIR, ENG_JPN_LINKS), 'r') as file:
            reader = csv.reader(file, delimiter='\t')

            for count, row in enumerate(reader):
                print("\rWriting row ", count, end="")
                eng_sent = eng_sents.pop(row[0], None)
                jpn_sent = jpn_sents.get(row[1])
                if eng_sent is not None:
                    eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences([eng_sent]), maxlen=MAX_SEQUENCE_ENG)
                    jpn_seq = pad_sequences(jpn_tokenizer.texts_to_sequences([jpn_sent]), maxlen=MAX_SEQUENCE_JPN)
                    embedding = embedder([eng_seq, jpn_seq])
                    row = [eng_sent, jpn_sent]
                    row.extend(embedding[0])
                    writer.writerow(row)
    print("Done!")


def build_som():
    sents = list()
    embeds = list()
    with open(join(SOM_DIR, 'embedded_sentences.tsv'), 'r') as file:
        tot = sum(1 for _ in file)
        file.seek(0)
        reader = csv.reader(file, delimiter='\t')
        for count, row in enumerate(reader):
            print("\rPreparing the dataset... %d%% " % np.floor((count / tot) * 100), end="")
            sents.append((row[0], row[1]))
            embeds.append(list(map(float, row[2:])))
    with open(join(SOM_DIR, 'sentences.pickle'), 'wb') as handle:
        pickle.dump(sents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(SOM_DIR, 'embeddings.pickle'), 'wb') as handle:
        pickle.dump(embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")

    dimension = int(np.sqrt(5 * np.sqrt(len(embeds))))
    selforgmap = MiniSom(dimension, dimension, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5,
                         activation_distance='cosine')
    selforgmap.random_weights_init(embeds)
    selforgmap.train_batch(embeds, len(embeds), verbose=True)
    with open(join(SOM_DIR, SOM), 'wb') as handle:
        pickle.dump(selforgmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("SOM saved to " + join(MODELS_DIR, SOM))

    winmap = [[list() for _ in range(dimension)] for _ in range(dimension)]
    tot = len(embeds)
    for idx, vector in enumerate(embeds):
        print("\rCreating winmap... %d%% " % np.floor((idx / tot) * 100), end="")
        x_c, y_c = selforgmap.winner(vector)
        winmap[x_c][y_c].append(idx)
    print("Done.")
    with open(join(SOM_DIR, WINMAP), 'wb') as handle:
        pickle.dump(winmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Winmap saved to " + join(MODELS_DIR, WINMAP))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    som, wm, sentences, embeddings = load_data()

    test_sent = sentences[4563]
    test_emb = embeddings[4563]
    x, y = som.winner(test_emb)
    similar_idx = wm[x][y]
    scores = dict()
    for index in similar_idx:
        sem_similarity = cosine_similarity(np.array(test_emb).reshape(1, -1),
                                           np.array(embeddings[index]).reshape(1, -1))[0][0]
        eng_pos_similarity = trigram_similarity(test_sent[0], sentences[index][0], 'eng')
        jpn_pos_similarity = trigram_similarity(test_sent[1], sentences[index][1], 'jpn')
        scores[sentences[index]] = (sem_similarity + eng_pos_similarity) - (sem_similarity * eng_pos_similarity)
        scores[sentences[index]] = (scores[sentences[index]] + jpn_pos_similarity) - (
                scores[sentences[index]] * jpn_pos_similarity)
    scores = {k: v for count, (k, v) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)) if
              count <= 10}
    for sent, score in scores.items():
        print(sent, score)
