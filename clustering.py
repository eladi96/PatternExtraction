import os
import csv
import keras
import pickle
import numpy as np
from constants import *
from minisom import MiniSom
from scipy.special import binom
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Recall, Precision
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt


def plot_data(data, mdl):
    plt.style.use('ggplot')
    labels = [30, 60, 90]
    ics = np.arange(len(labels))
    width = 0.15

    plt.figure(figsize=(5, 5))
    plt.bar(ics - width, data['SOM'], width, label='SOM', color='#00b3a7')
    plt.bar(ics, data['GMM'], width, label='GMM', color='#e27a03')
    plt.bar(ics + width, data['KMED'], width, label='K-Medoids', color='#009900')
    titles = {'baseline': 'Baseline',
              'eng': 'English',
              'eng_jpn': 'English - Japanese',
              'eng_ita': 'English - Italian'}
    plt.title(titles[mdl] + ' model')
    plt.xticks(ics, labels)
    plt.xlabel('Clusters')
    plt.ylim(top=1)
    plt.ylabel('ARI')
    plt.legend()
    plt.savefig(os.path.join(CLUSTERING, mdl + "_ari.png"))


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def preprocessing():
    with open(os.path.join(TATOEBA, TAGGED_SENT), 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        sentences = [row for row in reader]

    with open(os.path.join(CLASSIFICATION, TOKENIZERS, ENG_TOKENIZER), 'rb') as handle:
        print("Loading english tokenizer...", end=" ")
        eng_tokenizer = pickle.load(handle)
        print("Done.")

    with open(os.path.join(CLASSIFICATION, TOKENIZERS, JPN_TOKENIZER), 'rb') as handle:
        print("Loading japanese tokenizer...", end=" ")
        jpn_tokenizer = pickle.load(handle)
        print("Done.")

    with open(os.path.join(CLASSIFICATION, TOKENIZERS, ITA_TOKENIZER), 'rb') as handle:
        print("Loading italian tokenizer...", end=" ")
        ita_tokenizer = pickle.load(handle)
        print("Done.")

    baseline_model = load_model(os.path.join(CLASSIFICATION, MODELS, 'baseline_model.h5'),
                                custom_objects={'Recall': Recall, 'Precision': Precision})
    baseline_embedder = keras.backend.function(baseline_model.layers[0].input, baseline_model.layers[-3].output)

    eng_model = load_model(os.path.join(CLASSIFICATION, MODELS, 'eng_model.h5'),
                           custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_embedder = keras.backend.function(eng_model.layers[0].input, eng_model.layers[-3].output)

    eng_jpn_model = load_model(os.path.join(CLASSIFICATION, MODELS, 'eng_jpn_model.h5'),
                               custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_jpn_embedder = keras.backend.function(
        [eng_jpn_model.layers[0].input, eng_jpn_model.layers[1].input], eng_jpn_model.layers[-3].output)

    eng_ita_model = load_model(os.path.join(CLASSIFICATION, MODELS, 'eng_ita_model.h5'),
                               custom_objects={'Recall': Recall, 'Precision': Precision})
    eng_ita_embedder = keras.backend.function(
        [eng_ita_model.layers[0].input, eng_ita_model.layers[1].input], eng_ita_model.layers[-3].output)

    MAX_SEQUENCE_ENG = eng_model.layers[0].input_shape[0][1]
    MAX_SEQUENCE_JPN = eng_jpn_model.layers[1].input_shape[0][1]
    MAX_SEQUENCE_ITA = eng_ita_model.layers[1].input_shape[0][1]

    baseline_embeddings = list()
    eng_embeddings = list()
    eng_jpn_embeddings = list()
    eng_ita_embeddings = list()

    for count, sent in enumerate(sentences):
        print("\rSentence ", count, end="")
        eng_sent = sent[2]
        jpn_sent = sent[3]
        ita_sent = sent[4]
        eng_seq = pad_sequences(eng_tokenizer.texts_to_sequences([eng_sent]), maxlen=MAX_SEQUENCE_ENG)
        jpn_seq = pad_sequences(jpn_tokenizer.texts_to_sequences([jpn_sent]), maxlen=MAX_SEQUENCE_JPN)
        ita_seq = pad_sequences(ita_tokenizer.texts_to_sequences([ita_sent]), maxlen=MAX_SEQUENCE_ITA)
        baseline_embeddings.append(baseline_embedder(eng_seq)[0].tolist())
        eng_embeddings.append(eng_embedder(eng_seq)[0].tolist())
        eng_jpn_embeddings.append(eng_jpn_embedder([eng_seq, jpn_seq])[0].tolist())
        eng_ita_embeddings.append(eng_ita_embedder([eng_seq, ita_seq])[0].tolist())
    print("Done!")

    with open(os.path.join(CLUSTERING, 'sentences.pickle'), 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(CLUSTERING, 'baseline_embeddings.pickle'), 'wb') as handle:
        pickle.dump(baseline_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(CLUSTERING, 'eng_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(CLUSTERING, 'eng_jpn_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_jpn_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(CLUSTERING, 'eng_ita_embeddings.pickle'), 'wb') as handle:
        pickle.dump(eng_ita_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def som_clustering(data, dim_x, dim_y):
    som = MiniSom(dim_x, dim_y, EMBEDDING_DIM, sigma=0.3, learning_rate=0.5,
                  activation_distance='cosine')
    som.random_weights_init(data)
    som.train_batch(data, len(data))

    def compute_cluster(item):
        x, y = som.winner(item)
        return x * dim_x + y

    clustering = [compute_cluster(item) for item in data]
    return clustering


def gmm_clustering(data, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, max_iter=300)
    return gmm.fit_predict(data)


def kmedoids_clustering(data, n_clusters):
    kmed = KMedoids(n_clusters, metric='cosine', init='k-medoids++')
    kmed.fit(data)
    return kmed.predict(data)


def adjusted_rand_index(clustering):
    with open(os.path.join(CLUSTERING, 'sentences.pickle'), 'rb') as handle:
        sentences = pickle.load(handle)

    with open(os.path.join(TATOEBA, BEST_TAGS), mode='r') as file:
        tags_list = {line.split('\t')[0]: count for count, line in enumerate(file)}

    cont_table = np.zeros((len(tags_list), len(clustering)))
    for sent_id, sent in enumerate(sentences):
        cont_table[tags_list[sent[0]]][clustering[sent_id]] += 1

    n = len(sentences)
    sum_ij = sum(binom(n_ij, 2) for n_ij in np.nditer(cont_table))
    sum_i = sum(binom(a_i, 2) for a_i in np.sum(cont_table, axis=1))
    sum_j = sum(binom(b_j, 2) for b_j in np.sum(cont_table, axis=0))
    expected = (sum_i * sum_j) / binom(n, 2)

    score = (sum_ij - expected) / ((sum_i + sum_j) / 2 - expected)
    return np.around(score, decimals=4)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # preprocessing()
    for model in [
        'baseline',
        'eng',
        'eng_jpn',
        'eng_ita'
    ]:
        print(Bcolors.BOLD + "Clustering with", model, "model." + Bcolors.ENDC)
        with open(os.path.join(CLUSTERING, model + '_embeddings.pickle'), 'rb') as hdl:
            embeddings = pickle.load(hdl)
            som_data = [adjusted_rand_index(som_clustering(embeddings, 10, i)) for i in [3, 6, 9]]
            gmm_data = [adjusted_rand_index(gmm_clustering(embeddings, i)) for i in [30, 60, 90]]
            kmed_data = [adjusted_rand_index(kmedoids_clustering(embeddings, i)) for i in [30, 60, 90]]
            results = {'SOM': som_data, 'GMM': gmm_data, 'KMED': kmed_data}
            plot_data(results, model)
            with open(os.path.join(CLUSTERING, "results.txt"), 'a') as out:
                out.write("Clustering with " + model + " model.\n")
                out.write("SOM:" + str(som_data) + "\n")
                out.write("KMED: " + str(kmed_data) + "\n")
                out.write("GMM: " + str(gmm_data) + "\n\n\n")
