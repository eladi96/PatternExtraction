import numpy as np
import baseline_sentence_clustering as baseline
import eng_sentence_clustering as eng
import multilingual_sentence_clustering as multi
from sklearn.metrics.pairwise import cosine_similarity
from sequence_similarity import trigram_similarity


if __name__ == '__main__':

    indices = [31473]

    # BASELINE
    som, wm, sentences, embeddings = baseline.load_data()
    print("Model: baseline")
    for idx in indices:

        test_sent = sentences[idx]
        test_emb = embeddings[idx]

        x, y = som.winner(test_emb)
        similar_idx = wm[x][y]
        scores = dict()
        for index in similar_idx:
            sem_similarity = cosine_similarity(np.array(test_emb).reshape(1, -1),
                                               np.array(embeddings[index]).reshape(1, -1))[0][0]
            scores[sentences[index]] = sem_similarity
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        scores = {k: v for count, (k, v) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)) if
                  count <= 10}
        print("Sentence: ", test_sent)
        for sent, score in scores.items():
            print(sent, score)

    # MULTILINGUAL WITH POS
    som, wm, sentences, embeddings = multi.load_data()
    print("Model: Multilingual with Pos Similarity")
    for idx in indices:

        test_sent = sentences[idx]
        test_emb = embeddings[idx]
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
        print("Sentence: ", test_sent)
        for sent, score in scores.items():
            print(sent, score)
