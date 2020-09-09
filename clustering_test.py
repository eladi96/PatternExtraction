import pickle

import numpy as np
from sentence_clustering import load_data
from sklearn.metrics.pairwise import cosine_similarity
from pylatex import Document, Command, MultiColumn, LineBreak, Section, Tabular, MultiRow
from pylatex.utils import NoEscape, bold
import constants
import random
import os

if __name__ == '__main__':

    test_doc = Document(geometry_options={"margin": "1in",
                                          "bottom": "1in"})
    test_doc.preamble.append(Command('title', 'Sentence similarity models comparison'))
    test_doc.preamble.append(Command('author', 'Enrico Ladisa'))
    test_doc.preamble.append(Command('date', NoEscape(r'\today')))
    test_doc.append(NoEscape(r'\maketitle'))

    answers_doc = Document(geometry_options={"margin": "0.5in",
                                             "bottom": "1in"})
    answers_doc.preamble.append(Command('title', 'Sentence similarity models comparison - answers'))
    answers_doc.preamble.append(Command('author', 'Enrico Ladisa'))
    answers_doc.preamble.append(Command('date', NoEscape(r'\today')))
    answers_doc.append(NoEscape(r'\maketitle'))

    with test_doc.create(Section("Introduction")):
        test_doc.append(
            "The sequent test aims to choose which is the best model to compute sentences similarity between "
            "three alternatives.")
        test_doc.append(LineBreak())
        test_doc.append("Our objective is to find in a collection of phrases the most similar sentences to a given "
                        "clause, so we performed this task with three different models. We are going the present to "
                        "the examiners the top three similar sentences found by each model anonymously for 20 instances"
                        ", and they will have to choose which is, in their opinon, the best result.")

    with open(os.path.join(constants.SOM_DIR, 'sentences.pickle'), 'rb') as handle:
        sentences = pickle.load(handle)
    test_sents_idx = [31473, 34034, 12334, 1223, 45323, 12343, 5423]
    results = [dict() for _ in range(len(test_sents_idx))]
    models = ['baseline', 'eng', 'multilingual']

    # Similarity without pos sequence comparison
    for model in models:
        som, wm, embeddings = load_data(model)

        for count, idx in enumerate(test_sents_idx):
            test_sent = sentences[idx]
            test_emb = embeddings[idx]
            x, y = som.winner(test_emb)
            similar_idx = wm[x][y]
            scores = dict()
            for index in similar_idx:
                sem_similarity = cosine_similarity(np.array(test_emb).reshape(1, -1),
                                                   np.array(embeddings[index]).reshape(1, -1))[0][0]
                scores[sentences[index]] = sem_similarity
            scores = {k: v for count, (k, v) in
                      enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True)) if
                      count <= 3 and k is not test_sent}
            results[count][model] = list(scores.keys())
            results[count]['sentence'] = test_sent[0]

    with test_doc.create(Section('Test')):
        test_doc.append(Command('begin', 'center'))
        test_doc.append(Command(command="renewcommand\\arraystretch",
                                arguments='1.5'))

        answers_doc.append(Command('begin', 'center'))
        answers_doc.append(Command(command="renewcommand\\arraystretch",
                                   arguments='1.5'))

        for count, idx in enumerate(test_sents_idx):
            with answers_doc.create(Tabular('| p{13cm} | p{3cm} |', pos='centered')) as answer_table:
                with test_doc.create(Tabular('| p{13cm} | p{2.5cm} |', pos='centered')) as table:
                    table.add_hline()
                    answer_table.add_hline()
                    table.add_row((MultiColumn(2, align='| l |',
                                               data=bold(str(count + 1) + " - " + results[count]['sentence'])),))
                    table.add_hline()
                    answer_table.add_row((MultiColumn(2, align='| l |',
                                                      data=bold(str(count + 1) + " - " + results[count]['sentence'])),))
                    answer_table.add_hline()
                    random.shuffle(models)
                    for model in models:
                        answer_table.add_row(("", MultiRow(4, data=model)))
                        for sentence in results[count][model]:
                            if type(sentence) is tuple:
                                sentence = sentence[0]
                            table.add_row((sentence, ""))
                            answer_table.add_row((sentence, ""))
                        table.add_hline()
                        answer_table.add_hline()
            test_doc.append(LineBreak())
            answers_doc.append(LineBreak())

        test_doc.append(Command('end', 'center'))
        answers_doc.append(Command('end', 'center'))

    test_doc.generate_pdf(os.path.join(constants.SOM_DIR, 'models_comparison'), clean_tex=True)
    answers_doc.generate_pdf(os.path.join(constants.SOM_DIR, 'models_comparison_answers'), clean_tex=True)
