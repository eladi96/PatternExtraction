import tatoeba
import os
import fasttext
import pickle
import numpy as np
import keras
from constants import *
from tabulate import tabulate
from matplotlib import pyplot as plt
from japanese_tokenizer import JapaneseTokenizer
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pylatex.utils import NoEscape
from pylatex import Document, Section, Command, Tabular


def latex_summary():
    doc = Document()

    doc.preamble.append(Command('title', 'Models comparison'))
    doc.preamble.append(Command('author', 'Enrico Ladisa'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('Baseline Model')):
        values = avg_results(os.path.join(CLASSIFICATION, SUMMARY, 'baseline_model_summary.txt'))
        doc.append(Command('begin', 'center'))
        with doc.create(Tabular('l||r r r r', pos='centered')) as table:
            table.add_row(("Set", "Loss", "Accuracy", "Recall", "Precision"))
            table.add_hline()
            table.add_hline()
            table.add_row(("Training", values["Training"]["Loss"], values["Training"]["Accuracy"],
                           values["Training"]["Recall"], values["Training"]["Precision"]))
            table.add_row(("Validation", values["Validation"]["Loss"], values["Validation"]["Accuracy"],
                           values["Validation"]["Recall"], values["Validation"]["Precision"]))
            table.add_row(("Testing", values["Testing"]["Loss"], values["Testing"]["Accuracy"],
                           values["Testing"]["Recall"], values["Testing"]["Precision"]))
        doc.append(Command('end', 'center'))

    with doc.create(Section('English Recurrent Model')):
        values = avg_results(os.path.join(CLASSIFICATION, SUMMARY, 'eng_model_summary.txt'))
        doc.append(Command('begin', 'center'))
        with doc.create(Tabular('l||r r r r', pos='centered')) as table:
            table.add_row(("Set", "Loss", "Accuracy", "Recall", "Precision"))
            table.add_hline()
            table.add_hline()
            table.add_row(("Training", values["Training"]["Loss"], values["Training"]["Accuracy"],
                           values["Training"]["Recall"], values["Training"]["Precision"]))
            table.add_row(("Validation", values["Validation"]["Loss"], values["Validation"]["Accuracy"],
                           values["Validation"]["Recall"], values["Validation"]["Precision"]))
            table.add_row(("Testing", values["Testing"]["Loss"], values["Testing"]["Accuracy"],
                           values["Testing"]["Recall"], values["Testing"]["Precision"]))
        doc.append(Command('end', 'center'))

    with doc.create(Section('Eng-Jpn Model')):
        values = avg_results(os.path.join(CLASSIFICATION, SUMMARY, 'eng_jpn_model_summary.txt'))
        doc.append(Command('begin', 'center'))
        with doc.create(Tabular('l||r r r r', pos='centered')) as table:
            table.add_row(("Set", "Loss", "Accuracy", "Recall", "Precision"))
            table.add_hline()
            table.add_hline()
            table.add_row(("Training", values["Training"]["Loss"], values["Training"]["Accuracy"],
                           values["Training"]["Recall"], values["Training"]["Precision"]))
            table.add_row(("Validation", values["Validation"]["Loss"], values["Validation"]["Accuracy"],
                           values["Validation"]["Recall"], values["Validation"]["Precision"]))
            table.add_row(("Testing", values["Testing"]["Loss"], values["Testing"]["Accuracy"],
                           values["Testing"]["Recall"], values["Testing"]["Precision"]))
        doc.append(Command('end', 'center'))

    with doc.create(Section('Eng-Ita Model')):
        values = avg_results(os.path.join(CLASSIFICATION, SUMMARY, 'eng_ita_model_summary.txt'))
        doc.append(Command('begin', 'center'))
        with doc.create(Tabular('l||r r r r', pos='centered')) as table:
            table.add_row(("Set", "Loss", "Accuracy", "Recall", "Precision"))
            table.add_hline()
            table.add_hline()
            table.add_row(("Training", values["Training"]["Loss"], values["Training"]["Accuracy"],
                           values["Training"]["Recall"], values["Training"]["Precision"]))
            table.add_row(("Validation", values["Validation"]["Loss"], values["Validation"]["Accuracy"],
                           values["Validation"]["Recall"], values["Validation"]["Precision"]))
            table.add_row(("Testing", values["Testing"]["Loss"], values["Testing"]["Accuracy"],
                           values["Testing"]["Recall"], values["Testing"]["Precision"]))
        doc.append(Command('end', 'center'))

    doc.generate_pdf(os.path.join(CLASSIFICATION, SUMMARY, 'models_comparison'), clean_tex=False)


def avg_results(filename):
    values = {"Training": {"Loss": list(), "Accuracy": list(), "Recall": list(), "Precision": list()},
              "Validation": {"Loss": list(), "Accuracy": list(), "Recall": list(), "Precision": list()},
              "Testing": {"Loss": list(), "Accuracy": list(), "Recall": list(), "Precision": list()}}
    with open(filename, mode='r') as file:
        for row in file:
            row = row.split()
            if len(row) < 5:
                continue
            if row[0] == "Training" or row[0] == "Validation" or row[0] == "Testing":
                values[row[0]]["Loss"].append(float(row[1].strip()))
                values[row[0]]["Accuracy"].append(float(row[2].strip()))
                values[row[0]]["Recall"].append(float(row[3].strip()))
                values[row[0]]["Precision"].append(float(row[4].strip()))

    for collection, metrics in values.items():
        for metric, measures in metrics.items():
            values[collection][metric] = np.around(np.mean(measures), decimals=4)

    return values


def plot_history(h, filename):
    plt.style.use('ggplot')
    acc = h.history['accuracy']
    val_acc = h.history['val_accuracy']
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    ics = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ics, acc, 'b', label='Training acc')
    plt.plot(ics, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(ics, loss, 'b', label='Training loss')
    plt.plot(ics, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(filename)


def evaluate_model(model_name, outstream, x, y, v_x, v_y, t_x, t_y):
    model = load_model(os.path.join(CLASSIFICATION, MODELS, model_name + '.h5'),
                       custom_objects={'Recall': Recall, 'Precision': Precision})
    loss, accuracy, recall, precision = model.evaluate(x, y, verbose=False)
    v_loss, v_accuracy, v_recall, v_precision = model.evaluate(v_x, v_y, verbose=False)
    t_loss, t_accuracy, t_recall, t_precision = model.evaluate(t_x, t_y, verbose=False)
    outstream.write("______________________________________________________\n")
    outstream.write(tabulate([['Training', loss, accuracy, recall, precision],
                              ['Validation', v_loss, v_accuracy, v_recall, v_precision],
                              ['Testing', t_loss, t_accuracy, t_recall, t_precision]],
                             headers=['Set', 'Loss', 'Accuracy', 'Recall', 'Precision']))
    outstream.write("\n=======================================================\n\n")


def train_model(model, x, y, v_x, v_y):
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', Recall(name='recall'), Precision(name='precision')])
    model.summary()
    model_name = str(model.name)
    history = model.fit(x, y, epochs=100,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=15),
                            ModelCheckpoint(os.path.join(CLASSIFICATION, MODELS, model_name + '.h5'),
                                            monitor='val_loss', save_best_only=True)],
                        validation_data=(v_x, v_y), verbose=1)
    keras.utils.plot_model(model, os.path.join(CLASSIFICATION, GRAPH, model_name + "_graph.png"), show_shapes=True)
    plot_history(history, os.path.join(CLASSIFICATION, HISTORY, model_name + "_history.png"))


def main():
    # ==================================================================================================================
    # PREPARING THE TOKENIZERS

    if not os.path.exists(os.path.join(CLASSIFICATION, TOKENIZERS, ENG_TOKENIZER)):
        print("English tokenizer not found.")
        sentences = [elem for key, elem in tatoeba.read_sentences(ENG_SENT).items()]
        eng_tokenizer = Tokenizer()
        print("Building the tokenizer...", end=" ")
        eng_tokenizer.fit_on_texts(sentences)
        del sentences
        print("Done.")
        with open(os.path.join(CLASSIFICATION, TOKENIZERS, ENG_TOKENIZER), 'wb') as handle:
            pickle.dump(eng_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved to " + os.path.join(CLASSIFICATION, TOKENIZERS, ENG_TOKENIZER))
    else:
        with open(os.path.join(CLASSIFICATION, TOKENIZERS, ENG_TOKENIZER), 'rb') as handle:
            print("Loading eng tokenizer...", end=" ")
            eng_tokenizer = pickle.load(handle)
            print("Done.")

    if not os.path.exists(os.path.join(CLASSIFICATION, TOKENIZERS, JPN_TOKENIZER)):
        print("Japanese tokenizer not found.")
        sentences = [elem for key, elem in tatoeba.read_sentences(JPN_SENT).items()]
        jpn_tokenizer = JapaneseTokenizer()
        print("Building the tokenizer...", end=" ")
        jpn_tokenizer.fit_on_texts(sentences)
        del sentences
        print("Done.")
        with open(os.path.join(CLASSIFICATION, TOKENIZERS, JPN_TOKENIZER), 'wb') as handle:
            pickle.dump(jpn_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved to " + os.path.join(CLASSIFICATION, TOKENIZERS, JPN_TOKENIZER))
    else:
        with open(os.path.join(CLASSIFICATION, TOKENIZERS, JPN_TOKENIZER), 'rb') as handle:
            print("Loading jpn tokenizer...", end=" ")
            jpn_tokenizer = pickle.load(handle)
            print("Done.")

    if not os.path.exists(os.path.join(CLASSIFICATION, TOKENIZERS, ITA_TOKENIZER)):
        print("Italian tokenizer not found.")
        sentences = [elem for key, elem in tatoeba.read_sentences(ITA_SENT).items()]
        ita_tokenizer = Tokenizer()
        print("Building the tokenizer...", end=" ")
        ita_tokenizer.fit_on_texts(sentences)
        del sentences
        print("Done.")
        with open(os.path.join(CLASSIFICATION, TOKENIZERS, ITA_TOKENIZER), 'wb') as handle:
            pickle.dump(ita_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved to " + os.path.join(CLASSIFICATION, TOKENIZERS, ITA_TOKENIZER))
    else:
        with open(os.path.join(CLASSIFICATION, TOKENIZERS, ITA_TOKENIZER), 'rb') as handle:
            print("Loading ita tokenizer...", end=" ")
            ita_tokenizer = pickle.load(handle)
            print("Done.")

    eng_word_index = eng_tokenizer.word_index
    jpn_word_index = jpn_tokenizer.word_index
    ita_word_index = ita_tokenizer.word_index

    # ==================================================================================================================
    # PREPARING THE EMBEDDINGS
    # Next, we compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings:
    print("Loading eng embeddings...", end=" ")
    eng_embeddings_index = fasttext.load_model('fastText/cc.en.300.bin')

    # At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:
    # Matrice in cui ad ogni indice corrisponde l'embedding vector del token associato a quell'indice
    eng_embedding_matrix = np.zeros((len(eng_word_index) + 1, EMBEDDING_DIM))
    for word, i in eng_word_index.items():
        embedding_vector = eng_embeddings_index.get_word_vector(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            eng_embedding_matrix[i] = embedding_vector
    del eng_embeddings_index
    print("Done")

    print("Loading jpn embeddings...", end=" ")
    jpn_embeddings_index = fasttext.load_model('fastText/cc.ja.300.bin')
    jpn_embedding_matrix = np.zeros((len(jpn_word_index) + 1, EMBEDDING_DIM))
    for word, i in jpn_word_index.items():
        embedding_vector = jpn_embeddings_index.get_word_vector(word)
        if embedding_vector is not None:
            jpn_embedding_matrix[i] = embedding_vector
    del jpn_embeddings_index
    print("Done")

    print("Loading ita embeddings...", end=" ")
    ita_embeddings_index = fasttext.load_model('fastText/cc.it.300.bin')
    ita_embedding_matrix = np.zeros((len(ita_word_index) + 1, EMBEDDING_DIM))
    for word, i in ita_word_index.items():
        embedding_vector = ita_embeddings_index.get_word_vector(word)
        if embedding_vector is not None:
            ita_embedding_matrix[i] = embedding_vector
    del ita_embeddings_index
    print("Done")

    # ==================================================================================================================
    # PREPARING THE FEATURES

    print("Loading the dataset...", end=" ")
    train, valid, test = tatoeba.generate_dataset()

    train_x_eng = eng_tokenizer.texts_to_sequences([sample[2] for sample in train])
    train_x_jpn = jpn_tokenizer.texts_to_sequences([sample[3] for sample in train])
    train_x_ita = ita_tokenizer.texts_to_sequences([sample[4] for sample in train])

    valid_x_eng = eng_tokenizer.texts_to_sequences([sample[2] for sample in valid])
    valid_x_jpn = jpn_tokenizer.texts_to_sequences([sample[3] for sample in valid])
    valid_x_ita = ita_tokenizer.texts_to_sequences([sample[4] for sample in valid])

    test_x_eng = eng_tokenizer.texts_to_sequences([sample[2] for sample in test])
    test_x_jpn = jpn_tokenizer.texts_to_sequences([sample[3] for sample in test])
    test_x_ita = ita_tokenizer.texts_to_sequences([sample[4] for sample in test])

    ENG_SEQ_LEN = 0
    for sent in train_x_eng:
        ENG_SEQ_LEN = max(len(sent), ENG_SEQ_LEN)

    JPN_SEQ_LEN = 0
    for sent in train_x_jpn:
        JPN_SEQ_LEN = max(len(sent), JPN_SEQ_LEN)

    ITA_SEQ_LEN = 0
    for sent in train_x_ita:
        ITA_SEQ_LEN = max(len(sent), ITA_SEQ_LEN)

    train_x_eng = pad_sequences(train_x_eng, maxlen=ENG_SEQ_LEN)
    train_x_jpn = pad_sequences(train_x_jpn, maxlen=JPN_SEQ_LEN)
    train_x_ita = pad_sequences(train_x_ita, maxlen=ITA_SEQ_LEN)
    valid_x_eng = pad_sequences(valid_x_eng, maxlen=ENG_SEQ_LEN)
    valid_x_jpn = pad_sequences(valid_x_jpn, maxlen=JPN_SEQ_LEN)
    valid_x_ita = pad_sequences(valid_x_ita, maxlen=ITA_SEQ_LEN)
    test_x_eng = pad_sequences(test_x_eng, maxlen=ENG_SEQ_LEN)
    test_x_jpn = pad_sequences(test_x_jpn, maxlen=JPN_SEQ_LEN)
    test_x_ita = pad_sequences(test_x_ita, maxlen=ITA_SEQ_LEN)
    del eng_tokenizer
    del jpn_tokenizer
    del ita_tokenizer
    print("Done.")

    # ==================================================================================================================
    # PREPARING THE LABELS

    print("One-hot encoding the labels...", end=" ")
    with open(os.path.join(TATOEBA, BEST_TAGS), mode='r') as file:
        labels = [line.split('\t')[0] for line in file]

    # Empty arrays to hold labels as one hot encodings
    train_y = np.zeros((len(train), NUM_LABELS), dtype=np.int8)
    valid_y = np.zeros((len(valid), NUM_LABELS), dtype=np.int8)
    test_y = np.zeros((len(test), NUM_LABELS), dtype=np.int8)

    for sample_index, sample in enumerate(train):
        train_y[sample_index, labels.index(sample[0])] = 1
    for sample_index, sample in enumerate(valid):
        valid_y[sample_index, labels.index(sample[0])] = 1
    for sample_index, sample in enumerate(test):
        test_y[sample_index, labels.index(sample[0])] = 1
    print("Done.")

    # ==================================================================================================================
    # CREATING THE MODELS
    # We load this embedding matrix into an Embedding layer. Note that we set trainable=False to prevent the weights
    # from being updated during training.

    # ENGLISH INPUT - EMBEDDING
    eng_input = layers.Input(shape=(ENG_SEQ_LEN,), dtype='int32', name='eng_input')
    eng_embedding_layer = layers.Embedding(len(eng_word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[eng_embedding_matrix],
                                           input_length=ENG_SEQ_LEN,
                                           trainable=False,
                                           name='eng_embedding')
    eng_mask = eng_embedding_layer.compute_mask(eng_input)
    eng_embeddings = eng_embedding_layer(eng_input)
    eng_recurrent = layers.GRU(EMBEDDING_DIM, name='eng_recurrent')(eng_embeddings, mask=eng_mask)

    # JAPANESE INPUT - EMBEDDING
    jpn_input = layers.Input(shape=(JPN_SEQ_LEN,), dtype='int32', name='jpn_input')
    jpn_embedding_layer = layers.Embedding(len(jpn_word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[jpn_embedding_matrix],
                                           input_length=JPN_SEQ_LEN,
                                           trainable=False,
                                           name='jpn_embedding')
    jpn_mask = jpn_embedding_layer.compute_mask(jpn_input)
    jpn_embeddings = jpn_embedding_layer(jpn_input)
    jpn_recurrent = layers.GRU(EMBEDDING_DIM, name='jpn_recurrent')(jpn_embeddings, mask=jpn_mask)

    # ITALIAN INPUT - EMBEDDING
    ita_input = layers.Input(shape=(ITA_SEQ_LEN,), dtype='int32', name='ita_input')
    ita_embedding_layer = layers.Embedding(len(ita_word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[ita_embedding_matrix],
                                           input_length=ITA_SEQ_LEN,
                                           trainable=False,
                                           name='ita_embedding')
    ita_mask = ita_embedding_layer.compute_mask(ita_input)
    ita_embeddings = ita_embedding_layer(ita_input)
    ita_recurrent = layers.GRU(EMBEDDING_DIM, name='ita_recurrent')(ita_embeddings, mask=ita_mask)

    # BASELINE MODEL
    baseline_embedding = eng_embedding_layer(eng_input)
    baseline_mean = layers.Lambda(lambda t: keras.backend.mean(t, axis=1),
                                  output_shape=(EMBEDDING_DIM,), name='average')(baseline_embedding)
    baseline_dropout = layers.Dropout(0.5, name='dropout')(baseline_mean)
    baseline_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(baseline_dropout)
    eng_baseline_model = Model(eng_input, baseline_output, name='baseline_model')
    train_model(eng_baseline_model, train_x_eng, train_y, valid_x_eng, valid_y)
    del eng_baseline_model

    # ENGLISH MODEL
    eng_dropout = layers.Dropout(0.5, name='dropout')(eng_recurrent)
    eng_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(eng_dropout)
    eng_model = Model(eng_input, eng_output, name='eng_model')
    train_model(eng_model, train_x_eng, train_y, valid_x_eng, valid_y)
    del eng_model

    # ENG_JAP MODEL
    comb_merge = layers.Concatenate(name='merge')([eng_recurrent, jpn_recurrent])
    comb_merge = layers.Dense(EMBEDDING_DIM, activation='sigmoid', name='reduce')(comb_merge)
    comb_dropout = layers.Dropout(0.5, name='dropout')(comb_merge)
    comb_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(comb_dropout)
    comb_model = Model([eng_input, jpn_input], comb_output, name='eng_jpn_model')
    train_model(comb_model, [train_x_eng, train_x_jpn], train_y, [valid_x_eng, valid_x_jpn], valid_y)
    del comb_model

    # ENG_ITA MODEL
    comb_merge = layers.Concatenate(name='merge')([eng_recurrent, ita_recurrent])
    comb_merge = layers.Dense(EMBEDDING_DIM, activation='sigmoid', name='reduce')(comb_merge)
    comb_dropout = layers.Dropout(0.5, name='dropout')(comb_merge)
    comb_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(comb_dropout)
    comb_model = Model([eng_input, ita_input], comb_output, name='eng_ita_model')
    train_model(comb_model, [train_x_eng, train_x_ita], train_y, [valid_x_eng, valid_x_ita], valid_y)
    del comb_model

    with open(os.path.join(CLASSIFICATION, SUMMARY, 'baseline_model_summary.txt'), mode='a') as file:
        evaluate_model('baseline_model', file, train_x_eng, train_y, valid_x_eng, valid_y, test_x_eng, test_y)
    with open(os.path.join(CLASSIFICATION, SUMMARY, 'eng_model_summary.txt'), mode='a') as file:
        evaluate_model('eng_model', file, train_x_eng, train_y, valid_x_eng, valid_y, test_x_eng, test_y)
    with open(os.path.join(CLASSIFICATION, SUMMARY, 'eng_jpn_model_summary.txt'), mode='a') as file:
        evaluate_model('eng_jpn_model', file, [train_x_eng, train_x_jpn], train_y, [valid_x_eng, valid_x_jpn], valid_y,
                       [test_x_eng, test_x_jpn], test_y)
    with open(os.path.join(CLASSIFICATION, SUMMARY, 'eng_ita_model_summary.txt'), mode='a') as file:
        evaluate_model('eng_ita_model', file, [train_x_eng, train_x_ita], train_y, [valid_x_eng, valid_x_ita], valid_y,
                       [test_x_eng, test_x_ita], test_y)


if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # for _ in range(5):
    #     main()
    latex_summary()
