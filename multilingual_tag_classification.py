import tatoeba
import os
import fasttext
import pickle
import numpy as np
import keras
from os.path import join
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Recall, Precision
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tabulate import tabulate
from constants import *
from japanese_tokenizer import JapaneseTokenizer
from models.utils import plot_history


def evaluate_model(model_name, outstream, x, y, v_x, v_y, t_x, t_y):
    model = load_model(join(MODELS_DIR, model_name + '.h5'), custom_objects={'Recall': Recall, 'Precision': Precision})
    # model.summary(print_fn=lambda var: outstream.write(var + '\n'))
    loss, accuracy, recall, precision = model.evaluate(x, y, verbose=False)
    v_loss, v_accuracy, v_recall, v_precision = model.evaluate(v_x, v_y, verbose=False)
    t_loss, t_accuracy, t_recall, t_precision = model.evaluate(t_x, t_y, verbose=False)
    outstream.write("Model: " + model_name + "\n")
    outstream.write("______________________________________________________\n")
    outstream.write(tabulate([['Training', loss, accuracy, recall, precision],
                              ['Validation', v_loss, v_accuracy, v_recall, v_precision],
                              ['Testing', t_loss, t_accuracy, t_recall, t_precision]],
                             headers=['Set', 'Loss', 'Accuracy', 'Recall', 'Precision']))
    outstream.write("\n======================================================\n\n")


def train_model(model, x, y, v_x, v_y):
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', Recall(name='recall'), Precision(name='precision')])
    model.summary()
    model_name = str(model.name)
    history = model.fit(x, y, epochs=100,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=15),
                            ModelCheckpoint(join(MODELS_DIR, model_name + '.h5'),
                                            monitor='val_loss', save_best_only=True)],
                        validation_data=(v_x, v_y), verbose=1)
    keras.utils.plot_model(model, join(MODELS_DIR, "graph", model_name + "_graph.png"), show_shapes=True)
    plot_history(history, join(MODELS_DIR, "history", model_name + "_history.png"))


def main():
    # ==================================================================================================================
    # PREPARING THE TOKENIZERS

    if not os.path.exists(os.path.join(MODELS_DIR, ENG_TOKENIZER)):
        print("English tokenizer not found.")
        sentences = [elem for key, elem in tatoeba.read_sentences(ENG_SENT).items()]
        eng_tokenizer = Tokenizer()
        print("Building the tokenizer...", end=" ")
        eng_tokenizer.fit_on_texts(sentences)
        del sentences
        print("Done.")
        with open(os.path.join(MODELS_DIR, ENG_TOKENIZER), 'wb') as handle:
            pickle.dump(eng_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved to " + os.path.join(MODELS_DIR, ENG_TOKENIZER))
    else:
        with open(os.path.join(MODELS_DIR, ENG_TOKENIZER), 'rb') as handle:
            print("Loading eng tokenizer...", end=" ")
            eng_tokenizer = pickle.load(handle)
            print("Done.")

    if not os.path.exists(os.path.join(MODELS_DIR, JPN_TOKENIZER)):
        print("Japanese tokenizer not found.")
        sentences = [elem for key, elem in tatoeba.read_sentences(JPN_SENT).items()]
        jpn_tokenizer = JapaneseTokenizer()
        print("Building the tokenizer...", end=" ")
        jpn_tokenizer.fit_on_texts(sentences)
        del sentences
        print("Done.")
        with open(os.path.join(MODELS_DIR, JPN_TOKENIZER), 'wb') as handle:
            pickle.dump(jpn_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved to " + os.path.join(MODELS_DIR, JPN_TOKENIZER))
    else:
        with open(os.path.join(MODELS_DIR, JPN_TOKENIZER), 'rb') as handle:
            print("Loading jpn tokenizer...", end=" ")
            jpn_tokenizer = pickle.load(handle)
            print("Done.")

    eng_word_index = eng_tokenizer.word_index
    jpn_word_index = jpn_tokenizer.word_index

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

    # ==================================================================================================================
    # PREPARING THE FEATURES

    print("Loading the dataset...", end=" ")
    train, valid, test = tatoeba.generate_dataset()

    train_x_eng = eng_tokenizer.texts_to_sequences([sample[2] for sample in train])
    train_x_jpn = jpn_tokenizer.texts_to_sequences([sample[3] for sample in train])

    valid_x_eng = eng_tokenizer.texts_to_sequences([sample[2] for sample in valid])
    valid_x_jpn = jpn_tokenizer.texts_to_sequences([sample[3] for sample in valid])

    test_x_eng = eng_tokenizer.texts_to_sequences([sample[2] for sample in test])
    test_x_jpn = jpn_tokenizer.texts_to_sequences([sample[3] for sample in test])

    ENG_SEQ_LEN = 0
    for sent in train_x_eng:
        ENG_SEQ_LEN = max(len(sent), ENG_SEQ_LEN)

    JPN_SEQ_LEN = 0
    for sent in train_x_jpn:
        JPN_SEQ_LEN = max(len(sent), JPN_SEQ_LEN)

    train_x_eng = pad_sequences(train_x_eng, maxlen=ENG_SEQ_LEN)
    train_x_jpn = pad_sequences(train_x_jpn, maxlen=JPN_SEQ_LEN)
    valid_x_eng = pad_sequences(valid_x_eng, maxlen=ENG_SEQ_LEN)
    valid_x_jpn = pad_sequences(valid_x_jpn, maxlen=JPN_SEQ_LEN)
    test_x_eng = pad_sequences(test_x_eng, maxlen=ENG_SEQ_LEN)
    test_x_jpn = pad_sequences(test_x_jpn, maxlen=JPN_SEQ_LEN)
    del eng_tokenizer
    del jpn_tokenizer
    print("Done.")

    # ==================================================================================================================
    # PREPARING THE LABELS

    print("One-hot encoding the labels...", end=" ")
    with open(os.path.join(TATOEBA_PATH, BEST_TAGS), mode='r') as file:
        labels = [line.split(':')[0] for line in file]

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
    eng_recurrent = layers.LSTM(EMBEDDING_DIM, name='eng_recurrent')(eng_embeddings, mask=eng_mask)

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
    jpn_recurrent = layers.LSTM(EMBEDDING_DIM, name='jpn_recurrent')(jpn_embeddings, mask=jpn_mask)

    # BASELINE MODEL
    baseline_embedding = eng_embedding_layer(eng_input)
    baseline_mean = layers.Lambda(lambda t: keras.backend.mean(t, axis=1),
                                  output_shape=(EMBEDDING_DIM,), name='mean')(baseline_embedding)
    baseline_dropout = layers.Dropout(0.5, name='dropout')(baseline_mean)
    baseline_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(baseline_dropout)
    eng_baseline_model = Model(eng_input, baseline_output, name='eng_baseline')
    train_model(eng_baseline_model, train_x_eng, train_y, valid_x_eng, valid_y)
    del eng_baseline_model

    # ENGLISH MODEL
    eng_dropout = layers.Dropout(0.5, name='dropout')(eng_recurrent)
    eng_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(eng_dropout)
    eng_model = Model(eng_input, eng_output, name='eng_model')
    train_model(eng_model, train_x_eng, train_y, valid_x_eng, valid_y)
    del eng_model

    # JAPANESE MODEL
    jpn_dropout = layers.Dropout(0.5, name='dropout')(jpn_recurrent)
    jpn_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(jpn_dropout)
    jpn_model = Model(jpn_input, jpn_output, name='jpn_model')
    train_model(jpn_model, train_x_jpn, train_y, valid_x_jpn, valid_y)
    del jpn_model

    # COMBINED MODEL
    comb_merge = layers.Concatenate(name='merge')([eng_recurrent, jpn_recurrent])
    comb_merge = layers.Dense(EMBEDDING_DIM, activation='relu', name='reduce')(comb_merge)
    comb_dropout = layers.Dropout(0.5, name='dropout')(comb_merge)
    comb_output = layers.Dense(NUM_LABELS, activation='softmax', name='classification')(comb_dropout)
    comb_model = Model([eng_input, jpn_input], comb_output, name='combined_model')
    train_model(comb_model, [train_x_eng, train_x_jpn], train_y, [valid_x_eng, valid_x_jpn], valid_y)
    del comb_model

    with open(join(MODELS_DIR, 'summary.txt'), mode='a') as file:
        file.write(
            "\nTODO==============================================================================================" +
            "======================\n")
        evaluate_model('eng_baseline', file, train_x_eng, train_y, valid_x_eng, valid_y, test_x_eng, test_y)
        evaluate_model('eng_model', file, train_x_eng, train_y, valid_x_eng, valid_y, test_x_eng, test_y)
        evaluate_model('jpn_model', file, train_x_jpn, train_y, valid_x_jpn, valid_y, test_x_jpn, test_y)
        evaluate_model('combined_model', file, [train_x_eng, train_x_jpn], train_y, [valid_x_eng, valid_x_jpn], valid_y,
                       [test_x_eng, test_x_jpn], test_y)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
