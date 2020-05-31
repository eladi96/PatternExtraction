import keras
from keras.layers import Embedding, GRU, Input, Dense, Concatenate, Dropout, LSTM, Conv1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from tensorflow.python.keras.models import load_model

import tatoeba
import os.path as path
import fasttext
import pickle
import os
from constants import *
from models.utils import plot_history


def use_model(model, outstream, x, y, v_x, v_y, t_x, test_y):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    keras.utils.plot_model(model, os.path.join(MODELS_DIR, str(model.name + "_graph.png")), show_shapes=True)
    callbacks = [EarlyStopping(monitor='val_loss', patience=15),
                 ModelCheckpoint(os.path.join(MODELS_DIR, str(model.name + ".h5")), save_best_only=True,
                                 save_weights_only=False)]
    history = model.fit(x, y, epochs=100, callbacks=callbacks,
                        validation_data=(v_x, v_y))
    plot_history(history, os.path.join(MODELS_DIR, str(model.name + "_history.png")))
    model.summary(print_fn=lambda var: outstream.write(var + '\n'))
    _, accuracy = model.evaluate(x, y, verbose=False)
    outstream.write("Training Accuracy:  {:.4f}\n".format(accuracy))
    _, accuracy = model.evaluate(v_x, v_y, verbose=False)
    outstream.write("Validation Accuracy:  {:.4f}\n".format(accuracy))
    _, accuracy = model.evaluate(t_x, test_y, verbose=False)
    outstream.write("Testing Accuracy:  {:.4f}\n".format(accuracy))
    outstream.write("#============================================================================================\n\n")


def main():
    # ==================================================================================================================
    # PREPARING THE TOKENIZERS

    with open(os.path.join(MODELS_DIR, ENG_TOKENIZER), 'rb') as handle:
        print("Loading eng tokenizer...", end=" ")
        eng_tokenizer = pickle.load(handle)
        print("Done.")

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
    eng_embeddings_index = None
    print("Done")

    print("Loading jpn embeddings...", end=" ")
    jpn_embeddings_index = fasttext.load_model('fastText/cc.ja.300.bin')
    jpn_embedding_matrix = np.zeros((len(jpn_word_index) + 1, EMBEDDING_DIM))
    for word, i in jpn_word_index.items():
        embedding_vector = jpn_embeddings_index.get_word_vector(word)
        if embedding_vector is not None:
            jpn_embedding_matrix[i] = embedding_vector
    jpn_embeddings_index = None
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
        ENG_SEQ_LEN = len(sent) if len(sent) > ENG_SEQ_LEN else ENG_SEQ_LEN

    JPN_SEQ_LEN = 0
    for sent in train_x_jpn:
        JPN_SEQ_LEN = len(sent) if len(sent) > JPN_SEQ_LEN else JPN_SEQ_LEN

    train_x_eng = pad_sequences(train_x_eng, maxlen=ENG_SEQ_LEN)
    train_x_jpn = pad_sequences(train_x_jpn, maxlen=JPN_SEQ_LEN)
    valid_x_eng = pad_sequences(valid_x_eng, maxlen=ENG_SEQ_LEN)
    valid_x_jpn = pad_sequences(valid_x_jpn, maxlen=JPN_SEQ_LEN)
    test_x_eng = pad_sequences(test_x_eng, maxlen=ENG_SEQ_LEN)
    test_x_jpn = pad_sequences(test_x_jpn, maxlen=JPN_SEQ_LEN)
    print("Done.")

    # ==================================================================================================================
    # PREPARING THE LABELS

    print("One-hot encoding the labels...", end=" ")
    labels = list(set([sample[0] for sample in train]))

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
    eng_input = Input(shape=(ENG_SEQ_LEN,), dtype='int32', name='eng_input')
    eng_embedding_layer = Embedding(len(eng_word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[eng_embedding_matrix],
                                    input_length=ENG_SEQ_LEN,
                                    trainable=False,
                                    name='eng_embedding')
    eng_mask = eng_embedding_layer.compute_mask(eng_input)

    # JAPANESE INPUT - EMBEDDING
    jpn_input = Input(shape=(JPN_SEQ_LEN,), dtype='int32', name='jpn_input')
    jpn_embedding_layer = Embedding(len(jpn_word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[jpn_embedding_matrix],
                                    input_length=JPN_SEQ_LEN,
                                    trainable=False,
                                    name='jpn_embedding')
    jpn_mask = jpn_embedding_layer.compute_mask(jpn_input)

    with open(path.join(MODELS_DIR, 'multilingual_models_summary.txt'), mode='w') as file:

        # ENGLISH MODEL
        eng_embeddings = eng_embedding_layer(eng_input)
        eng_recurrent = GRU(EMBEDDING_DIM, name='eng_recurrent')(eng_embeddings, mask=eng_mask)
        eng_dropout = Dropout(0.5, name='dropout')(eng_recurrent)
        eng_output = Dense(NUM_LABELS, activation='softmax', name='classification')(eng_dropout)
        eng_model = Model(eng_input, eng_output, name='eng_model')
        use_model(eng_model, file, train_x_eng, train_y, valid_x_eng, valid_y, test_x_eng, test_y)

        # JAPANESE MODEL
        jpn_embeddings = jpn_embedding_layer(jpn_input)
        jpn_recurrent = GRU(EMBEDDING_DIM, name='jpn_recurrent')(jpn_embeddings, mask=jpn_mask)
        jpn_dropout = Dropout(0.5, name='dropout')(jpn_recurrent)
        jpn_output = Dense(NUM_LABELS, activation='softmax', name='classification')(jpn_dropout)
        jpn_model = Model(jpn_input, jpn_output, name='jpn_model')
        use_model(jpn_model, file, train_x_jpn, train_y, valid_x_jpn, valid_y, test_x_jpn, test_y)

        # COMBINED MODEL
        merged = Concatenate(name='merging_layer')([eng_recurrent, jpn_recurrent])
        combined_dropout = Dropout(0.5, name='dropout')(merged)
        combined_output = Dense(NUM_LABELS, activation='softmax', name='classification')(combined_dropout)
        combined_model = Model([eng_input, jpn_input], combined_output, name='combined_model')
        use_model(combined_model, file, [train_x_eng, train_x_jpn], train_y, [valid_x_eng, valid_x_jpn], valid_y,
                  [test_x_eng, test_x_jpn], test_y)


        # PRETRAINED MODEL
        pretrained = Model([eng_input, jpn_input], combined_output, name='pretrained_model')
        pretrained.get_layer('eng_recurrent').set_weights(eng_model.get_layer('eng_recurrent').get_weights())
        pretrained.get_layer('eng_recurrent').trainable = False
        pretrained.get_layer('jpn_recurrent').set_weights(jpn_model.get_layer('jpn_recurrent').get_weights())
        pretrained.get_layer('jpn_recurrent').trainable = False
        use_model(pretrained, file, [train_x_eng, train_x_jpn], train_y, [valid_x_eng, valid_x_jpn], valid_y,
                  [test_x_eng, test_x_jpn], test_y)


if __name__ == '__main__':
    main()
