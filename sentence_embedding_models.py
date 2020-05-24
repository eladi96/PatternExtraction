from keras.layers import Embedding, GRU, Input, Dense, Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tatoeba
import matplotlib.pyplot as plt
import os.path as path
import fasttext
import pickle
import os

MODELS_DIR = 'models'
TOKENIZER = 'tokenizer.pickle'


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


def use_model(model, chekpoint, summary, plot):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(path.join(MODELS_DIR, chekpoint), save_best_only=True,
                                 save_weights_only=False)]
    history = model.fit(train_x, train_y, epochs=100, callbacks=callbacks, validation_data=(validation_x, validation_y))
    with open(path.join(MODELS_DIR, summary), mode='w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda var: file.write(var + '\n'))
        _, accuracy = model.evaluate(train_x, train_y, verbose=False)
        file.write("Training Accuracy:  {:.4f}\n".format(accuracy))
        _, accuracy = model.evaluate(validation_x, validation_y, verbose=False)
        file.write("Validation Accuracy:  {:.4f}\n".format(accuracy))
        _, accuracy = model.evaluate(test_x, test_y, verbose=False)
        file.write("Testing Accuracy:  {:.4f}\n".format(accuracy))
        plot_history(history, path.join(MODELS_DIR, plot))


if __name__ == '__main__':

    # ==================================================================================================================
    # PREPARING THE TOKENIZER

    if not os.path.exists(os.path.join(MODELS_DIR, TOKENIZER)):
        print("Tokenizer not found.")
        sentences = [elem for key, elem in tatoeba.read_sentences(tatoeba.ENG_SENT).items()]
        tokenizer = Tokenizer()
        print("Building the tokenizer...", end=" ")
        tokenizer.fit_on_texts(sentences)
        sentences = None
        print("Done.")
        with open(os.path.join(MODELS_DIR, TOKENIZER), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved to " + os.path.join(MODELS_DIR, TOKENIZER))
    else:
        with open(os.path.join(MODELS_DIR, TOKENIZER), 'rb') as handle:
            print("Loading tokenizer...", end=" ")
            tokenizer = pickle.load(handle)
            print("Done.")

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # ==================================================================================================================
    # PREPARING THE EMBEDDINGS
    # Next, we compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings:
    print("Loading embeddings...", end=" ")
    embeddings_index = fasttext.load_model('fasttext/cc.en.300.bin')
    print("Done")

    EMBEDDING_DIM = 300
    # At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:
    # Matrice in cui ad ogni indice corrisponde l'embedding vector del token associato a quell'indice
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get_word_vector(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    embeddings_index = None

    # ==================================================================================================================
    # PREPARING THE FEATURES

    print("Loading the dataset...", end=" ")
    train, validation, test = tatoeba.generate_dataset()
    train_x = [sample[3] for sample in train]
    validation_x = [sample[3] for sample in validation]
    test_x = [sample[3] for sample in test]

    # Then we can format our text samples and labels into tensors that can be fed into a neural network. To do this,
    # we will rely on Keras utilities keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences.
    train_x = tokenizer.texts_to_sequences(train_x)
    validation_x = tokenizer.texts_to_sequences(validation_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    tokenizer = None

    MAX_SEQUENCE_LENGTH = 0
    for sent in train_x:
        MAX_SEQUENCE_LENGTH = len(sent) if len(sent) > MAX_SEQUENCE_LENGTH else MAX_SEQUENCE_LENGTH
    train_x = pad_sequences(train_x, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    validation_x = pad_sequences(validation_x, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    test_x = pad_sequences(test_x, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    print("Done.")

    # ==================================================================================================================
    # PREPARING THE LABELS

    print("One-hot encoding the labels...", end=" ")
    labels = list(set([sample[0] for sample in train]))
    NUM_LABELS = len(labels)

    # Empty arrays to hold labels as one hot encodings
    train_y = np.zeros((len(train_x), NUM_LABELS), dtype=np.int8)
    validation_y = np.zeros((len(validation_x), NUM_LABELS), dtype=np.int8)
    test_y = np.zeros((len(test_x), NUM_LABELS), dtype=np.int8)

    for sample_index, sample in enumerate(train):
        train_y[sample_index, labels.index(sample[0])] = 1
    for sample_index, sample in enumerate(validation):
        validation_y[sample_index, labels.index(sample[0])] = 1
    for sample_index, sample in enumerate(test):
        test_y[sample_index, labels.index(sample[0])] = 1
    print("Done.")

    # ==================================================================================================================
    # CREATING THE MODEL
    # We load this embedding matrix into an Embedding layer. Note that we set trainable=False to prevent the weights
    # from being updated during training.

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    embedded_sequences = embedding_layer(sequence_input)
    # Mask for the layers that can handle it
    mask = embedding_layer.compute_mask(sequence_input)
    preds = Dense(NUM_LABELS, activation='softmax')

    # MODEL 1 - Conv1D
    x = Conv1D(EMBEDDING_DIM, 3, activation='relu')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)
    output = preds(x)
    model_1 = Model(sequence_input, output, name='model_1')
    use_model(model_1, 'model_1.h5', 'model_1_summary.txt', 'model_1_history.png')

    # MODEL 2 - LSTM
    x = LSTM(EMBEDDING_DIM)(embedded_sequences, mask=mask)
    output = preds(x)
    model_2 = Model(sequence_input, output, name='model_2')
    use_model(model_2, 'model_2.h5', 'model_2_summary.txt', 'model_2_history.png')

    # # MODEL 3 - GRU
    x = GRU(EMBEDDING_DIM)(embedded_sequences, mask=mask)
    output = preds(x)
    model_3 = Model(sequence_input, output, name='model_3')
    use_model(model_3, 'model_3.h5', 'model_3_summary.txt', 'model_3_history.png')

    # MODEL 4 - Bidirectional with GRU - CONCAT - DENSE
    x = Bidirectional(GRU(EMBEDDING_DIM))(embedded_sequences, mask=mask)
    x = Dense(300, activation='relu')(x)
    output = preds(x)
    model_4 = Model(sequence_input, output, name='model_4')
    use_model(model_4, 'model_4.h5', 'model_4_summary.txt', 'model_4_history.png')

    # MODEL 5 - Bidirection with GRU - AVG
    x = Bidirectional(GRU(EMBEDDING_DIM), merge_mode='ave')(embedded_sequences, mask=mask)
    output = preds(x)
    model_5 = Model(sequence_input, output, name='model_5')
    use_model(model_5, 'model_5.h5', 'model_5_summary.txt', 'model_5_history.png')
