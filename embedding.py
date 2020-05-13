from keras.layers import Embedding, GRU, Input, Dense, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import csv
import tatoeba

if __name__ == '__main__':

    # List of text samples
    with open("tatoeba/tagged_sentences.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        sentences = [row[2] for row in reader]
        print("Read sentences.")

    train, validation, test = tatoeba.generate_dataset()

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARING THE FEATURES
    train_x = [sample[2] for sample in train]
    validation_x = [sample[2] for sample in validation]
    test_x = [sample[2] for sample in test]

    # Then we can format our text samples and labels into tensors that can be fed into a neural network. To do this,
    # we will rely on Keras utilities keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    train_x = tokenizer.texts_to_sequences(train_x)
    validation_x = tokenizer.texts_to_sequences(validation_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = 0
    for sent in sentences:
        MAX_SEQUENCE_LENGTH = len(sent) if len(sent) > MAX_SEQUENCE_LENGTH else MAX_SEQUENCE_LENGTH
    train_x = pad_sequences(train_x, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    validation_x = pad_sequences(validation_x, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    test_x = pad_sequences(test_x, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of data tensor:', train_x.shape)

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARING THE LABELS

    print("One-hot encoding the labels...")
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
    print("Labels ready.")

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARING THE EMBEDDINGS
    # Next, we compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings:
    embeddings_index = {}
    f = open('fasttext/cc.en-reduced.300.vec')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    EMBEDDING_DIM = 300
    # At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:
    # Matrice in cui ad ogni indice corrisponde l'embedding vector del token associato a quell'indice
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # ------------------------------------------------------------------------------------------------------------------
    # CREATING THE MODEL
    # We load this embedding matrix into an Embedding layer. Note that we set trainable=False to prevent the weights
    # from being updated during training.
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = GRU(units=EMBEDDING_DIM)(embedded_sequences)
    x = Dense(EMBEDDING_DIM, activation='relu')(x)
    x = Dropout(0.3)(x)
    preds = Dense(NUM_LABELS, activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')
    model.summary()

    # Create callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint('/models/model-1.h5', save_best_only=True,
                                 save_weights_only=False)]

    history = model.fit(train_x, train_y, epochs=5, callbacks=callbacks, validation_data=(validation_x,
                                                                                          validation_y))
    print(model.evaluate(test_x, test_y))
