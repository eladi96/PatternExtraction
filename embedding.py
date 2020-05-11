from keras.layers import Embedding, GRU, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv
import time


def reduce_embedding():
    with open('fasttext/cc.en-reduced.300.vec', mode='w') as out:
        f = open('fasttext/cc.en.300.vec')
        for count, line in enumerate(f):
            out.write(line)
            if count > 100000:
                break
        f.close()


if __name__ == '__main__':

    # List of text samples
    with open("tatoeba/eng/en_2000.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        eng_sentences = [row[2] for row in reader]
        print("Read sentences.")

    # Then we can format our text samples and labels into tensors that can be fed into a neural network. To do this,
    # we will rely on Keras utilities keras.preprocessing.text.Tokenizer and keras.preprocessing.sequence.pad_sequences.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(eng_sentences)
    sequences = tokenizer.texts_to_sequences(eng_sentences)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = 0
    for sent in eng_sentences:
        MAX_SEQUENCE_LENGTH = len(sent) if len(sent) > MAX_SEQUENCE_LENGTH else MAX_SEQUENCE_LENGTH
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data tensor:', data.shape)

    # Next, we compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings:
    embeddings_index = {}
    f = open('fasttext/cc.en-reduced.300.vec')
    start = time.time()
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    end = time.time()
    print('Found %s word vectors in %s seconds.' % (len(embeddings_index), str(end - start)))

    EMBEDDING_DIM = 300
    # At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:
    # Matrice in cui ad ogni indice corrisponde l'embedding vector del token associato a quell'indice
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # We load this embedding matrix into an Embedding layer. Note that we set trainable=False to prevent the weights
    # from being updated during training.
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    output_layer = GRU(units=EMBEDDING_DIM)(embedded_sequences)
    model = Model(sequence_input, output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    start = time.time()
    test = model.predict(data)
    end = time.time()
    print(test.shape)
    print("2000 sent in %s seconds." % str(end - start))
