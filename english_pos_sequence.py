import string
import nltk
from nltk.tokenize import word_tokenize
import tatoeba
from constants import *

punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""


def pos_seq_similarity(s1, s2):

    s1 = s1.translate(str.maketrans('', '', punctuation))
    s1 = nltk.pos_tag(word_tokenize(s1), tagset='universal')
    s2 = s2.translate(str.maketrans('', '', punctuation))
    s2 = nltk.pos_tag(word_tokenize(s2), tagset='universal')
    maxlen = max(len(s1), len(s2))
    count = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i][1] == s2[i][1]:
            count += 1

    return count / maxlen


if __name__ == '__main__':

    sents = tatoeba.read_sentences(ENG_SENT)

    sent = sents['1284']
    print(sent)
    for key, item in sents.items():

        similarity = pos_seq_similarity(sent, item)

        if similarity > 0.8:
            print(item, similarity)
