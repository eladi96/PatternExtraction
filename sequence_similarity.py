import editdistance
import nltk
from nltk.tokenize import word_tokenize

from japanese_tokenizer import JapaneseTokenizer

punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""


def pos_seq_similarity(s1, s2, lang):

    if lang == 'eng':
        s1 = s1.translate(str.maketrans('', '', punctuation))
        s1 = [elem[1] for elem in nltk.pos_tag(word_tokenize(s1), tagset='universal')]
        s2 = s2.translate(str.maketrans('', '', punctuation))
        s2 = [elem[1] for elem in nltk.pos_tag(word_tokenize(s2), tagset='universal')]
    elif lang == 'jpn':
        tokenizer = JapaneseTokenizer()
        s1 = [elem[1] for elem in tokenizer.pos_tags_mecab(s1)]
        s2 = [elem[1] for elem in tokenizer.pos_tags_mecab(s2)]

    maxlen = max(len(s1), len(s2))
    count = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            count += 1

    return count / maxlen


def trigram_similarity(s1, s2, lang):

    if lang == 'eng':
        s1 = s1.translate(str.maketrans('', '', punctuation))
        s1 = [elem[1] for elem in nltk.pos_tag(word_tokenize(s1), tagset='universal')]
        s2 = s2.translate(str.maketrans('', '', punctuation))
        s2 = [elem[1] for elem in nltk.pos_tag(word_tokenize(s2), tagset='universal')]
    elif lang == 'jpn':
        tokenizer = JapaneseTokenizer()
        s1 = [elem[1] for elem in tokenizer.pos_tags_mecab(s1)]
        s2 = [elem[1] for elem in tokenizer.pos_tags_mecab(s2)]

    s1.append("_")
    s1.insert(0, "_")
    n1 = [tuple(s1[i - 1:i + 2]) for i in range(1, len(s1) - 1)]

    s2.append("_")
    s2.insert(0, "_")
    n2 = [tuple(s2[i - 1:i + 2]) for i in range(1, len(s2) - 1)]

    equals = sum(1 for elem in zip(n1, n2) if elem[0] == elem[1])

    distinct = set(n1)
    distinct.update(n2)
    distinct = len(distinct)

    return equals / distinct


def levenshtein_similairty(s1, s2, lang):

    if lang == 'eng':
        s1 = s1.translate(str.maketrans('', '', punctuation))
        s1 = [elem[1] for elem in nltk.pos_tag(word_tokenize(s1), tagset='universal')]
        s2 = s2.translate(str.maketrans('', '', punctuation))
        s2 = [elem[1] for elem in nltk.pos_tag(word_tokenize(s2), tagset='universal')]
    elif lang == 'jpn':
        tokenizer = JapaneseTokenizer()
        s1 = [elem[1] for elem in tokenizer.pos_tags_mecab(s1)]
        s2 = [elem[1] for elem in tokenizer.pos_tags_mecab(s2)]

    return 1 - (editdistance.eval(s1, s2) / max(len(s1), len(s1)))