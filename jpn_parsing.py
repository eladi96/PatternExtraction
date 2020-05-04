import Mykytea
from enum import Enum


class Pos2En(Enum):
    # This is a mapping between Kytea's japanese pos and their english translations
    # retrieved from Neubig's github
    # https://gist.github.com/neubig/2555399
    N = "名詞"  # Noun
    PRP = "代名詞"  # Pronoun
    DT = "連体詞"  # Adjectival determiner
    V = "動詞"  # Verb
    ADJ = "形容詞"  # Adjective
    ADJV = "形状詞"  # Adjectival verb
    ADV = "副詞"  # Adverb
    PRT = "助詞"  # Particle
    AUXV = "助動詞"  # Auxiliary verb
    PUNCT = "補助記号"  # Punctuation
    SYM = "記号"  # Symbol
    SUF = "接尾辞"  # Suffix
    PRE = "接頭辞"  # Prefix
    TAIL = "語尾"  # Word tail (conjugation)
    CC = "接続詞"  # Conjunction
    URL = "URL"  # URL
    ENG = "英単語"  # English word
    FIL = "言いよどみ"  # Filler
    MSP = "web誤脱"  # Misspelling
    INT = "感動詞"  # Interjection
    UNK = "新規未知語"  # Unclassified unknown word


class JapaneseTokenizer:

    def __init__(self):
        # You can pass arguments KyTea style like following
        opt = "-tagmax 1"
        # You can also set your own model
        # opt = "-model kytea-0.4.7/data/model.bin"
        self.mk = Mykytea.Mykytea(opt)

    def tokenize(self, sent):
        return [word for word in self.mk.getWS(sent)]

    def pos_tags(self, sent):
        def get_pos(tags):
            return Pos2En(tags[0][0][0]).name

        return [(word.surface, get_pos(word.tag)) for word in self.mk.getTags(sent)]


if __name__ == '__main__':

    s = "私は山にいました"
    jptk = JapaneseTokenizer()
    print(jptk.tokenize(s))
    print(jptk.pos_tags(s))
