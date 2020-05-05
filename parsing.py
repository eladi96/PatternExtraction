import Mykytea
import MeCab
from polyglot.text import Text
from enum import Enum


class Pos2En(Enum):
    """
    Polyglot recognizes 17 parts of speech, this set is called the universal part of speech tag set:

    ADJ: adjective
    ADP: adposition
    ADV: adverb
    AUX: auxiliary verb
    CONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other

    However, Kytea uses a different set of tags:
    "名詞" => "N", # Noun
    "代名詞" => "PRP", # Pronoun
    "連体詞" => "DT", # Adjectival determiner
    "動詞" => "V", # Verb
    "形容詞" => "ADJ", # Adjective
    "形状詞" => "ADJV", # Adjectival verb
    "副詞" => "ADV", # Adverb
    "助詞" => "PRT", # Particle
    "助動詞" => "AUXV", # Auxiliary verb
    "補助記号" => ".", # Punctuation
    "記号" => "SYM", # Symbol
    "接尾辞" => "SUF", # Suffix
    "接頭辞" => "PRE", # Prefix
    "語尾" => "TAIL", # Word tail (conjugation)
    "接続詞" => "CC", # Conjunction
    "代名詞" => "PRP", # Pronoun
    "URL" => "URL", # URL
    "英単語" => "ENG", # English word
    "言いよどみ" => "FIL", # Filler
    "web誤脱" => "MSP", # Misspelling
    "感動詞" => "INT", # Interjection
    "新規未知語" => "UNK", # Unclassified unknown word

    This is a mapping between Kytea's japanese pos and their english translations
    retrieved from Neubig's github
    https://gist.github.com/neubig/2555399

    I'm trying to use consistent names for the same parts of speech
    """
    ADJ = "形容詞"  # Adjective
    # missing Adposition
    ADV = "副詞"  # Adverb
    AUX = "助動詞"  # Auxiliary verb
    CONJ = "接続詞"  # Conjunction
    DET = "連体詞"  # Adjectival determiner
    INTJ = "感動詞"  # Interjection
    NOUN = "名詞"  # Noun
    # missing Numeral
    PART = "助詞"  # Particle
    PRON = "代名詞"  # Pronoun
    # missing Proper Noun
    PUNCT = "補助記号"  # Punctuation
    # missing Subordinating conjunction
    SYM = "記号"  # Symbol
    VERB = "動詞"  # Verb
    X = "新規未知語"  # Unclassified unknown word

    # Theese tags are only present in the japanese parser
    ADJV = "形状詞"  # Adjectival verb
    SUF = "接尾辞"  # Suffix
    PRE = "接頭辞"  # Prefix
    TAIL = "語尾"  # Word tail (conjugation)
    URL = "URL"  # URL
    ENG = "英単語"  # English word
    FIL = "言いよどみ"  # Filler
    MSP = "web誤脱"  # Misspelling


class JapaneseTokenizer:

    def __init__(self):
        # You can pass arguments KyTea style like following
        opt = "-tagmax 1"
        # You can also set your own model
        # opt = "-model kytea-0.4.7/data/model.bin"
        self.kytea = Mykytea.Mykytea(opt)
        self.mecab = MeCab.Tagger('')

    def tokenize(self, sent):
        return [word for word in self.kytea.getWS(sent)]

    def pos_tags_kytea(self, sent):
        def get_pos(tags):
            return Pos2En(tags[0][0][0]).name

        return [(word.surface, get_pos(word.tag)) for word in self.kytea.getTags(sent)]

    def pos_tags_mecab(self, sent):
        def get_word(chunk):
            return chunk.split('\t')[0]

        def get_pos(chunk):
            return Pos2En(chunk.split('\t')[1].split(',')[0]).name

        return [(get_word(chunk), get_pos(chunk)) for chunk in
                self.mecab.parse(sent).splitlines()[:-1]]


class ItalianTokenizer:

    @staticmethod
    def tokenize(sent):
        return Text(sent, hint_language_code='it').words

    # After calling the pos_tags property once,
    # the words objects will carry the POS tags.
    # print(text.words[0].pos_tag)
    @staticmethod
    def pos_tags(sent):
        return Text(sent, hint_language_code='it').pos_tags


class EnglishTokenizer:

    @staticmethod
    def tokenize(sent):
        return Text(sent, hint_language_code='en').words

    # After calling the pos_tags property once,
    # the words objects will carry the POS tags.
    # print(text.words[0].pos_tag)
    @staticmethod
    def pos_tags(sent):
        return Text(sent, hint_language_code='en').pos_tags


if __name__ == '__main__':
    jptk = JapaneseTokenizer()
    ittk = ItalianTokenizer()
    entk = EnglishTokenizer()

    print(jptk.pos_tags_kytea("彼は時々変です"))
    print("________________________________________________________")
    print(jptk.pos_tags_mecab("彼は時々変です"))
    print("________________________________________________________")
    print(entk.pos_tags("Sometimes he can be a strange guy"))
    print("________________________________________________________")
    print(ittk.pos_tags("A volte può essere un tizio strano"))
