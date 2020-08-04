"""Tokenizer for Japanese with the same structure of the Keras' tokenizer
"""

import MeCab
from collections import OrderedDict, defaultdict
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
    PRE = "接頭詞"  # Prefix
    TAIL = "語尾"  # Word tail (conjugation)
    URL = "URL"  # URL
    ENG = "英単語"  # English word
    FIL = "言いよどみ"  # Filler
    MSP = "web誤脱"  # Misspelling
    FILLER = "フィラー"


class JapaneseTokenizer:

    def __init__(self,
                 oov_token=None):
        self.mecab = MeCab.Tagger('')
        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.index_docs = defaultdict(int)
        self.word_index = dict()
        self.index_word = dict()
        self.oov_token = oov_token

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['mecab']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.mecab = MeCab.Tagger('')

    def tokenize(self, sent):
        def get_word(chunk):
            return chunk.split('\t')[0]
        return [get_word(chunk) for chunk in self.mecab.parse(sent).splitlines()[:-1] if get_word(chunk) not in "、。"]

    def pos_tags_mecab(self, sent):
        def get_word(chunk):
            return chunk.split('\t')[0]

        def get_pos(chunk):
            return Pos2En(chunk.split('\t')[1].split(',')[0]).name

        return [(get_word(chunk), get_pos(chunk)) for chunk in
                self.mecab.parse(sent).splitlines()[:-1]]

    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: a list of strings.
        """
        for text in texts:
            seq = self.tokenize(text)

            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts):
        """Transforms each text in texts to a sequence of integers.

        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Returns
            A list of sequences.
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            seq = self.tokenize(text)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect
