"""
The purpose of this module is to create a linked resource from the sentences contained in the .tsv
files downloaded from tatoeba.org, aligning the sentences from the three different languages.
"""
import csv
import os
import random
import wget
import tarfile

TATOEBA_PATH = 'tatoeba'
TATOEBA_LINKS_URL = 'https://downloads.tatoeba.org/exports/links.tar.bz2'
COUPLED_LINKS = 'links.csv'
TAGS = 'tags.csv'
ENG_SENT = 'eng_sentences.tsv'
ITA_SENT = 'ita_sentences.tsv'
JPN_SENT = 'jpn_sentences.tsv'
ENG_TAGS = 'eng_tags.tsv'
BEST_TAGS = 'best_tags.txt'
TAGGED_SENT = 'tagged_sentences.tsv'


def generate_dataset():
    """
    Script used to split the tagged sentences into training, validation and test set.
    """
    train = []
    valid = []
    test = []

    with open(os.path.join(TATOEBA_PATH, BEST_TAGS), mode='r') as file:
        tags_list = [line.split(':')[0] for line in file]
        tags_list.reverse()

    for tag in tags_list:
        with open(os.path.join(TATOEBA_PATH, TAGGED_SENT), mode='r') as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            samples = [line for line in reader if line[0] == tag]
            random.shuffle(samples)
            for count, sample in enumerate(samples):
                if count < 160:
                    train.append(sample)
                if 160 <= count < 180:
                    valid.append(sample)
                if count >= 180:
                    test.append(sample)

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return train, valid, test


def tagged_sentences(destination):
    """
    Script used to generate a file containing 200 sentences per tag.
    Every row of the output file will contain [tag, count, tatoebaId, sentence]
    :param destination: name of the destination file
    """
    with open(os.path.join(TATOEBA_PATH, BEST_TAGS), mode='r') as file:
        tags_list = [line.split(':')[0] for line in file]
        tags_list.reverse()

    sentences = read_sentences(ENG_SENT)
    output = open(os.path.join(TATOEBA_PATH, destination), mode='w')
    for tag in tags_list:
        count = 1
        with open(os.path.join(TATOEBA_PATH, ENG_TAGS)) as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            gen = (row for row in reader if count <= 200)
            for row in gen:
                if row[1] == tag and sentences.get(row[0], None) is not None:
                    sent = sentences.pop(row[0])
                    output.write(str(tag) + '\t' + str(count) + "\t" + row[0] + "\t" + sent[1] + "\n")
                    count += 1
    output.close()


def sentences_tags(sent_file):
    """
    Script used to save in a file the tags associated to a certain set of sentences.
    :param sent_file: the name of the file containing the sentences
    """

    sentences = read_sentences(os.path.join(TATOEBA_PATH, sent_file))
    filename = sent_file.replace('_sentences.tsv', '_tags.tsv')
    output = open(os.path.join(TATOEBA_PATH, filename), mode='w', encoding='utf8')

    with open(os.path.join(TATOEBA_PATH, TAGS), mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            if row[0] in sentences:
                output.write(row[0] + "\t" + row[1] + "\n")


def coupled_links(lang1_file, lang2_file):
    """
    Script used to save in a file the links between sentences in two languages
    :param: path_lang1: the path to the file containing the first language sentences
    :param: path_lang2 the path to the file containing the second language sentences
    """

    lang1_sents = read_sentences(os.path.join(TATOEBA_PATH, lang1_file))
    lang2_sents = read_sentences(os.path.join(TATOEBA_PATH, lang2_file))

    filename = lang1_file[0:3] + '_' + lang2_file[0:3] + '_links.tsv'
    output = open(os.path.join(TATOEBA_PATH, filename), mode='w', encoding='utf8')

    if not os.path.exists(os.path.join(TATOEBA_PATH, COUPLED_LINKS)):
        print("Downloading links.tsv from tatoeba.org..", end="")
        filename = wget.download(TATOEBA_LINKS_URL, TATOEBA_PATH)
        print("\rExtracting archive...", end="")
        tar = tarfile.open(filename)
        tar.extractall(TATOEBA_PATH)
        tar.close()
        os.remove(filename)
        print("\rCompleted!")

    with open(os.path.join(TATOEBA_PATH, COUPLED_LINKS), mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            if row[0] in lang1_sents and row[1] in lang2_sents:
                output.write(row[0] + "\t" + row[1] + "\n")


def read_sentences(filename):
    """
    Method for readings sentences from tatoeba files.
    :param: path: the path to the tsv file containing the sentences
    :return: dict of sentencese in the form {tatoebaId : (lang, sentence)}
    """
    # Reading sentences from files and adding them to the sentences dictionary
    with open(os.path.join(TATOEBA_PATH, filename), mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        sentences = dict([(row[0], tuple([row[1], row[2]])) for row in reader])
        print("Read sentences.")

    return sentences
