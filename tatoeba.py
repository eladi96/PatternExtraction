"""
The purpose of this module is to create a dataset from the sentences contained in the .tsv
files downloaded from tatoeba.org, aligning the sentences from the three different languages.
"""
import csv
import os
import random
import wget
import tarfile
from constants import *


def generate_dataset():
    """
    Script used to split the tagged sentences into training, validation and test set.
    """
    train = []
    valid = []
    test = []

    with open(os.path.join(TATOEBA, BEST_TAGS), mode='r') as file:
        tags_list = [line.split('\t')[0] for line in file]

    for tag in tags_list:
        with open(os.path.join(TATOEBA, TAGGED_SENT), mode='r') as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            samples = [line for line in reader if line[0] == tag]
            random.shuffle(samples)
            train_dim = (len(samples) / 100) * 80
            val_dim = (len(samples) / 100) * 10
            for count, sample in enumerate(samples):
                if count < train_dim:
                    train.append(sample)
                if train_dim <= count < train_dim + val_dim:
                    valid.append(sample)
                if count >= train_dim + val_dim:
                    test.append(sample)

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return train, valid, test


def tagged_sentences():
    """
    Script used to generate a file containing up to 500 sentences per tag in the three languages.
    Every row of the output file will contain [tag, eng_id, eng_sentence, jpn_sentence, ita_sentence]
    """
    with open(os.path.join(TATOEBA, BEST_TAGS), mode='r') as file:
        tags_list = [line.split('\t')[0] for line in file]

    eng_sentences = read_sentences(ENG_SENT)
    jpn_sentences = read_sentences(JPN_SENT)
    ita_sentences = read_sentences(ITA_SENT)
    links_jpn = {row[0]: row[1] for row in
                 csv.reader(open(os.path.join(TATOEBA, ENG_JPN_LINKS), mode='r'), delimiter='\t')}
    links_ita = {row[0]: row[1] for row in
                 csv.reader(open(os.path.join(TATOEBA, ENG_ITA_LINKS), mode='r'), delimiter='\t')}

    output = open(os.path.join(TATOEBA, TAGGED_SENT), mode='w')
    for tag in tags_list:
        count = 1
        with open(os.path.join(TATOEBA, ENG_TAGS)) as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            gen = (row for row in reader if count <= 500)
            for row in gen:
                eng_sent = eng_sentences.get(row[0], None)

                jpn_id = links_jpn.get(row[0], None)
                jpn_sent = jpn_sentences.get(jpn_id, None)

                ita_id = links_ita.get(row[0], None)
                ita_sent = ita_sentences.get(ita_id, None)

                if row[1] == tag and eng_sent is not None and jpn_sent is not None and ita_sent is not None:
                    output.write(str(tag) + '\t' + row[0] + "\t" + eng_sentences.pop(row[0]) + "\t" + jpn_sentences.pop(
                        jpn_id) + "\t" + ita_sentences.pop(ita_id) + "\n")
                    count += 1
    output.close()


def sentences_tags(sent_file):
    """
    Script used to save in a file the tags associated to a certain set of sentences.
    :param sent_file: the name of the file containing the sentences
    """

    sentences = read_sentences(sent_file)
    filename = sent_file.replace('_sentences.tsv', '_tags.tsv')
    output = open(os.path.join(TATOEBA, filename), mode='w', encoding='utf8')

    with open(os.path.join(TATOEBA, TAGS), mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            if row[0] in sentences:
                output.write(row[0] + "\t" + row[1] + "\n")


def coupled_links(lang1_file, lang2_file):
    """
    Script used to save in a file the links between sentences in two languages
    :param: path_lang1: the file containing the first language sentences
    :param: path_lang2 the file containing the second language sentences
    """

    lang1_sents = read_sentences(lang1_file)
    lang2_sents = read_sentences(lang2_file)

    filename = lang1_file[0:3] + '_' + lang2_file[0:3] + '_links.tsv'
    output = open(os.path.join(TATOEBA, filename), mode='w', encoding='utf8')

    if not os.path.exists(os.path.join(TATOEBA, COUPLED_LINKS)):
        print("Downloading links.tsv from tatoeba.org..", end="")
        filename = wget.download(TATOEBA_LINKS_URL, TATOEBA)
        print("\rExtracting archive...", end="")
        tar = tarfile.open(filename)
        tar.extractall(TATOEBA)
        tar.close()
        os.remove(filename)
        print("\rCompleted!")

    with open(os.path.join(TATOEBA, COUPLED_LINKS), mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            if row[0] in lang1_sents and row[1] in lang2_sents:
                output.write(row[0] + "\t" + row[1] + "\n")


def read_sentences(filename):
    """
    Method for readings sentences from tatoeba files.
    :param: filename: the name of the file containing the sentences
    :return: dict of sentences in the form {tatoebaId : sentence}
    """
    # Reading sentences from files and adding them to the sentences dictionary
    with open(os.path.join(TATOEBA, filename), mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        sentences = dict([(row[0], row[2]) for row in reader])
        print("Read sentences.")

    return sentences
