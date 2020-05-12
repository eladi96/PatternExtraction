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


# def check_well_formed_xml(path):
#
#     with open(path, mode='r', encoding='utf8') as file:
#         for count, row in enumerate(file):
#             if row[0] != "<" and row[0] != " ":
#                 print(row)
#                 print("Error in row " + str(count))
#                 input()
#     print("Check completed.")
#
#
# def prettify(elem):
#     """Return a pretty-printed XML string for the Element.
#     """
#     rough_string = ElementTree.tostring(elem, 'utf-8')
#     reparsed = minidom.parseString(rough_string)
#     return reparsed.toprettyxml(indent="  ")
#
#
# def generate_cloze_xml(out_path_it, out_path_ja):
#
#     en_sent, it_sent, ja_sent = read_sentences()
#
#     with open("tatoeba/en_ja_links.tsv", mode='r', encoding='utf8') as tsv:
#         reader = csv.reader(tsv, delimiter='\t')
#         en_ja = dict([(row[0], row[1]) for row in reader])
#
#     with open("tatoeba/en_it_links.tsv", mode='r', encoding='utf8') as tsv:
#         reader = csv.reader(tsv, delimiter='\t')
#         en_it = dict([(row[0], row[1]) for row in reader])
#
#     root_it = ElementTree.Element('sentences')
#     root_ja = ElementTree.Element('sentences')
#
#     length = str(len(en_sent))
#     sentId = 1
#     for count, enId, in enumerate(en_sent.keys()):
#         print("\rElement " + str(count) + " of " + length, end='')
#         itId = en_it.get(enId, None)
#         jaId = en_ja.get(enId, None)
#
#         if itId is not None and jaId is None:
#             sentence = ElementTree.SubElement(root_it, 'sentence', {'id': str(sentId)})
#             sentId = sentId + 1
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'eng', 'tatoebaId': str(enId)})
#             item.text = en_sent[enId][1]
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'ita', 'tatoebaId': str(itId)})
#             item.text = it_sent[itId][1]
#
#         if jaId is not None and itId is None:
#             sentence = ElementTree.SubElement(root_ja, 'sentence', {'id': str(sentId)})
#             sentId = sentId + 1
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'eng', 'tatoebaId': str(enId)})
#             item.text = en_sent[enId][1]
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'jpn', 'tatoebaId': str(jaId)})
#             item.text = ja_sent[jaId][1]
#
#     output = open(out_path_it, 'w', encoding='utf8')
#     output.write(prettify(root_it))
#     output = open(out_path_ja, 'w', encoding='utf8')
#     output.write(prettify(root_ja))
#
#
# def generate_xml(out_path = 'tatoeba.xml'):
#
#     en_sent, it_sent, ja_sent = read_sentences()
#
#     with open("tatoeba/en_ja_links.tsv", mode='r', encoding='utf8') as tsv:
#         reader = csv.reader(tsv, delimiter='\t')
#         en_ja = dict([(row[0], row[1]) for row in reader])
#
#     with open("tatoeba/en_it_links.tsv", mode='r', encoding='utf8') as tsv:
#         reader = csv.reader(tsv, delimiter='\t')
#         en_it = dict([(row[0], row[1]) for row in reader])
#
#     root = ElementTree.Element('sentences')
#     length = str(len(en_ja.items()))
#     sentId = 1
#     for count, (enId, jaId) in enumerate(en_ja.items()):
#         print("\rElement " + str(count) + " of " + length, end='')
#         itId = en_it.get(enId, None)
#         if itId is not None:
#             sentence = ElementTree.SubElement(root, 'sentence', {'id': str(sentId)})
#             sentId = sentId + 1
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'eng', 'tatoebaId': str(enId)})
#             item.text = en_sent[enId][1]
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'jpn', 'tatoebaId': str(jaId)})
#             item.text = ja_sent[jaId][1]
#             item = ElementTree.SubElement(sentence, 'item', {'lang': 'ita', 'tatoebaId': str(itId)})
#             item.text = it_sent[itId][1]
#
#     output = open(out_path, 'w', encoding='utf8')
#     output.write(prettify(root))
def generate_dataset():
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
                if count < 120:
                    train.append(sample)
                if 120 <= count < 160:
                    valid.append(sample)
                if count >= 160:
                    test.append(sample)

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return train, valid, test



def tagged_sentences(destination):
    """
    Script used to generate a file containing 200 sentences per tag
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
                    output.write(str(tag) + '\t' + str(count) + "\t" + sentences.pop(row[0])[1] + "\n")
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


if __name__ == '__main__':