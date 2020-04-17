"""
The purpose of this module is to create a linked resource from the sentences contained in the .tsv
files downloaded from tatoeba.org, aligning the sentences from the three different languages.
"""

import csv
from xml.etree import ElementTree
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_xml():

    en_sent, it_sent, ja_sent = read_sentences()

    with open("en_ja_links.tsv", mode='r', encoding='utf8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        en_ja = dict([(row[0], row[1]) for row in reader])

    with open("en_it_links.tsv", mode='r', encoding='utf8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        en_it = dict([(row[0], row[1]) for row in reader])

    root = ElementTree.Element('sentences')
    length = str(len(en_ja.items()))
    sentId = 1
    for count, (enId, jaId) in enumerate(en_ja.items()):
        print("\rElement " + str(count) + " of " + length, end='')
        itId = en_it.get(enId, None)
        if itId is not None:
            sentence = ElementTree.SubElement(root, 'sentence', {'id': str(sentId)})
            sentId = sentId + 1
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'en', 'tatoebaId': str(enId)})
            item.text = en_sent[enId][1]
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'ja', 'tatoebaId': str(jaId)})
            item.text = ja_sent[jaId][1]
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'it', 'tatoebaId': str(itId)})
            item.text = it_sent[itId][1]

    output = open('tatoeba.xml', 'w', encoding='utf8')
    output.write(prettify(root))


def coupled_links():
    """
    Script used to obtain the links between couples of languages
    """

    en_sent, it_sent, ja_sent = read_sentences()

    en_ja = open("en_ja_links.tsv", mode='w', encoding='utf8')
    en_it = open("en_it_links.tsv", mode='w', encoding='utf8')

    with open("links.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for count, row in enumerate(reader):
            if row[0] in en_sent:
                print("\rReading line " + str(count) + " of 17.318.501", end='')
                if row[1] in ja_sent:
                    en_ja.write(row[0] + "\t" + row[1] + "\n")
                if row[1] in it_sent:
                    en_it.write(row[0] + "\t" + row[1] + "\n")


def en_it_ja_links():
    """
    Script used to obtain only the links in the interested languages from the links file
    """

    sentences, it_sent, ja_sent = read_sentences()
    sentences.update(it_sent)
    sentences.update(ja_sent)
    with open("links.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        with open("eng_ita_jpn_links.tsv", mode='w') as destination:
            for count, row in enumerate(reader):
                print("\rReading line " + str(count) + " of 17.318.500", end='')
                if row[0] in sentences and row[1] in sentences:
                    destination.write(row[0] + "\t" + row[1] + "\n")


def read_sentences():
    """
    Method for readings sentences from tatoeba files.
    @:return eng_sentences, ita_sentences, jap_sentences
    """
    # Reading sentences from files and adding them to the sentences dictionary
    with open("en/en_sentences.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        eng_sentences = dict([(row[0], tuple([row[1], row[2]])) for row in reader])
        print("Read English sentences.")

    with open("it/it_sentences.tsv.", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        ita_sentences = dict([(row[0], tuple([row[1], row[2]])) for row in reader])
        print("Read Italian sentences.")

    with open("ja/ja_sentences.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        jap_sentences = dict([(row[0], tuple([row[1], row[2]])) for row in reader])
        print("Read Japanese sentences.")

    return eng_sentences, ita_sentences, jap_sentences


if __name__ == '__main__':

    read_sentences()
