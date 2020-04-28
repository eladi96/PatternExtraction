"""
The purpose of this module is to create a linked resource from the sentences contained in the .tsv
files downloaded from tatoeba.org, aligning the sentences from the three different languages.
"""

import csv
from xml.dom import minidom
from xml.etree import ElementTree

# cd Documents\Tesi\PatternExtraction\tatoeba
# python tatoeba.py


def check_well_formed_xml(path):

    with open(path, mode='r', encoding='utf8') as file:
        for count, row in enumerate(file):
            if row[0] != "<" and row[0] != " ":
                print(row)
                print("Error in row " + str(count))
                input()
    print("Check completed.")


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_cloze_xml(out_path_it, out_path_ja):

    en_sent, it_sent, ja_sent = read_sentences()

    with open("en_ja_links.tsv", mode='r', encoding='utf8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        en_ja = dict([(row[0], row[1]) for row in reader])

    with open("en_it_links.tsv", mode='r', encoding='utf8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        en_it = dict([(row[0], row[1]) for row in reader])

    root_it = ElementTree.Element('sentences')
    root_ja = ElementTree.Element('sentences')

    length = str(len(en_sent))
    sentId = 1
    for count, enId, in enumerate(en_sent.keys()):
        print("\rElement " + str(count) + " of " + length, end='')
        itId = en_it.get(enId, None)
        jaId = en_ja.get(enId, None)

        if itId is not None and jaId is None:
            sentence = ElementTree.SubElement(root_it, 'sentence', {'id': str(sentId)})
            sentId = sentId + 1
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'en', 'tatoebaId': str(enId)})
            item.text = en_sent[enId][1]
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'it', 'tatoebaId': str(itId)})
            item.text = it_sent[itId][1]

        if jaId is not None and itId is None:
            sentence = ElementTree.SubElement(root_ja, 'sentence', {'id': str(sentId)})
            sentId = sentId + 1
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'en', 'tatoebaId': str(enId)})
            item.text = en_sent[enId][1]
            item = ElementTree.SubElement(sentence, 'item', {'lang': 'ja', 'tatoebaId': str(jaId)})
            item.text = ja_sent[jaId][1]

    output = open(out_path_it, 'w', encoding='utf8')
    output.write(prettify(root_it))
    output = open(out_path_ja, 'w', encoding='utf8')
    output.write(prettify(root_ja))


def generate_xml(out_path = 'tatoeba.xml'):

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

    output = open(out_path, 'w', encoding='utf8')
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

    generate_cloze_xml('tatoeba_en_it.xml', 'tatoeba_en_ja.xml')