import xml.etree.ElementTree as ET
import csv
import os
from xml.dom import minidom

# cd Documents\Tesi\Pattern_extraction\tatoeba
# python preprocessing.py


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# semagrams = ET.parse(os.path.join(os.path.dirname(os.getcwd()), "semagrams_300.xml"))


def check_well_formed_xml(path):

    with open(path, mode='r', encoding='utf8') as file:
        for count, row in enumerate(file):
            if row[0] != "<" and row[0] != " ":
                print(row)
                print("Error in row " + str(count))
                input()
    print("Check completed.")


def tatoeba_to_semagram_eng():

    lang = "EN"
    semagrams = ET.parse(os.path.join(os.path.dirname(os.getcwd()), 'semagrams.xml')).getroot()
    root = ET.Element('sentences')

    with open("en/eng_sentences.tsv", mode='r', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        eng_sentences = dict([(row[0], row[2]) for row in reader])
        print("Read English sentences.")

    slots = dict()
    for semagram in semagrams:
        slot = ET.SubElement(root, 'semagram')
        slot.set('babelsynset', semagram.attrib['babelsynset'])
        slot.set('name', semagram.attrib['name'])
        slots[semagram.attrib['babelsynset']] = slot

    print("Reading sentences...")
    for count, (sentId, sentence) in enumerate(eng_sentences.items()):
        tokenized = Text(sentence).words
        for synsetId, slot in slots.items():
            keyword = slot.get('name')
            if keyword in tokenized:
                item = ET.SubElement(slot, 'sentence')
                item.set('id', sentId)
                item.set('lang', lang)
                item.set('source', 'tatoeba')
                item.text = sentence
        # print("\r" + str(count) + " of 1311973", end='')

    output = open('tatoeba/en/tatoeba_semagram_eng.xml', 'w', encoding='utf8')
    output.write(prettify(root))
    print()
    print("Completed!")


# def eng_pos_accordance(xmlfile):
#
#     babelnet_pos = {'n': 'NOUN', 'v': 'VERB'}
#     tree = ET.parse(xmlfile)
#     root = tree.getroot()
#     for semagram in root:
#         pos = babelnet_pos[semagram.get('babelsynset')[-1]]
#         word = semagram.get('name')
#         for item in semagram:
#             tagged = dict(Text(item.text, hint_language_code='en').pos_tags)
#             if pos != tagged[word]:
#                 semagram.remove(item)
#
#     tree.write(os.path.join(xmlfile.strip(".xml"), "_cleaned.xml"), encoding='utf8')


# if __name__ == '__main__':

