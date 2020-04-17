from urllib.request import Request, urlopen
from urllib.parse import urlencode
import json
import gzip
from io import BytesIO

KEY = '3cce11db-d3f4-452b-98cc-649578755c2b'


def search_by_lemma(lemma, lang):
    """
    Function used to search a lemma in the given language in Babelnet.
    Retrieves the IDs of the Babel synsets (concepts) denoted by a given word.
    e.g. search_by_lemma("apple", "EN")

    :param lemma: The word you want to search for
    :param lang: The language of the word. EN, IT, JA
    :return: returns a list of dicts
            {'id': babelSynsetID,
             'pos': part of speach,
             'source': source of the synset}
             None if the lemma is not found.
    """

    service_url = 'https://babelnet.io/v5/getSynsetIds'

    params = {
        'lemma': lemma,
        'searchLang': lang,
        'key': KEY
    }

    url = service_url + '?' + urlencode(params)
    request = Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urlopen(request)

    data = None
    if response.info().get('Content-Encoding') == 'gzip':
        buf = BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())

    return data


def lemmas_by_id(babelnet_id, lang="EN"):
    """
    Retrieves the list of simple-lemmas of a given synset.
    e.g. search_by_id('bn:00043021n')

    :param babelnet_id: id of the synset you want to retrieve
    :param lang: The languages in which the data are to be retrieved. Default value
    is the English. Italian = IT, Japanese = JA
    :return: list of simple-lemmas of a synset in the given language. Empty list if id not found.
    """
    service_url = 'https://babelnet.io/v5/getSynset'

    params = {
        'id': babelnet_id,
        'key': KEY,
        'targetLang': lang
    }

    url = service_url + '?' + urlencode(params)
    request = Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urlopen(request)

    lemmas = list()
    if response.info().get('Content-Encoding') == 'gzip':
        buf = BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())

        # retrieving BabelSense data
        senses = data['senses']
        for result in senses:
            properties = result['properties']
            lemmas.append(properties['simpleLemma'].replace("_", " "))

    return lemmas

