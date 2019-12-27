from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from stop_words import get_stop_words
import logging
import glob
import mwxml
import mwparserfromhell
import tokenize_uk
import argparse
import re
import pymorphy2
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(42)
import nltk
nltk.download('wordnet')

morph = pymorphy2.MorphAnalyzer(lang='uk')


def process_dump(dump, path):
    global lang
    print("Lang:", lang )
    if lang == 'uk':
        with open('ukrainian-stopwords.txt', 'r') as f:
            stop_words = f.readlines()

        stop_words = [word.strip() for word in stop_words]
        stop_words.extend(['категорія', 'із', 'р', 'jpg', 'png', 'с', 'c', 'й', 'і', 'i'])
    elif lang == 'en':
        stop_words = gensim.parsing.preprocessing.STOPWORDS
    for page in dump:
        if page.namespace == 0 and page.redirect is None:
            for revision in page:
                if lang == 'uk':
                    text = revision.text
                    text = filter_wiki(text)
                    tokens = tokenize_uk.tokenize_words(text)
                    words_normalized = []
                    for token in tokens:
                        p = morph.parse(token)[0]
                        if p.normal_form not in stop_words and re.match('\w+', p.normal_form) and not re.match('\d+',
                                                                                                               p.normal_form)\
                                and len(token) > 3:
                            words_normalized.append(p.normal_form)
                    yield page.id, page.title, words_normalized
                    break
                elif lang == 'en':
                    text = revision.text
                    text = filter_wiki(text)
                    words_normalized = []
                    for token in gensim.utils.simple_preprocess(text):
                        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and re.match('\w+',
                                                                                                               token):
                            lemmatized = WordNetLemmatizer().lemmatize(token, pos='v')
                            words_normalized.append(lemmatized)
                    yield page.id, page.title, words_normalized


parser = argparse.ArgumentParser()
parser.add_argument("lang")
args = parser.parse_args()
lang = args.lang
print('lang', lang)
paths = glob.glob(f'{lang}wiki*.bz2')

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


df = pd.DataFrame(columns=['page_id', 'title', 'text'])

count = 0
for page_id, page_title, text in mwxml.map(process_dump, paths):
    count += 1
    if count < 10:
        print(text)
    if count % 5000 == 0:
        print("Processed ", count, "pages...")
    df = df.append({'page_id': page_id, 'title': page_title, 'text': text}, ignore_index=True)

df.reset_index().to_json(f'preprocessed_dump_{lang}.json')

