import logging
from gensim.test.utils import common_corpus, common_dictionary, datapath
from gensim import corpora
from gensim.models import HdpModel, LdaMulticore
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lang")
args = parser.parse_args()
lang = args.lang

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)

df = pd.read_json(f'preprocessed_dump_{lang}.json', orient='records')



texts = df['text'].tolist()
id2word = corpora.Dictionary(texts)
if lang=='uk':
    bad_tokens = ['свій','рок','__noeditsection__','__notoc__', 'йога']
    id2word.filter_tokens(bad_ids=[id2word.token2id[token] for token in bad_tokens])
corpus = [id2word.doc2bow(text) for text in texts]
lda = LdaMulticore(corpus, id2word=id2word, num_topics=40, passes=15, iterations=100)
lda_file = datapath(f'./lda_{lang}')
lda.save(lda_file)
print("==================================")
for i in range(lda.num_topics):
    topic_terms = lda.get_topic_terms(topicid=i)
    print("Topic ", i)
    for term in topic_terms:
        print(id2word[term[0]], " : ", term[1])