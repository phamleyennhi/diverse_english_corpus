import nltk
import csv
import sys
import os
import string
import pandas as pd
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *

csv.field_size_limit(sys.maxsize)

## Collect and preprocess existing English words corpora
# RUN ONCE to save to CSV.

#This is a combined corpus of several existing English words corpora in NLTK (http://www.nltk.org/nltk_data/)
# 1. wordnet2021
# 2. masc_tagged
# 3. English stopwords
# 4. Word Lists 


# 1. wordnet2021
nltk.download("wordnet2021")
wordnet2021 = LazyCorpusLoader(
    "wordnet2021",
    WordNetCorpusReader,
    LazyCorpusLoader("omw-1.4", CorpusReader, r".*/wn-data-.*\.tab", encoding="utf8"),
)
# break down words e.g. heels_over_head since we do one word comparison
processed_wordnet2021 = set()
for word in wordnet2021.words():
    processed_wordnet2021 = processed_wordnet2021.union(set(word.split("_")))
print(processed_wordnet2021)

cw = csv.writer(open("../data/wordnet2021.csv","w"))
cw.writerow(processed_wordnet2021)


# 2. masc_tagged
nltk.download("masc_tagged")
masc_tagged = LazyCorpusLoader(
    "masc_tagged",
    CategorizedTaggedCorpusReader,
    r"(spoken|written)/.*\.txt",
    cat_file="categories.txt",
    tagset="wsj",
    encoding="utf-8",
    sep="_",
)
lowercase_masc_tagged = list(map(str.lower, masc_tagged.words()))
processed_masc_tagged = set()

# filter non-english words
punctuation = list(string.punctuation)
punctuation.remove('-')
for word in lowercase_masc_tagged:
    add_word = True
    for p in punctuation:
        if p in word:
            add_word = False
    for w in word:
        if w.isdigit():
            add_word = False
    if word == '' or word[-1] == '-':
        add_word = False
    if add_word:
        processed_masc_tagged.add(word)

cw = csv.writer(open("../data/masc_tagged.csv","w"))
cw.writerow(processed_masc_tagged)


# 3. English stopwords
stopwords = nltk.corpus.stopwords.words('english')
cw = csv.writer(open("../data/en_stop_words.csv","w"))
cw.writerow(set(stopwords))


# 4. Word Lists 
en_words_set = list(map(str.lower, nltk.corpus.words.words()))
cw = csv.writer(open("../data/en_words.csv","w"))
cw.writerow(set(en_words_set))


# COMBINE ALL ENGLISH WORDS CORPORA
files = os.listdir('../data/')
if '.DS_Store' in files:
    files.remove('.DS_Store')
english_corpus = set()

for file in files:
    with open(f"../data/{file}", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        list_of_words = list(csv_reader)
        english_corpus.update(list_of_words[0])
cw = csv.writer(open("../data/english_corpus.csv","w"))
cw.writerow(english_corpus)

