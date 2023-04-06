import csv
import os
from nltk.stem import WordNetLemmatizer
import pandas as pd
from collections import Counter


# CREATE ENGLISH WORDS SET
lemmatizer = WordNetLemmatizer()
english_words_list = []
with open('../../english_words_corpora/data/english_corpus.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    list_of_csv = list(csv_reader)
    english_words_list = list_of_csv[0]      
english_words_set = set(english_words_list)



def generate_word_corpus(filename):
    corpus = []
    df = pd.read_csv(filename)
    cleaned_content = df["cleaned_content"]

    for row in cleaned_content:
        if not pd.isnull(row) and len(row) > 0:
            all_words_in_sentence = row.strip().split(" ")
            corpus += all_words_in_sentence
    return corpus


def generate_word_frequency(city, approach, corpus):
    counter=Counter(corpus)
    most=counter.most_common()
    write_to_csv(f"data/{city}/{approach}_topMostFrequentWords", most[:100])
    write_to_csv(f"data/{city}/{approach}_topLeastFrequentWords", most[-100::])
    
    # non-English words
    non_english_words = []
    for tup in most:
        if (lemmatizer.lemmatize(tup[0]) not in english_words_set and tup[0].isnumeric() == False):
            non_english_words.append(tup)
    return non_english_words
    

def average_sentence_length(city):
    total_length = 0
    count = 0
    with open(f"data/{city}.csv") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            if len(row) > 3:
                all_words_in_sentence = row[3].split(" ")
                total_length += len(all_words_in_sentence)
                count += 1
    print("Average sentence length:", total_length/count)


## EXPLORE LINGUISTIC VARIATIONS

def lexical_variation(corpus):
	# filter out digits
	filtered_corpus = [item for item in corpus if not item.isdigit()]
	non_english_words = []
	for word in filtered_corpus:
	    if word not in english_words_set:
	        non_english_words.append(word)
	counter=Counter(non_english_words)  

	# percentage of non-English words
	filtered_corpus_set = set(filtered_corpus)
	non_english_count = 0
	for word in filtered_corpus_set:
	    if word not in english_words_set: 
	        non_english_count += 1
	        
	print(non_english_count/len(filtered_corpus_set))


# create word corpus of all English varieties
gen_path = "../../diverse_corpus/data/not_annotated"

corpus_accra = generate_word_corpus(f"{gen_path}/Accra/Accra.csv")
corpus_islamabad = generate_word_corpus(f"{gen_path}/Islamabad/Islamabad.csv")
corpus_delhi = generate_word_corpus(f"{gen_path}/New Delhi/New Delhi.csv")
corpus_manila = generate_word_corpus(f"{gen_path}/Manila/Manila.csv")
corpus_singapore = generate_word_corpus(f"{gen_path}/Singapore/Singapore.csv")

corpus_london = generate_word_corpus(f"{gen_path}/London/London.csv")
corpus_newyork = generate_word_corpus(f"{gen_path}/New York/New York.csv")


# create combined corpora of Western and non-Western English varieties
corpus_non_western = []
corpus_non_western.extend(corpus_singapore)
corpus_non_western.extend(corpus_accra)
corpus_non_western.extend(corpus_islamabad)
corpus_non_western.extend(corpus_delhi)
corpus_non_western.extend(corpus_manila)

corpus_western = []
corpus_western.extend(corpus_london)
corpus_western.extend(corpus_newyork)


lexical_variation(corpus_non_western)
lexical_variation(corpus_western)


