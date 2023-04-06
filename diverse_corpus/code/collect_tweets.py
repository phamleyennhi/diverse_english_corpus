# import libraries
import itertools
import re
import csv
import nltk
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from collections import Counter
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)


def date_range(start, end):
    start_date = datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.strptime(end, '%Y-%m-%d').date()
    delta = end_date - start_date 
    days = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    return list(map(lambda n: n.strftime("%Y-%m-%d"), days))

def get_tweets_over_period(city, amount_per_day, data_collection_period, distance='20km'):
    df_city = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(f'near:"{city}" within:{distance} since:{data_collection_period[0]} until:{data_collection_period[1]} ').get_items(), amount_per_day))[['date', 'content']]
    for i in tqdm(range(1, len(data_collection_period)-1)):
        df_temp = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(f'near:"{city}" within:{distance} since:{data_collection_period[i]} until:{data_collection_period[i+1]} ').get_items(), amount_per_day))[['date', 'content']]
        df_city = df_city.append(df_temp)
    df_city = df_city.rename(columns={"content": "rawContent"})
    os.makedirs(f"data/{city}", exist_ok = True)
    df_city.to_csv(f"data/{city}/{amount_per_day*len(data_collection_period)}_tweets_over_period.csv")
    return df_city

def clean_tweet(tweet):
    if type(tweet) == float:
        return ""
    temp = tweet.lower()
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('[.*?\-…]',' ', temp)
    temp = re.sub('&amp;','and', temp)
    temp = re.sub("\n"," ", temp)
    temp = re.sub("\t"," ", temp)
    temp = re.sub("[^a-z0-9À-ž ]","", temp)
    temp = temp.split()
    temp = " ".join(word for word in temp)
    return temp


def clean_tweets_from_file(file_name):
    df_city = pd.read_csv(file_name, index_col=[0])
    df_city = df_city.rename(columns={"content": "rawContent"})
    df_city['cleaned_content'] = [clean_tweet(i) for i in df_city['rawContent']]
    df_city['id'] = range(1, len(df_city)+1)
    df_city = df_city[['id'] + [x for x in df_city.columns if x != 'id']]
    df_city.set_index('id')
    df_city.to_csv(file_name)
    return df_city


data_collection_period = date_range('2022-01-01', '2022-09-01')

# Note that current Twitter API is not working
# Our data was collected in 2022.
# non-Western English varieties
df_accra = get_tweets_over_period('Accra', 100, data_collection_period)
df_islamabad = get_tweets_over_period('Islamabad', 100, data_collection_period)
df_manila = get_tweets_over_period('Manila', 100, data_collection_period)
df_newdelhi = get_tweets_over_period('New Delhi', 100, data_collection_period)
df_singapore = get_tweets_over_period('Singapore', 100, data_collection_period)

# Wester English varieties
df_london = get_tweets_over_period('London', 100, data_collection_period)
df_newyork = get_tweets_over_period('New York', 100, data_collection_period)


