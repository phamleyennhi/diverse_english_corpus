# import libraries
import os
import langid
import re
import pandas as pd
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from langid.langid import LanguageIdentifier, model
from googletrans import Translator


df_result = pd.DataFrame(columns=['lang_identifier', "location", 'syntactic_english', 'non_syntactic_english', 'informal_english', 'code_switched', 'stats [each category, total]'])
def save_stats(df, df_col_raw, lang_id, location):
    global df_result
    
    df_informal = df[df["label"] == "informal-english"]
    df_informal_english = df_informal[df_informal[df_col_raw] == "en"]

    df_syntactic = df[df["label"] == "syntactic-english"]
    df_syntactic_english = df_syntactic[df_syntactic[df_col_raw] == "en"]

    df_non_syntactic = df[df["label"] == "non-syntactic-english"]
    df_non_syntactic_english = df_non_syntactic[df_non_syntactic[df_col_raw] == "en"]

    df_code_switch = df[df["label"] == "code-switched"]
    df_code_switch_english = df_code_switch[df_code_switch[df_col_raw] == "en"]
    
    df_incidental = df[df["label"] == "incidental-english"]
    df_non_english = df[df["label"] == "not-english"]

    total_raw = (df_syntactic_english.shape[0] + df_non_syntactic_english.shape[0] + df_informal_english.shape[0] + df_code_switch_english.shape[0])
    df_result = df_result.append({'lang_identifier': lang_id,
                                  'location': location,
                                  'syntactic_english': df_syntactic_english.shape[0],
                                  'non_syntactic_english': df_non_syntactic_english.shape[0],
                                  'informal_english': df_informal_english.shape[0],
                                  'code_switched': df_code_switch_english.shape[0],
                                  'stats [each category, total]': (df_syntactic.shape[0],
                                                                   df_non_syntactic.shape[0],
                                                                   df_informal.shape[0],
                                                                   df_code_switch.shape[0],
                                                                   df_incidental.shape[0],
                                                                   df_non_english.shape[0], total_raw)}, ignore_index=True)

# SET UP MODELS
# spacy
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
   return LanguageDetector()
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('language_detector', last=True)

# google trans API
translator = Translator()

def gen_raw_text(df):
	df["raw_text"] = df["annotate_text"].str.replace("<b>Raw:</b> ", "").str.split("<b>Clean</b>: ").str[0]
	df["clean_text"] = df["annotate_text"].str.replace("<b>Raw:</b> ", "").str.split("<b>Clean</b>: ").str[1]
	return df

def lang_id(df, city):
    print("LANGID")
    df["langid_raw"] = None
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    for i in range(df.shape[0]):
        rtext = df["raw_text"].iloc[i]
        df["langid_raw"].iloc[i] = identifier.classify(rtext)[0]
    save_stats(df, "langid_raw", "lang_id.py", city)
    
def spacy_id(df, city):
    print("SPACY")
    df["spacy_raw"] = None
    df["spacy_clean"] = None

    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    for i in range(df.shape[0]):
        rtext = df["raw_text"].iloc[i]
        df["spacy_raw"].iloc[i] = nlp(rtext)._.language['language']
    save_stats(df, "spacy_raw", "spacy", city)
    # df.to_csv(f'../spacy_annotated/{city}.csv')  # uncomment to save automatically annotated data

def google_trans_api(df, city):
    print("GOOGLE TRANSLATOR")
    df["googletrans_raw"] = None
    df["googletrans_clean"] = None
    translator = Translator()
    for i in range(df.shape[0]):
        rtext = df["raw_text"].iloc[i]
        df["googletrans_raw"].iloc[i] = translator.detect(rtext).lang
    save_stats(df, "googletrans_raw", "googletrans_clean", "googletransAPI", city)


# EXAMPLES ON EVAL MODELS
# df_accra = gen_raw_text(pd.read_csv("../../diverse_corpus/data/annotated/Accra.csv"))
# lang_id(df_accra, "Accra")
# print(df_result)
        