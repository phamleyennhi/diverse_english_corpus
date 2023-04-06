from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

import pandas as pd
from sklearn.utils import shuffle

# Result might change based on the sample selected.
# Overall, we received higher performance when NB trained on manually annotated data of all varieties.


gen_path = "../spacy_annotated"

# non-Western English
df1 = pd.read_csv(f"{gen_path}/Accra.csv")
df2 = pd.read_csv(f"{gen_path}/Islamabad.csv")
df3 = pd.read_csv(f"{gen_path}/Manila.csv")
df4 = pd.read_csv(f"{gen_path}/New Delhi.csv")
df5 = pd.read_csv(f"{gen_path}/Singapore.csv")

# Western English
df6 = pd.read_csv(f"{gen_path}/New York.csv")
df7 = pd.read_csv(f"{gen_path}/London.csv")

def split_english_vs_non_english(df, balanced=False):
    df_informal = df[df["label"] == "informal-english"]
    df_syntactic = df[df["label"] == "syntactic-english"]
    df_non_syntactic = df[df["label"] == "non-syntactic-english"]
    df_code_switch = df[df["label"] == "code-switched"]
    df_incidental = df[df["label"] == "incidental-english"]
    df_non_english = df[df["label"] == "not-english"]
    
    if balanced:
    	df_english = pd.concat([df_informal.sample(75), df_syntactic.sample(75), df_non_syntactic.sample(75), df_code_switch.sample(75)], ignore_index=True)
		df_non_english = pd.concat([df_incidental.sample(150), df_non_english.sample(150)], ignore_index=True)
	else:
	    df_english = pd.concat([df_informal, df_syntactic, df_non_syntactic, df_code_switch], ignore_index=True)
	    df_non_english = pd.concat([df_incidental, df_non_english], ignore_index=True)
	    
    return df_english, df_non_english


def naive_bayes(test, train):
    tweets = []
    labels = []

    for i in range(train.shape[0]):
        tweets.append(train["raw_text"].iloc[i])
        if train["label"].iloc[i] == "not-english" or train["label"].iloc[i] == "incidental-english":
            labels.append(0)
        else:
            labels.append(1)

    for i in range(test.shape[0]):
        tweets.append(test["raw_text"].iloc[i])
        if test["label"].iloc[i] == "not-english" or test["label"].iloc[i] == "incidental-english":
            labels.append(0)
        else:
            labels.append(1)
            
    vectorizer = CountVectorizer()  
    X = vectorizer.fit_transform(tweets)

    X_train = X[:train.shape[0]]
    Y_train = labels[:train.shape[0]]

    X_test = X[train.shape[0]:]
    Y_test = labels[train.shape[0]:]
    
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    accuracy = clf.score(X_test, Y_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    
    print("Classifier accuracy: {:.2f}%".format(accuracy*100))
    print("Classifier precision: {:.2f}%".format(precision*100))
    print("Classifier recall: {:.2f}%".format(recall*100))



## AUTO ANNOTATED vs. MANUALLY ANNOTATED
def balanced_naive_bayes(df):

	df_english, df_non_english = split_english_vs_non_english(df, True)

	train_english =df_english.sample(frac=0.8,random_state=200)
	train_non_english =df_non_english.sample(frac=0.8,random_state=200)
	train = pd.concat([train_english, train_non_english])
	train = shuffle(train)

	test_english = df_english.drop(train_english.index)
	test_non_english = df_non_english.drop(train_non_english.index)
	test = pd.concat([test_english, test_non_english])
	test = shuffle(test)

	naive_bayes(test, train)

	# RESULTS when balanced

	# Trained on SpaCY-annotated
	# Classifier accuracy: 86.67%
	# Classifier precision: 89.29%
	# Classifier recall: 83.33%

	# Trained on our corpus
	# Classifier accuracy: 92.50%
	# Classifier precision: 96.36%
	# Classifier recall: 88.33%



def non_balanced_naive_bayes(df):
	df_english, df_non_english = split_english_vs_non_english(df)

	train_english =df_english.sample(frac=0.8,random_state=200)
	train_non_english =df_non_english.sample(frac=0.8,random_state=200)
	train = pd.concat([train_english, train_non_english])
	train = shuffle(train)

	test_english = df_english.drop(train_english.index)
	test_non_english = df_non_english.drop(train_non_english.index)
	test = pd.concat([test_english, test_non_english])
	test = shuffle(test)

	naive_bayes(test, train)

	# RESULTS when non balanced

	# Trained on SpaCY-annotated
	# Classifier accuracy: 78.29%
	# Classifier precision: 96.80%
	# Classifier recall: 75.44%

	# Trained on our corpus
	# Classifier accuracy: 88.00%
	# Classifier precision: 95.09%
	# Classifier recall: 89.68%



## SOLELY ON MANUALLY ANNOTATED DATA

def naive_bayes_train_on_western(df_varieties, df_western):
    global df_varieties, df_western
    
    df_english_western, df_non_english_western = split_english_vs_non_english(df_western)
    df_english_varieties, df_non_english_varieties = split_english_vs_non_english(df_varieties)
    
    train_english =df_english_western.sample(frac=0.8,random_state=200)
    train_non_english =df_non_english_western.sample(frac=0.8,random_state=200)
    train = pd.concat([train_english, train_non_english])
    train = shuffle(train)

    test_english_western = df_english_western.drop(train_english.index)
    test_non_english_western = df_non_english_western.drop(train_non_english.index)
    test_english_varieties =df_english_varieties.sample(frac=0.,random_state=200)
    test_non_english_varieties =df_non_english_varieties.sample(frac=0.2,random_state=200)
    
    test = pd.concat([test_english_western, test_non_english_western, test_english_varieties, test_non_english_varieties])
    test = shuffle(test)
    print(test_english_western.shape[0], test_non_english_western.shape[0], test_english_varieties.shape[0], test_non_english_varieties.shape[0])
    
    naive_bayes(test, train)
    
    
def naive_bayes_train_on_all(df_varieties, df_western):
    global df_varieties, df_western
    df = pd.concat([df_varieties, df_western], ignore_index=True)
    df_english, df_non_english = split_english_vs_non_english(df)
    
    train_english =df_english.sample(frac=0.8,random_state=200)
    train_non_english =df_non_english.sample(frac=0.8,random_state=200)
    train = pd.concat([train_english, train_non_english])
    train = shuffle(train)

    test_english = df_english.drop(train_english.index)
    test_non_english = df_non_english.drop(train_non_english.index)
    test = pd.concat([test_english, test_non_english])
    test = shuffle(test)
    
    naive_bayes(test.sample(209), train.sample(560)) # equal amount to when trained on western English
    
	# frames1 = [df1, df2, df3, df4, df5]
	# df_varieties = pd.concat(frames1, ignore_index=True)

	# frames2 = [df6, df7]
	# df_western = pd.concat(frames2, ignore_index=True)

	# naive_bayes_train_on_western(df_varieties, df_western)
	# naive_bayes_train_on_all(df_varieties, df_western)

	# RESULTS
	# Train on Western English, test on ALL
	# Classifier accuracy: 87.08%
	# Classifier precision: 84.56%
	# Classifier recall: 95.04%

	# Train on ALL, test on ALL
	# Classifier accuracy: 93.78%
	# Classifier precision: 98.19%
	# Classifier recall: 94.22%



