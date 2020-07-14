#Get modules

# support both Python 2 and Python 3 with minimal overhead.
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')
#For data analysis
import re    # for regular expressions
import nltk  # for text manipulation
import string
import numpy as np
import pandas as pd
#For Visuals
import matplotlib.pyplot as plt
#models and evaluation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier # notice its from ntlk not sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# Evaluation packages
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Load data
df_intent2=pd.read_excel('intent_5classes.xlsx')
df_intent2=df_intent2.rename(columns={'Unnamed: 0':'random_columns'}) # a trick to tackle random index values
df_intent2= df_intent2.drop('random_columns', axis=1)
df_intent2['first_text']= df_intent2['first_text'].str[:300]

#Data Cleaning
contractions = {
"ain't": "am not ",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"

}
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def preprocess(text):
    text = text.str.replace("(<br>)", "")
    text = text.str.replace("(<br />)", "")
    text = text.str.replace('(<p>)', '')
    text = text.str.replace('( </p>)', '')
    text = text.str.replace('(;</p>)', '')
    text = text.str.replace('(&rsquo;)', '')
    text = text.str.replace('(&rdquo;)', '')
    text = text.str.replace('(&ldquo;)', '')
    text = text.str.replace('(&nbsp)', ' ')
    text = text.str.replace('(www)', ' ')
    text = text.str.replace('(https)', ' ')
    text = text.str.replace('(http)', ' ')
    return text
df_intent2["clean"] = preprocess(df_intent2['first_text'])
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_]')
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['http','https']) # extend stopwords; rt means re-tweet
STOPWORDS = set(STOPWORDS)

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS]) # delete stopwords from text
    text = text.strip()
    return text
df_intent2['clean']=[text_prepare(x) for x in df_intent2['clean']]
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
tokenized_text = df_intent2['clean'].apply(lambda x: x.split()) # tokenizing
tokenized_text = tokenized_text.apply(lambda x: [stemmer.stem(i) for i in x])
# stitch these tokens back together.
for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])
df_intent2['clean'] = tokenized_text

#Function for category id
df_intent2['category_id'] = df_intent2['dep'].factorize()[0]
category_id_df = df_intent2[['dep', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'dep']].values)

#vectorization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_intent = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

#1. Intent Model
features = tfidf_intent.fit_transform(df_intent2.clean).toarray()
labels = df_intent2['dep'].astype(str)
model_intent = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df_intent2.index, test_size=0.33, random_state=0)
model_intent.fit(X_train, y_train)
y_pred_intent = model_intent.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn import metrics
print("Accuracy Score for Intent Model is :", accuracy_score(y_test, y_pred_intent))
#print(metrics.classification_report(y_test, y_pred))

#API testing
texts=["I was asked to test a saal photobook and I was so delighted with the result! It arrived with in 10 days and was of such high quality, with a white leather look cover and an acrylic glass to protect the front photo. It has made a lovely lockdown gift for my best friend."]
text_features = tfidf_intent.transform(texts)
pred_class_int = model_intent.predict(text_features)
print('"{}"'.format(texts))
print("  - Predicted as: '{}'".format(pred_class_int))
prob_score_int=pd.DataFrame(model_intent._predict_proba_lr(text_features), columns=model_intent.classes_)
print("Probability distribution for intent model is:",prob_score_int.to_json(orient='records'))

#2. Response Model
df_response2=pd.read_excel('response_5classes.xlsx')
df_response2=df_response2.rename(columns={'Unnamed: 0':'random_columns'}) # a trick to tackle random index values
df_response2['first_text']= df_response2['first_text'].str[:300]
df_response2= df_response2.drop('random_columns', axis=1)

# clean
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
def preprocess(text):
    text = text.str.replace("(<br>)", "")
    text = text.str.replace("(<br />)", "")
    text = text.str.replace('(<p>)', '')
    text = text.str.replace('( </p>)', '')
    text = text.str.replace('(;</p>)', '')
    text = text.str.replace('(&rsquo;)', '')
    text = text.str.replace('(&rdquo;)', '')
    text = text.str.replace('(&ldquo;)', '')
    text = text.str.replace('(&nbsp)', ' ')
    text = text.str.replace('(www)', ' ')
    text = text.str.replace('(https)', ' ')
    text = text.str.replace('(http)', ' ')
    return text
df_response2['clean'] = preprocess(df_response2['first_text'])
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_]')
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['http', 'https'])  # extend stopwords; rt means re-tweet
STOPWORDS = set(STOPWORDS)
def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])  # delete stopwords from text
    text = text.strip()
    return text
df_response2['clean'] = [text_prepare(x) for x in df_response2['clean']]
#from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
tokenized_text = df_response2['clean'].apply(lambda x: x.split()) # tokenizing
tokenized_text = tokenized_text.apply(lambda x: [stemmer.stem(i) for i in x])
# stitch these tokens back together.
for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])
df_response2['clean'] = tokenized_text

#Function for category id
df_response2['category_id'] = df_response2['firstusedtextblock'].factorize()[0]
category_id_df = df_response2[['firstusedtextblock', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'firstusedtextblock']].values)

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_response = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf_response.fit_transform(df_response2.clean).toarray()
labels = df_response2['firstusedtextblock']
model_response = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df_response2.index, test_size=0.33, random_state=0)
model_response.fit(X_train, y_train)
y_pred_res = model_response.predict(X_test)
from sklearn.metrics import accuracy_score
print("")
print("Accuracy Score for Response Model is :" , accuracy_score(y_test, y_pred_res))

#API testing
texts=["I was asked to test a saal photobook and I was so delighted with the result! It arrived with in 10 days and was of such high quality, with a white leather look cover and an acrylic glass to protect the front photo. It has made a lovely lockdown gift for my best friend."]
text_features = tfidf_response.transform(texts)
pred_class_res = model_response.predict(text_features)
print('"{}"'.format(texts))
print("  - Predicted as: '{}'".format(pred_class_res))
prob_score_res=pd.DataFrame(model_response._predict_proba_lr(text_features), columns=model_response.classes_)
print("Probability distribution for response model is:", prob_score_res.to_json(orient='records'))