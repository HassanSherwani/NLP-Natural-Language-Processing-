# Spam-Detection-Text-Classifier

classify spam versus non spam SMS

# Method- Text Classifier

Text classification is the process of assigning tags or categories to text according to its content. It’s one of the fundamental tasks in Natural Language Processing (NLP) with broad applications such as sentiment analysis, topic labeling, spam detection, and intent detection(fraud, hate speech, trolling etc).

# Dataset
Data source :https://archive.ics.uci.edu/ml/machine-learning-databases/00228/ 

The files contain one message per line. Each line is composed by two columns: one with label (ham or spam) and other with the raw text.

# Content
 
This is an end to end project that will keep NLP pipeline for text data in focus. So, it consists of following Jupyter Notebooks;

- 1.Introduction to dataset : Loading data, Data exploration, visualization and checking most frequent words' cloud
- 2.Cleaning Data: Using text preprocessing techniques (Lower casing, Punctuation removal,Stopwords removal,Frequent words removal,Rare words removal,Spelling correction,Tokenization,Stemming,Lemmatization)
- 3.Basic Model: Applying Logistic classification model plus model evaluation
- 4.Vectorize+Model:Performing different types of vectorization to check which one performs better four techniques for vectorization(Bag of words,TF-IDF,Word2Vec Embedding,Doc2Vec Embedding)
- 5.Applying all four vectorize techniques with multiple machine learning algorithms to choose best tested model
- 6.Applying recurrent neural network for text classification by using keras and tensorflow


# Modules
pandas <br>
numpy <br>
matplotlib.pyplot <br>
nltk <br>
re <br>
sklearn <br>
tensorflow <br>
keras <br>
