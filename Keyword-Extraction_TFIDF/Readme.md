# Problem Statement

Extracting keywords from text document by applying TF-IDF method.

# Content

- Notebook1: using bag of word + TFIDF
- Notebook2: applying tfidf and getting results evaluated titles, abstract, main text.
- Notebook3: Keeping TFIDF as main method but, Other techniques such as gensim TextRank summarization,RAKE(Rapid Automatic Keyword Extraction) ,Yet Another Keyword Extractor (Yake) ,python-based keyphrase extraction (pke), were explored.

# TF-IDF

TF-IDF stands for term frequency–inverse document frequency, a formula that measures how important a word is to a document in a collection of documents.

This metric calculates the number of times a word appears in a text (term frequency) and compares it with the inverse document frequency (how rare or common that word is in the entire data set).

Multiplying these two quantities provides the TF-IDF score of a word in a document. The higher the score is, the more relevant the word is to the document.

# Dataset

https://www.kaggle.com/benhamner/nips-papers/data

# Modules

# References

- http://datameetsmedia.com/bag-of-words-tf-idf-explained/

- https://monkeylearn.com/keyword-extraction/

- https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

- https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34

- https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.XfdsYZNKjUI

- https://www.kaggle.com/akhatova/extract-keywords

- For yake: https://github.com/LIAAD/yake

- gensim summarization: https://radimrehurek.com/gensim/summarization/keywords.html

- gensim tutorial: https://rare-technologies.com/text-summarization-with-gensim/
