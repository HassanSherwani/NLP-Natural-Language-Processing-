# Problem Statement

Aim of this project is to detect programming language from stack overflow dataset given by Kaggle.

It is an implementation of Information retrival. I have taken "Go" as case language. The reason is that "Go" can be used for many other meaings than a programming language. So challenge is to figure out correct entity as of "Go" in only programming context.

# Content

- Notebook a contains code while we are retrieving one programming language i.e go using text data
- Notebook b contains code while we are using multi-token for programming language i.e Objective-C++ (there are 4 tokens). Also this notebook contains code for more than one programming language i.e ruby, python, c, js etc.
- Notebook c: Evaluating our matcher while using labelled dataset. Matcher method with spacy and Machine learning algorithms have been implemented.

# Data

https://www.kaggle.com/stackoverflow/stacksample

# Modules

Spacy, re , pandas , string , sklearn

# Credit:

- Spacy Documentation:https://spacy.io/usage/linguistic-features
- Explosion:https://github.com/koaning/spacy-youtube-material
- Data Lebeling: https://www.cloudfactory.com/data-labeling-guide
