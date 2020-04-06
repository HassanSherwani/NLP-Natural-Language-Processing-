# Problem Statement

We aim to extract key information from text data. Information is in form of entities.

For this, we need to create customized Entity Ruler pipelines using pre-trained Spacy model.

# Results

We have used visualization for showing outcome of our model by using Displacy. 

For evaluating our mode, we have used beam confidence score.

# Content

- Custom_NER1 : Using EntityRuler() for each entity.
- Custom_NER2 : Using rulerAll() method covering each entity.
- Custom_NER3 : Creating table with Entity Class and Entity Type.
- Custom_NER4 : Add custom colors and Visualization.
- Custom_NER5 : Training on customized entities using Spacy's pre-trained model.
- Add custom entities in form of table
- Custom_NER6 : Custom_ner_kaggle : Implementing Named Entity Recognition Using Kaggle Sentiment Analysis dataset. 
https://www.kaggle.com/davidg089/all-djtrum-tweets


- Confidence Score 1 : Confidence Score for Entities using default Spacy + beam search algorithm.
- Confidence Score 2 : For Customized Entities using beam search algorithm.
- Confidence Score 3 : For Customized Entities with better labelling list and hence, improved results


# Modules

Spacy, pandas,numpy

# References


- https://www.kaggle.com/nirant/hitchhiker-s-guide-to-nlp-in-spacy/data
- https://support.prodi.gy/t/accessing-probabilities-in-ner/94
- https://www.kaggle.com/davidg089/all-djtrum-tweets
- https://spacy.io/usage/rule-based-matching
- https://spacy.io/usage/examples
- Multiple Rulers: https://stackoverflow.com/questions/57536044/add-multiple-entityruler-with-spacy-valueerror-entity-ruler-already-exists-i

