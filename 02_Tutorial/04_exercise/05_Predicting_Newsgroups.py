
# coding: utf-8

# In[22]:


# Use sklearn.datasets.fetch_20newsgroups() to load data for the 
# following newsgroups: sci.crypt, sci.electronics, sci.med, and sci.
# space (will be slow the first time!). Make sure to shuffle the data.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(shuffle=True, categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'])


# In[23]:


# Use dir() to understand the object you just used. What is the data about?

dir(news)


# In[24]:


# Print the label names corresponding to the first 20 documents to verify
# the data is not ordered.

print(news['DESCR'])
print(news['target_names'])


# In[25]:


# Read up on Multinomial Naive Bayes (MultiNB) classifier. How does 
# MultiNB work, what assumptions does it make? Is it well suited for 
# text analysis?

# MultinomialNB implements the naive Bayes algorithm for multinomially 
# distributed data, and is one of the two classic naive Bayes variants 
# used in text classification. Thus, the multinomial Naive Bayes classifier 
# is suitable for text classification.


# In[27]:


# Build a pipeline consisting of a TfidfVectorizer (with english stopwords
# provided by nltk and the stem- ming/tokenziation function provided in 
# class.

import nltk

_stopwords = nltk.corpus.stopwords.words('english')

from string import digits, punctuation

remove = digits + punctuation
_stemmer = nltk.snowball.SnowballStemmer('english')


# In[28]:


def tokenize_and_stem(text):
    """Return tokens of document deprived of numbers and interpunctuation."""
    text = text.translate(str.maketrans({p: "" for p in remove}))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]


# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([('vecotrizer', TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)), ('nn', MultinomialNB())])


# In[30]:


# Perform a grid search with 2-fold cross-validation and two different 
# values for alpha. (It may make sense to use multiple cores using the 
# parameter njobs.)Whatareyourbestparameters?

param_grid = {'nn__alpha': [0.00001, 0.001]}
grid = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_)


# In[31]:


# Load the test set using sklearn.datasets.fetch_20newsgroups(subset="test").

news_test = fetch_20newsgroups(categories=news["target_names"], subset="test")

X_test = news_test['data']
y_test = news_test['target']


# In[32]:


# Use the best estimator to predict on the test set.

preds = grid.predict(X_test)


# In[33]:


# Print the classification report for the test set.

from sklearn.metrics import classification_report
report = classification_report(y_test, preds,
                               target_names=news_test["target_names"])
print(report)

