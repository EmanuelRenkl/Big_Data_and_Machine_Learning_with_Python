
# coding: utf-8

# In[2]:


# Read up about glob.glob. Use it to read the text files in ./data/
# speeches into a corpus (i.e. a list of strings). The files represent 
# a non-random selection of speeches of central bankers, which have 
# already been stripped off meta information.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob

data = glob.glob('01_Data/speeches/*',recursive=False)
data


# In[8]:


# Vectorize the speeches using tfidf using up 1-grams, 2-grams and 
# 3-grams while removing English stop- words and proper tokenization 
# (i.e., you create a Count matrix).

text=[]
for name in data:
    try:
        with open(name, 'r') as inf:
            text.extend(inf.readlines())
        text
    except:
        print(name)
        
len(text)


# In[9]:


import nltk

_stopwords = nltk.corpus.stopwords.words('english')
_stopwords

from string import digits, punctuation

remove = digits + punctuation
remove

_stemmer = nltk.snowball.SnowballStemmer('english')


# In[11]:


def tokenize_and_stem(text):
    """Return tokens of document deprived of numbers and interpunctuation."""
    text = text.translate(str.maketrans({p: "" for p in remove}))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)
count.fit(text)


# In[13]:


count_matrix = count.transform(text)
count_matrix   


# In[16]:


terms = count.get_feature_names()
print(terms[:10])


# In[17]:


df = pd.DataFrame(count_matrix.toarray().T, index=terms)
df.head()


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf.fit_transform(text)
tfidf_terms = tfidf.get_feature_names()

df_tfidf = pd.DataFrame(tfidf_matrix.toarray().T, index=tfidf_terms)
df_tfidf.tail()


# In[28]:


# Pickle the resulting sparse matrix using pickle.dump() as ./out/
# speech_matrix.pk.

import pickle

file = open("02_output/speech_matrix.pk", 'wb')
pickle.dump(tfidf_matrix, file)
file.close()

fl = open("02_output/speech_matrix.pk", 'rb')
speech_matrix = pickle.load(fl)

