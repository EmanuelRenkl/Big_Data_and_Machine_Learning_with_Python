
# coding: utf-8

# In[7]:


# Read the count-matrix from exercise ”Speeches I” (./out/speech_matrix.pk)
# using pickle.load().

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

fl = open("02_Output/speech_matrix.pk", 'rb')
speech_matrix = pickle.load(fl)


# In[5]:


# Perform an LDA with 2 topics. How would you name these topics?

from sklearn.decomposition import LatentDirichletAllocation as LDA

N_TOPICS = 2
lda = LDA(n_components=N_TOPICS, learning_method='online')
topic_matrix = lda.fit_transform(speech_matrix)
keys = topic_matrix.argmax(axis=1).tolist()  # Clusters/Topics

print(keys)
print(topic_matrix)

