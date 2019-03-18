
# coding: utf-8

# In[3]:


# Read the count-matrix from exercise ”Speeches I” (./out/speech_
# matrix.pk) using pickle.load().

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

fl = open("02_Output/speech_matrix.pk", 'rb')
speech_matrix = pickle.load(fl)


# In[6]:


# Using the matrix, create a dendrogram of hierarchical clustering. 
# Remove the x-ticks from the plot.

from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(speech_matrix.toarray())
plt.subplots(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.xticks([])

# Save the dendrogram as ./out/speeches_dendrogram.pdf.

plt.savefig("02_Output/speeches_dendogram.pdf")

