
# coding: utf-8

# In[1]:


# Read the textfile ./data/olympics.csv into a DataFrame using the first 
# column as index. The data lists the individual performances of 33 male 
# athlets during various competitions of the 1988 Olympic summer games.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OLYMP = "01_Data/olympics.csv"
olympics = pd.read_csv(OLYMP, sep=",")
olympics = olympics.set_index("id")


# In[2]:


# Print summary statistics for each of the variables.

olympics.describe()


# In[4]:


# Scale the data such that all variables have unit variance.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
olympics_scaled = scaler.fit_transform(olympics)

print(olympics_scaled)


# In[ ]:


# Plain PCA model


# In[14]:


# Fit a plain vanilla PCA model. Store the components in a DataFrame to 
# display the loadings of each variable. 

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(olympics_scaled)

components = pd.DataFrame(pca.components_, columns=[olympics.columns])
components.index = components.index + 1
components


# In[17]:


# Which variables load most prominently on the first component? Which ones 
# on the second? Which ones on the third? How would you thus interpret 
# those components?

components.max(axis=1)

# Variables that load most prominently: 
    # first component: 110
    # second: disq
    # third: haut
    
# sns.heatmap(pca.components_, cmap="viridis")


# In[10]:


# How many components do you need to explain at least 90% of the data? 
# (Hint: use np.cumsum() for this.)

var = pd.DataFrame(pca.explained_variance_ratio_, 
                   columns=["Explained Variance"])
var.index = var.index + 1 
var["Cum. explained variance"] = var["Explained Variance"].cumsum()
var # to explain at leat 90% of the data, you need 7 components.

