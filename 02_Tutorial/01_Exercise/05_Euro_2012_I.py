
# coding: utf-8

# In[30]:


# Read the data from ./data/Euro_2012.csv into a DataFrame with column 
# ”Teams” as index. The data is on the UEFA Champtionship 2012 (Euro 2012).

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("01_Data/Euro_2012.csv", index_col="Team")
df.head()


# In[31]:


# How many teams played in the Euro 2012?

df.shape # 16 Teams


# In[32]:


# Which team has the highest shooting accuracy?

df['Shooting Accuracy'] = df['Shooting Accuracy'].str.rstrip('%').astype(
    'float') / 100.0 # convert %String to float
df["Passing Accuracy"] = df["Passing Accuracy"].str.rstrip('%').astype(
    'float') / 100.0
max_acc = df['Shooting Accuracy'].idxmax()
print("Team with highest Shooting Accuracy: %s" % max_acc)


# In[33]:


# Plot shooting accuracy versus passing accuracy.

df[['Shooting Accuracy', 'Passing Accuracy']].plot.bar()


# In[34]:


# Which team has the second-most shots on target?

df['Shots on target'].sort_values() 
df['Shots on target'].sort_values().shift(-1).idxmax() # Italy


# In[35]:


# Eliminate Italy from the dataset. Which team has the second-most 
# shots on target now?

df['Shots on target'].drop(index='Italy').sort_values().shift(-1).idxmax()
# Germany


# In[36]:


# How many penalty goals did England score?

df['Penalty goals']['England']


# In[37]:


# Present only the Shooting Accuracy from England, Italy and Russia.

df['Shooting Accuracy'][['England', 'Russia', 'Italy']]


# In[38]:


# Create a new DataFrame called discipline using the columns ”Yellow 
# Cards” and ”Red Cards” (and the index).

discipline = df[['Yellow Cards', 'Red Cards']]
discipline.head()


# In[39]:


# Sort discipline primarily by red cars and secondarily by yellow cards.

discipline.sort_values(by=['Red Cards', 'Yellow Cards'])


# In[40]:


# Output the data as tab-separated textfile ./out/discipline.tsv.

df.to_csv(path_or_buf="02_Output/discipline.tsv", sep="\t")

