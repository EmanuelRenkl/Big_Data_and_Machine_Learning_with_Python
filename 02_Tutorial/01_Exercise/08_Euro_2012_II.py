
# coding: utf-8

# In[6]:


# Load the data from ./data/Euro_2012.csv into a DataFrame.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import wikipedia

df = pd.read_csv("01_Data/Euro_2012.csv")
df.head()


# In[8]:


# Add a column ”Wikipedia” displaying the Wikipedia page ID of the 
# country (use .apply() on column ”Team” to apply the corresponding 
# function which queries Wikipedia individually).

df["Wikipedia"] = df["Team"].apply(lambda x: wikipedia.page(wikipedia.search(x)).pageid)
df["Wikipedia"]


# In[9]:


# Output the data as semicolon-separated ./out/wikipedia.ssv.

df.to_csv(path_or_buf="02_Output/wikipedia.ssv", sep=";")

