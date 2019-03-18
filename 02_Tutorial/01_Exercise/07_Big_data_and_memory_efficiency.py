
# coding: utf-8

# In[1]:


# Load the comma-separated data from https://query.data.world/s/
# wsjbxdqhw6z6izgdxijv5p2lfqh7gx into a DataFrame ‘.read csv()‘

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = "https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx"
df = pd.read_csv(DATA)


# In[2]:


# Inspect the DataFrame using .info() and with .info(memory\_usage="deep").
# What is the difference between the two calls? How much space does the 
# DataFrame require in memory?

df.info()
df.info(memory_usage="deep")

# second info shows much larger memory usage 
# default: memory estimation based in column dtype and number of 
    # rows assuming values consume the same memory amount for 
    # corresponding dtypes. 
# deep: real memory usage calculation is performed, 
    # cost of computational resources
# Memory space required: 861.6 MB


# In[3]:


# Create a copy of the object with only columns of type object by using
# .select\_dtypes (include=['object']).

df_obj = df.select_dtypes(include=['object'])
df_obj.head()


# In[5]:


# Look at the summary of this object new (using .describe()). Which 
# columns have very few unique values compared to the number of 
# observations?

df_obj.describe()


# In[6]:


# Does it make sense to convert a column of type object to type category
# if more than 50% of the observations contain unique values? Why/Why not?

columns_few = []

for col in df_obj.columns:
    if df_obj[col].nunique()/df_obj[col].count() <= 0.05:
        columns_few.append(col)

columns_few

type(columns_few)

# Converting a column to type category only makes sense when the feature 
# takes on very few unique values, in that case it saves memory 
# significantly if >50% of values are unique, category is not unseful.


# In[9]:


# Convert all columns of type object to type category where you deem this 
# appropriate.

# own criterion: if <10 unique values
for i in range(len(columns_few)):
    df_obj[columns_few[i]] = df_obj[columns_few[i]].astype('category')





# In[10]:


# What is the final size in memory?

# memory usage: 97.3 MB -> significant reduction
df_obj.info(memory_usage="deep") 


# In[ ]:


# Could above routine have speeded up somewhere? Hint: Look at the 
# documentation for .read csv().

# Speed up routine
# use dtype when reading the csv, change type to category


