
# coding: utf-8

# In[1]:


# Importthepipe-separateddatasetfromhttps://raw.githubusercontent.com/
# justmarkham/DAT8/master/ data/u.user into a DataFrame. The data is 
# on occupations and demographic information.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OCCUPATIONS = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"
df = pd.read_csv(OCCUPATIONS, sep='|')


# In[2]:


# Set ”user id” as index and name the index ”User”.

df = df.set_index("user_id")
df.index.names = ["User"]


# In[3]:


#Print the last 10 entries and the first 25 entries.

df.tail(10)
df.head(25)


# In[4]:


# What is the type of each column?

df.dtypes # or df.info()


# In[6]:


# How many different occupations are there in the dataset? What is the 
# most frequent occupation?

df["occupation"].nunique() # number of occupations is 21
df["occupation"].value_counts() 
print(df["occupation"].value_counts().index[0]) # most frequent occupation is student


# In[7]:


# What is the age with the least occurrence?

counts = df['age'].value_counts()
min_value = counts.min()
min_value
df["age"].value_counts() # least frequent ages are 7,10,11,66,73 


# In[12]:


# Create a histogram for occupations, sorted alphabetically.

df = df.sort_values(by='occupation')
df.plot.hist("occupation")
# df["occupation"].value_counts().sort.index.boxplot
# df['occupation'].value_counts().plot(kind='barh')

# Save the figure as ./out/occupations.pdf

plt.savefig('02_Output/occupations.pdf')

