
# coding: utf-8

# In[1]:


# Read the Iris dataset fromhttps://archive.ics.uci.edu/ml/machine-learning
# -databases/iris/iris. data directly from the Internet.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FNAME = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# In[3]:


# Name the columns in the following way: ”sepal length (in cm)”, ”sepal width (in cm)”, 
# ”petal length (in cm)”, ”petal width (in cm)” and ”class”.

df = pd.read_csv(FNAME, header=None, names=['sepal length (in cm)','sepal_width (in cm)','petal_length (in cm)','petal_width (in cm)','class'])
df.head()


# In[4]:


# Set values of the rows 10 to 29 of the column ’petal length (in cm)’ 
# to missing.

df.loc[10:29,'petal_length (in cm)'] = np.nan
df.loc[10:29,'petal_length (in cm)']


# In[5]:


# Replace missing values with 1.0.

df = df.fillna(1.0)
df.loc[10:29,'petal_length (in cm)']


# In[7]:


# Save the comma-separated file as ./out/iris.csv without index.

df.to_csv(path_or_buf="02_Output/iris.csv", sep=",")


# In[8]:


# Visualize the distribution of all of the continuous variables by 
# ”class” with a catplot of your choice.

melted = df.melt(id_vars="class", var_name="var", value_vars=['sepal length (in cm)','sepal_width (in cm)','petal_length (in cm)','petal_width (in cm)'])
ax = sns.catplot(x='var', y='value', col='class', data=melted, kind='violin')

# Save the figure as ./out/iris.pdf.

ax.savefig("02_Output/iris.pdf")

