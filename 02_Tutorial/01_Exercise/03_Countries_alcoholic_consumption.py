
# coding: utf-8

# In[3]:


# Load the data from ./data/drink.csv into a DataFrame. The data is on 
# country’s alcoholic consumption.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FNAME = "01_Data/drinks.csv"
df = pd.read_csv(FNAME, sep = ',')
df.head()


# In[4]:


# Which continent drinks most beer on average?

grouped = df.groupby(["continent"])
print(grouped["beer_servings"].mean()) # EU drinks most beer on average


# In[5]:


# Which continent drinks most wine on average? 

print(grouped["wine_servings"].mean()) # EU drinks most wine on average


# In[8]:


#  Create a Boxenplot of ”beer servings” by continent.

fig = sns.catplot(x = "continent", y = "beer_servings", data = df, 
                  kind= "boxen")


# In[11]:


# Reshape the data and create a 3x1 figure for ”beer servings”, ”wine 
# servings” and ”spirit servings” by continent (i.e. three boxenplots 
# that share their y-axis and a their color coding).

melted = df.melt(id_vars="continent", var_name="alcohol", 
                 value_vars=['beer_servings', 'spirit_servings',
                             'wine_servings'])
ax = sns.catplot(x='continent', y='value', col='alcohol', data=melted, 
                 kind='boxen')


# Save the figure as ./out/alcohol.pdf

ax.savefig("02_Output/alcohol.pdf")

