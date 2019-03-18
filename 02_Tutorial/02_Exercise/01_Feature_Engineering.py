
# coding: utf-8

# In[28]:


# Load the Boston House Price dataset using sklearn.datasets.load_boston()

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
print(boston["feature_names"])
df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
df.head()


# In[22]:


# Extract polynomial features (without bias) and interactions up to a 
# degree of 2.

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree =2, include_bias=False)
values = df.values.reshape(-13,13)
print(poly.fit_transform(values))
print(poly.get_feature_names())


# In[23]:


# How many features do you end up with?

len(poly.get_feature_names())


# In[36]:


# Create a pandas DataFrame using the polynomials.

# Use the originally provided feature names to generate names for the 
# polynomials (Hint: .get_feature_names() accepts a parameter) and 
# use them as column names.

columns = poly.get_feature_names(boston["feature_names"])

df2 = pd.DataFrame(poly.fit_transform(values), columns = columns)
df2.head()


# In[37]:


# Add the dependent variable to the dataframe and name it ”y”.

df2['y'] = boston['target']
df2.head()


# In[38]:


# Save the DataFrame as comma-separated textfile named 
# ./out/polynomials.csv.

df2.to_csv("02_Output/polynomials.csv")

