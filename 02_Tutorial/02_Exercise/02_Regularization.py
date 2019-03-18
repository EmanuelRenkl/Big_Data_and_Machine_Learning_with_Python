
# coding: utf-8

# In[16]:


# Read the data from exercise ”Feature Engineering” ./out/polynomials.csv 
# into a DataFrame

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FNAME = "02_Output/polynomials.csv"
df = pd.read_csv(FNAME, sep = ',')
df.head()


# In[17]:


# Use column ”y” as target variable and all other columns as predicting
# variables and split them as usual.

from sklearn.model_selection import train_test_split

X = df.loc[:, df.columns != 'y']
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[4]:


# Learn a Ridge and a Lasso  model using the provided data with default
# parameters. Why do you get a ConvergenceWarning? What are the accuracy 
# scores?


# In[18]:


# Ridge

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
preds = ridge.predict(X_test)

print(ridge.score(X_test, y_test))
print(ridge.score(X_train, y_train))


# In[19]:


# Lasso

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
preds = lasso.predict(X_test)

print(lasso.score(X_test, y_test))
print(lasso.score(X_train, y_train))


# In[20]:


# Why do you get a ConvergenceWarning?

# You migth want to increase the amount of iterations 
# A small alpha yields lower precision


# In[21]:


# Create a DataFrame containing the learned coefficients of both models
# and the feature names as index. In how many rows are the Lasso 
# coefficients equal to 0 while the Ridge coefficients are not?

coefs = pd.DataFrame(ridge.coef_, columns=["Ridge"], index=X.columns)
coefs["Lasso"] = lasso.coef_

coefs[(coefs["Lasso"] == 0) & (coefs["Ridge"] != 0)].shape[0]


# In[22]:


# Using matplotlib.pyplot, create a horizontal bar plot of dimension 10x30 
# showing the coefficient sizes

coefs.plot.barh(y=["Lasso", "Ridge"], figsize=(10, 30))

# Save the figure as ./out/polynomials.pdf.

plt.savefig('02_Output/polynomials.pdf')

