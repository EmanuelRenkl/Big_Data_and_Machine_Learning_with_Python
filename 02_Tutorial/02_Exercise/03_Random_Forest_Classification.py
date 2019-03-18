
# coding: utf-8

# In[1]:


# Load the breast cancer dataset using sklearn.datasets.load_breast_cancer().

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
CANCER = load_breast_cancer()


# In[4]:


# Create a histogram for the target variable.

pd.DataFrame(CANCER['target']).plot.hist()


# In[6]:


# Split the data as usual into a test and training set.

from sklearn.model_selection import train_test_split

X = CANCER["data"]
y = CANCER["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[10]:


# Learn a Random Forest model for the following parameters using 5-fold 
# Cross-Validation (i.e. GridSearchCV(): 10 estimators, 50 estimators, 
# 100 estimators (a pipeline is not necessary). What is the best parameter?

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

param_grid = {"n_estimators": [10, 50, 100]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid)
grid_search.fit(X, y)

print(grid_search.best_params_)


# In[12]:


# Print the confusion matrix for the best model to screen. Does it look 
# good?

from sklearn.metrics import confusion_matrix

preds = grid_search.predict(X_test) 
confusion_m = confusion_matrix(y_test, preds.astype(int))

import seaborn as sns

sns.heatmap(confusion_m, annot=True)


# In[13]:


# For the best model, print accuracy score, precision, recall and f1-score.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(accuracy_score(y_test, preds.astype(int)))
print(precision_score(y_test, preds.astype(int)))
print(recall_score(y_test, preds.astype(int)))
print(f1_score(y_test, preds.astype(int)))


# In[18]:


# Plot the feature importances as as simple barplot(with axis names) and 
# save it as ./out/forest_breast_importances.pdf

features = pd.DataFrame({'importances': grid_search.best_estimator_.feature_importances_},
                         index=CANCER['feature_names'])
fig = features.plot.bar()

plt.savefig("02_Output/forest_breast_importances.pdf")

