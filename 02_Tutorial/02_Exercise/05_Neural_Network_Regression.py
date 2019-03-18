
# coding: utf-8

# In[17]:


# Load the diabetes dataset using sklearn.datasets.load_diabetes(). 
# The data is on health and diabetes of 442 patients. Split the data 
# as usual.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
print(diabetes.keys())

print(diabetes['DESCR'])
df = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
df.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes['data'], diabetes['target'])


# In[18]:


# Learn a a Neural Network with 1000 iterations, lbfgs-solver and 
# tanh-activation after Standard-Scaling with in total nine parameter 
# combinations of your choice using 4-fold Cross-Validation.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([("scalar",StandardScaler()),
                 ("nn",MLPRegressor(activation = 'tanh',
                                    max_iter=1000, solver="lbfgs"))])
param_grid = {"nn__hidden_layer_sizes": [(20,),(10,),(50,)],
              "nn__alpha": [0.00001, 0.0001, 0.001]}
grid = GridSearchCV(pipe, param_grid, cv=4, return_train_score=True)
grid.fit(X_train, y_train)


# In[19]:


# What are your best parameters? How well does it perform?

results = pd.DataFrame(grid.cv_results_)
results

from numpy import array

scores = array(results["mean_test_score"]).reshape(3,3)
scores


# In[23]:


# Plot a heat map showing the coefficients with the irvariable names and 
# save it as./out/nn_diabetes_importances.p (Hint: the best estimator has 
# a property _final_estimator).

fig = sns.heatmap(scores, annot=True,
           xticklabels=param_grid["nn__hidden_layer_sizes"],
           yticklabels=param_grid["nn__alpha"])

plt.savefig("02_Output/nn_diabetes_importances.pdf")

print(grid.best_params_)
print(grid.best_score_)

best = grid.best_estimator_
best._final_estimator

