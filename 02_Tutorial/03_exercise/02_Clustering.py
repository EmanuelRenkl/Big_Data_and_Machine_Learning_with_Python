
# coding: utf-8

# In[2]:


# Load the iris dataset using sklearn.datasets.load_iris(). 
# The data is on classifying flowers.

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris = load_iris()
print(iris['DESCR'])


# In[3]:


# Scale the data such that each variable has unit variance.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = iris['data']
y = iris['target']

X_scaled = scaler.fit_transform(X)
X_scaled


# In[4]:


# Assume there are three clusters. Fit a K-Means model, an Agglomerative 
# Model and a DBSCN model with min sample equal to 2 and eps equal to 2.4.
# Store the cluster assignments in a DataFrame. 


# In[5]:


# Kmeans

from sklearn.cluster import KMeans  

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

df = pd.DataFrame(X, columns = iris['feature_names'])
df['cluster_kmeans'] = kmeans.labels_
df.head()


# In[6]:


# Agglomerative

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X_scaled)

df['cluster_agg'] = agg.labels_
df.head()


# In[7]:


# DBScan

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(min_samples=2, eps=1)
dbscan.fit(X_scaled)

df['cluster_dbscan'] = dbscan.labels_
df.head()


# In[8]:


# Read up about the Silhouette Score. What does it do? Compute the 
# silhouette scores using sklearn.metrics.silhouette_score() for each 
# cluster type versus the scaled features.

# The score is bounded between -1 for incorrect clustering and +1 for
# highly dense clustering. Scores around zero indicate overlapping 
# clusters. The score is higher when clusters are dense and well separated,
# which relates to a standard concept of a cluster.

from sklearn import metrics

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_scaled, kmeans.labels_))

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_scaled, agg.labels_))

# To compute the Silhouette Score for DCSCAN clsutering one needs to get 
# rid of the noise

df.loc[df["cluster_dbscan"]==-1] # Identifying the noise

df_scaled=pd.DataFrame(X_scaled)
df_scaled.drop([41,], inplace = True) # Deleting the noise

labels = pd.DataFrame(dbscan.labels_)
labels.drop([41,], inplace = True) # Deleting the noise

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(df_scaled, labels))


# In[9]:


# Add sepal width and petal length including the corresponding original
# names to the DataFrame. (Beware of the dimensionality)

# Already included


# In[12]:


# Plot a three-fold scatter plot using sepal width as x-variable and petal
# length as y-variable, with dots colored by the cluster assignment and 
# facets by cluster algorithm. (Hint: Melt the DataFrame using columns
# ”sepal length (cm)” and ”sepal width (cm)” as id variables.)

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = df.melt(id_vars=["sepal length (cm)","sepal width (cm)", "petal width (cm)",
                      "petal length (cm)", ], var_name = "cluster method", value_name="cluster")

g = sns.FacetGrid(df, hue="cluster method", col = "cluster")
g = g.map(plt.scatter, "sepal width (cm)", "petal length (cm)")

