
# coding: utf-8

# In[15]:


# Load seaborn’s tips dataset using seaborn.load_dataset("iris").

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TIPS = sns.load_dataset("tips")


# In[12]:


# Plot ”tips” on ”total bill” with markers, with line styles and color 
# by ”day”, and facets by ”sex”.

g = sns.relplot(x='total_bill', y='tip', data=TIPS, kind="line", hue="day",
                col = "sex", marker=True)


# In[13]:


# Label the axis so that the unit becomes apparent.

g = sns.relplot(x='total_bill', y='tip', data=TIPS, kind="line", hue="day",
                col = "sex", marker=True)
g.set(xlabel='Total bill (cost of the meal), including tax, in US dollars',
      ylabel='Tip (gratuity) in US dollars')


# In[14]:


# Add a title to the legend and use the full day of the week-name 
# (i.e. ”Thursday” instead of ”Thu”).

g = sns.relplot(x='total_bill', y='tip', data=TIPS, kind="line", hue="day",
                col = "sex", marker=True)
g.set(xlabel='Total bill (cost of the meal), including tax, in US dollars',
      ylabel='Tip (gratuity) in US dollars')
g._legend.set_title("Legend")
g._legend.texts[1].set_text("Thursday")
g._legend.texts[2].set_text("Friday")
g._legend.texts[3].set_text("Saturday")
g._legend.texts[4].set_text("Sunday")

# Save the figure as ./out/tips.pdf

plt.savefig('02_Output/tips.pdf')

