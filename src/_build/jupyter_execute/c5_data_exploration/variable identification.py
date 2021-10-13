#!/usr/bin/env python
# coding: utf-8

# ## Variable Identification

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_theme()


# In[2]:


sns.set(rc={'figure.figsize':(16,12)})


# In[3]:


titanic_df = sns.load_dataset('titanic')
titanic_df.head()


# In[4]:


titanic_df.info()


# In[5]:


titanic_df.describe(include=['int64', 'float64'])


# In[6]:


titanic_df.describe(include=['category','object', 'bool'])


# In[7]:


ax = sns.histplot(data=titanic_df, x='age')

ax.axvline(titanic_df.age.mean(), color='cyan')
ax.axvline(titanic_df.age.median(), color='magenta')
ax.axvline(titanic_df.age.mode()[0], color='yellow')


# https://www.tableau.com/about/blog/examining-data-viz-rules-dont-use-red-green-together

# In[8]:


from sklearn.preprocessing import normalize


# In[9]:


titanic_age_df = pd.DataFrame(normalize(titanic_df[['age']].dropna()), columns=['age'])
print(titanic_age_df)
ax = sns.histplot(data=titanic_age_df, x='age', bins=10)

ax.axvline(titanic_age_df.age.mean(), color='cyan')
ax.axvline(titanic_age_df.age.median(), color='magenta')
ax.axvline(titanic_age_df.age.mode()[0], color='yellow')


# In[ ]:




