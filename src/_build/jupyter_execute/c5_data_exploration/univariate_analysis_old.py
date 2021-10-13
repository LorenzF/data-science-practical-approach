#!/usr/bin/env python
# coding: utf-8

# ## Univariate Analysis

# In[1]:


import pandas as pd
import seaborn as sns
sns.set_style()
sns.set(rc={'figure.figsize':(16,12)})


# In[2]:


titanic_df = sns.load_dataset('titanic')
titanic_df.head()


# ### Nominal Data

# In[3]:


nominal_titanic_df = titanic_df[['sex', 'deck' , 'embark_town' ,'alone', 'who']]
nominal_titanic_df.head()


# In[4]:


nominal_titanic_df.describe()


# In[5]:


for name, col in nominal_titanic_df.iteritems():
    print(name)
    print(col.value_counts())
    print()


# In[6]:


nominal_titanic_df.who.value_counts().plot.pie()


# In[7]:


nominal_titanic_df.embark_town.value_counts().plot.bar()


# ### Ordinal data

# In[8]:


titanic_df.head()


# In[9]:


ordinal_titanic_df = titanic_df[['class', 'alive', 'embark_town' , 'sex', 'who', 'deck', 'alone']]
ordinal_titanic_df


# In[10]:


ordinal_titanic_df.describe()


# In[11]:


ordinal_titanic_df['class'].value_counts()


# In[12]:


ordinal_titanic_df['class'].value_counts()[['First', 'Second', 'Third']].plot()


# In[13]:


ordinal_titanic_df['class'].groupby(ordinal_titanic_df.alive).apply(lambda x: x.value_counts()).unstack(0).reindex(['First', 'Second', 'Third']).plot()


# ### Numerical data

# In[14]:


numerical_titanic_df = titanic_df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']]
numerical_titanic_df.head()


# In[15]:


numerical_titanic_df.describe()


# In[16]:


numerical_titanic_df.fare.plot.hist()


# In[17]:


print('median')
print(numerical_titanic_df.fare.median())
print('mean')
print(numerical_titanic_df.fare.mean())


# In[18]:


numerical_titanic_outliers_df = numerical_titanic_df.fare[numerical_titanic_df.fare<50]
numerical_titanic_outliers_df.plot.hist()


# In[19]:


print('median')
print(numerical_titanic_outliers_df.median())
print('mean')
print(numerical_titanic_outliers_df.mean())


# In[ ]:




