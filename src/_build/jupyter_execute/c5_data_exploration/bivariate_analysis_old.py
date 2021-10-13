#!/usr/bin/env python
# coding: utf-8

# ## Bivariate Analysis

# In[1]:


import pandas as pd
import seaborn as sns
from scipy import stats
sns.set_style()
sns.set(rc={'figure.figsize':(16,12)})


# In[2]:


titanic_df = sns.load_dataset('titanic')
titanic_df


# ### categorical vs categorical

# In[3]:


titanic_stacked_df = titanic_df.groupby(['embark_town', 'class']).survived.count().unstack('class')
titanic_stacked_df


# In[4]:


chi, p, dof, exp = stats.chi2_contingency(titanic_stacked_df, correction=True)


# In[5]:


print('the expected values if no bias was present')
pd.DataFrame(exp, index=titanic_stacked_df.index, columns= titanic_stacked_df.columns)


# In[6]:


p


# In[ ]:





# In[7]:


titanic_survived_stacked_df = titanic_df.groupby(['embark_town', 'class']).survived.sum().unstack('class')/titanic_stacked_df
titanic_survived_stacked_df.head()


# In[8]:


chi, p, dof, exp = stats.chi2_contingency(titanic_survived_stacked_df, correction=True)


# In[9]:


p


# In[10]:


print('the expected values if no bias was present')
pd.DataFrame(exp, index=titanic_stacked_df.index, columns= titanic_stacked_df.columns)


# ### categorical vs numerical

# In[11]:


t, p = stats.ttest_ind(
    titanic_df.fare[titanic_df.who=='man'],
    titanic_df.fare[titanic_df.who=='woman']
)
t, p


# In[12]:


titanic_df.fare[titanic_df.who=='man'].mean()


# In[13]:


titanic_df.fare[titanic_df.who=='woman'].mean()


# In[14]:


titanic_df.groupby(['who', 'class']).fare.mean().unstack('class')


# In[15]:


titanic_df.groupby(['who', 'class']).fare.count().unstack('class')


# In[16]:


F, p = stats.f_oneway(
    titanic_df.fare[titanic_df.pclass==1],
    titanic_df.fare[titanic_df.pclass==2],
    titanic_df.fare[titanic_df.pclass==3]
)
F, p


# In[17]:


F, p = stats.f_oneway(
    titanic_df.age[titanic_df.pclass==1].dropna(),
    titanic_df.age[titanic_df.pclass==2].dropna(),
    titanic_df.age[titanic_df.pclass==3].dropna()
)
F, p


# In[18]:


titanic_df.groupby('pclass').age.mean()


# In[19]:


titanic_df.groupby('survived').age.mean()


# can you find out if the age was relevant for survival?

# ### continuous vs continuous

# In[20]:


stats.spearmanr(a=titanic_df[['age','fare']].dropna())


# In[21]:


ax = sns.lmplot(data=titanic_df, x='age', y='fare')
ax.set(yscale='log')


# In[ ]:




