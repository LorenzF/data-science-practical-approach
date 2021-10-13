#!/usr/bin/env python
# coding: utf-8

# ## Variable identification
# 
# in this notebook we are going to look into a few simple but interesting techniques about getting to know more about what is inside the dataset you are given. Whenever you start out on a new project these steps are usually the first that are performed in order to know how to proceed.
# 
# We start out by loading the titanic dataset from seaborn

# In[1]:


import seaborn as sns
titanic_df = sns.load_dataset('titanic')
sns.set_theme()
sns.set(rc={'figure.figsize':(16,12)})


# ### description
# 
# Let us start out simple and retrieve information about each column, using the .info method we can get non-null counts (giving us an idea if there are nans) and the type of each column (to see if we need to change types).

# In[2]:


titanic_df.info()


# it looks like all types are already correctlyaddressed, but we can see a lot of nans are present for age and deck, this might be a problem!
# 
# For numerical columns we can get a bunch of information using the .describe method. this can also be used for categories but has less info

# In[3]:


titanic_df.describe()


# In[4]:


titanic_df.describe(include=['category', 'object'])


# ### Uniques, frequencies and ranges
# 
# the describe method is a bit lacklusting for categorical features, so we use some good old data wrangling to get more info, asking for unique values gives us all the possible values for a column. Aside from the uniques, we can also  get the value counts or frequencies and the range of a column.

# In[5]:


titanic_df['embark_town'].unique()


# In[6]:


titanic_df['embark_town'].value_counts()


# In[7]:


titanic_df['age'].min(), titanic_df['age'].max()


# ### mean and deviation
# 
# to get more information about a numerical range, we calculate the mean and deviation. Note that these statistics imply that our column is normally distributed!
# 
# You can also see that I applied the dropna method, this because the calculations cannot handle nan values, but this means our outcome might be distorted from the truth, thread carefuly.

# In[8]:


import statistics


# In[9]:


titanic_df['age'].dropna().mean()


# In[10]:


titanic_df['age'].dropna().median()


# ### median and interquantile range
# 
# When our distribution is not normal, using the median and IQR is advised.
# First we apply the shapiro wilk test and it has a very low p-value (the second value) which means we can reject the null-hypothesis that there is a normal distribution. more info about shapiro-wilk can be found on [wikipedia](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)

# In[11]:


from scipy.stats import shapiro
shapiro(titanic_df['age'].dropna())


# In[12]:


titanic_df['age'].dropna().median()


# In[13]:


from scipy.stats import iqr
iqr(titanic_df['age'].dropna())


# In[14]:


from scipy.stats.mstats import mquantiles
mquantiles(titanic_df['age'].dropna())


# Appearently the average of 29.70 is fairly higher than the median at 28, meaning that there is a shift towards older people.
# You can also see this on the following plot, where we note the mean, median and mode.

# In[15]:


ax = sns.histplot(data=titanic_df, x='age')

ax.axvline(titanic_df.age.mean(), color='cyan')
ax.axvline(titanic_df.age.median(), color='magenta')
ax.axvline(titanic_df.age.mode()[0], color='yellow')


# ### modes and frequencies
# 
# When we don't have numerical data we can still find some interesting results, here we use the mode ( most frequent value) and the proporties of each value to deduce the proporties of people that embarked in the 3 different towns. Nearly 3/4 people embarked in one harbour.

# In[16]:


titanic_df['embark_town'].mode()


# In[17]:


titanic_df['embark_town'].value_counts()/len(titanic_df)

