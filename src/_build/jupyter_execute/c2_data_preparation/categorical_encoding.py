#!/usr/bin/env python
# coding: utf-8

# ## Categorical encoding
# 
# Often we deal with categorical data and this kind of data is something computer algorithms are not able to understand.
# On the other hand long categorical features might take up unnecessary memory in our dataset, so converting to a categorical feature is optimal.

# In[1]:


import pandas as pd


# ### Raw Material Charaterization
# 
# In this dataset, we have a few numerical features describing characteristics of our material, next to that we also have an Outcome feature describing the state of our material in a category.
# 
# Let's have a look at the data

# In[2]:


raw_material_df = pd.read_csv('./data/raw-material-characterization.csv')
raw_material_df.head()


# So we can see that the outcome is indeed a text field with a human interpretable value.
# The different values are:

# In[3]:


raw_material_df.Outcome.unique()


# Image that we would like to get all records where the Outcome is less than adequate, using strings this is not possible as the computer does not understand relations of Adequate and Poor when they are denoted as text.

# In[4]:


raw_material_df[raw_material_df.Outcome<'Adequate']


# To overcome this we can change the type of the feature from 'object' (string) to 'category' let us look at the data types of our data

# In[5]:


raw_material_df.info()


# Now we can change that of Outcome to category using the astype method

# In[6]:


raw_material_df.Outcome = raw_material_df.Outcome.astype('category')
raw_material_df.info()


# Our feature might be of categorical nature now, however we still have to define it is an ordinal category and has an order.

# In[7]:


raw_material_df.Outcome = raw_material_df.Outcome.cat.as_ordered().cat.reorder_categories(['Poor', 'Adequate'])


# If we retry to effort to only take the records where the Outcome is less than Adequate, we now get an outcome!
# Since we only have 2 categories this is a bit unfortunate, but you should get the idea behind it.

# In[8]:


raw_material_df[raw_material_df.Outcome<'Adequate']


# Let's take this a step further, since computer algorithms still have no idea what the numerical relation is between Adequate and Poor, we could use a Label Encoder for that.

# In[9]:


from sklearn.preprocessing import LabelEncoder


# the label encoder is inputted with the Outcome feature and recognises 2 types, it chooses a numerical value for each while fitting.

# In[10]:


le = LabelEncoder()
le.fit(raw_material_df.Outcome)


# After fitting we can use this encoder to transform our dataset!

# In[11]:


raw_material_df['outcome_label'] = le.transform(raw_material_df.Outcome)
raw_material_df.head()


# It seems something unfortunate has happened, the encoder gave the Adequate an outcome label of 0, which is lower than the label of Poor (1), this might be bad if we would like to give a score as our outcome.
# 
# There is in pandas another method of mapping a label to a category albeit less automated, as you would have to know the categories in your feature.

# In[12]:


raw_material_df.outcome_label = raw_material_df.Outcome.map({'Poor': 0, 'Adequate':1})
raw_material_df.head()


# Yes! This did the trick, now we can use that outcome label to predict an outcome for future samples.

# ## Restaurant tips
# 
# Now we are going to look at a dataset of tips, here a restaurant tracked the table bills and tips for a few days in the week whilst also noting the gender, smoking habit and time of day.
# This led to a small yet very interesting dataset, let's have a look!

# In[13]:


tips_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
tips_df


# We can see here that we have a lot of categorical variables: gender, smoker, the day and the time.
# In later sections we will see how we can aggregate on these categorical variables.
# Now however we would like to process them for a machine learning exercise, where we need numbers not text.
# For the features smoker and day, you could argue there is a clear numbering between them, smoking is 1 if the person was smoking and e.g. Sun relates to 7 as it is the seventh day of the week.
# 
# But for the gender this is different, we can't really say that women are 1 and Men are 0 or vice versa (although in this binary case it might work).
# The same theory applies for time, if we would say that breakfast, lunch and dinner equal to 0, 1 and 2 this would give our algorithm a bad impression as it would think dinner is twice lunch...
# 
# We use One Hot Encoding for this, the idea is that for each value of the feature we create a new column.

# In[14]:


from sklearn.preprocessing import OneHotEncoder


# First we create our encoder, then we give it the day column to learn and see which values of categories there are.

# In[15]:


ohe = OneHotEncoder()
ohe.fit(tips_df[['day']])


# Now we can perform an encoding, here we insert the day column and it returns a matrix with 4 columns corresponding to the 4 days in our feature.

# In[16]:


ohe.transform(tips_df[['day']]).todense()


# As this is a rather mathematical approach for this simple problem I prefer to use the pandas approach for this, which is the get_dummies method.
# The outcome is much more pleasing yet completely the same.

# In[17]:


pd.get_dummies(tips_df.day)


# As an exercise you could create a script that given a specific feature (e.g. day):
# - extracts that feature
# - creates dummies
# - concattenates it to the dataframe
# 
# Good luck!

# In[ ]:




