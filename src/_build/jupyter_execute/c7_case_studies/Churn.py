#!/usr/bin/env python
# coding: utf-8

# # Case Study: Churn
# 
# In this case study we try to create an answer why customers have left our service, a telecom operator.
# 
# The case study is divided into several parts:
# - Goals
# - Parsing
# - Preparation (cleaning)
# - Processing
# - Exploration
# - Visualization
# - Conclusion

# ## Goals
# 
# In this section we define questions that will be our guideline througout the case study
# 
# - Why are customers leaving us?
# - Can we cluster types of customers?
# 
# We'll (try to) keep these question in mind when performing the case study.

# ## Parsing
# 
# we start out by importing all libraries

# In[1]:


import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
get_ipython().run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('svg')


# in order to download datasets from kaggle, we need an API key to access their API, we'll make that here

# In[2]:


if not os.path.exists("/root/.kaggle"):
    os.mkdir("/root/.kaggle")

with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(
        {
            "username":"lorenzf",
            "key":"7a44a9e99b27e796177d793a3d85b8cf"
        }
        , f)


# now we can import kaggle too and download the datasets

# In[3]:


import kaggle
kaggle.api.dataset_download_files(dataset='blastchar/telco-customer-churn', path='./data', unzip=True)


# the csv files are now in the './data' folder, we can now read them using pandas, here is the list of all csv files in our folder

# In[4]:


os.listdir('./data')


# This dataset only contains 1 file, in it each row has all the information about a single customer and which services he or she has or had before churning.

# In[5]:


churn_df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print('shape: ' + str(churn_df.shape))
churn_df.head()


# Looks like there is some personal info and the configuration of the service, such as if they had an internet service, with or without options such as security, backup,...
# By the lookds of it these Yes/No answers are not booleans (i.e. 2 options) but rather categories as they have a third option, 'No ... service'.

# ## Preparation
# 
# here we perform tasks to prepare the data in a more pleasing format.

# ### Data Types
# 
# Before we do anything with our data, it is good to see if our data types are in order

# In[6]:


churn_df.info()


# I am opting to change the sernior citizan from 0/1 to No/Yes and convert them all to categories, let's do that right now.

# In[7]:


churn_df.SeniorCitizen = churn_df.SeniorCitizen.map({0: 'No', 1:'Yes'})
churn_df[['gender', 'SeniorCitizen', 'Partner','Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']] = churn_df[['gender', 'SeniorCitizen', 'Partner','Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']].astype('category')
churn_df.info()


# Now our yes/no answers are configured as categories, for numbers we see that there are 2: 'MontlyCharges' and 'TotalCharges'.
# I'm going to make them floating numbers

# In[8]:


churn_df[['MonthlyCharges', 'TotalCharges']] = churn_df[['MonthlyCharges', 'TotalCharges']].astype('float')
churn_df.info()


# Looks like we have encountered some problems, there are strings in the Total charges that are not able to be converted to a decimal number.
# We print out the rows that create an error and observe.

# In[9]:


churn_df[pd.to_numeric(churn_df.TotalCharges,errors='coerce').isna()]


# Seems that there are some customers being so new they have no total charges, for convenience i'm going to change the space to a 0.

# In[10]:


churn_df.TotalCharges = churn_df.TotalCharges.replace(' ', '0')


# In[11]:


churn_df[['MonthlyCharges', 'TotalCharges']] = churn_df[['MonthlyCharges', 'TotalCharges']].astype('float')
churn_df.info()


# ### Missing values
# 
# for each dataframe we apply a few checks in order to see the quality of data

# In[12]:


print(100*churn_df.isna().sum()/churn_df.shape[0])


# No missing values (if we do not count the ones we solved earlier), sometimes luck is on our side.

# ### Duplicates
# 
# For any reason, our dataset might be containing duplicates that would be counted twice and will introduce a bias we would not want. On the other hand, duplicates can be subjected to interpretation, here we would say that if 2 records are completely the same they are duplicates.

# In[13]:


churn_df.duplicated().any()


# ### Indexing
# 
# It is more convenient to work with an index, our dataset already contains an id which we can use as index

# In[14]:


churn_df = churn_df.set_index('customerID')
churn_df.head()


# ## Processing

# ### Churn vs no churn
# I would like to compare between persons that have churned and others, therefore a function that calculates the counts between churn and a given column would be convenient.
# By using functions I keep things dynamic without having to store a dataframe for each column, but static dataframes work equally well!

# In[15]:


def count_matrix(col_name):
  return churn_df.groupby(['Churn', col_name]).size().unstack()


# In[16]:


count_matrix('DeviceProtection')


# aside from the counts I would also like to know the mean, as some groups have a smaller population yet their proportion of churned persons might be higher.

# In[17]:


def mean_matrix(col_name):
  df = churn_df.groupby(['Churn', col_name]).size().unstack()
  return df.divide(df.sum(axis='columns'),axis='index')


# In[18]:


mean_matrix('DeviceProtection')


# out of curiosity, let's print all those 'mean matrices'

# In[19]:


for col in churn_df.columns.drop('Churn'):
  print(mean_matrix(col))
  print()


# We already see some big differences between populations of churn and no churn for some of these features, promising!

# ### one hot encoding
# I would also like to run the data into an algorithm, yet computers don't like categories, so I 'one hot encode' the categories and get a column/feature for each category in my categorical variables.

# In[20]:


churn_ohe_df = pd.concat(
    [
     pd.get_dummies(churn_df.drop(columns=['Churn'])),
     churn_df.Churn.eq('Yes').astype(int)
    ], axis='columns'
)
churn_ohe_df.head()


# ### correlation
# I went ahead and already calculated the correlation matrix for this dataset, with the ohe version of the data we can figure out which categories are related.
# In the next cell I printed out all correlations with the churn feature.

# In[21]:


churn_corr_df = churn_ohe_df.corr()
churn_corr_df['Churn']


# We can see that complementary categories show an inverse correlation, indicating that we are dealing with a excess of information.
# Logical as when option A is not chosen, option B is.
# However in this case, as some categoricals have 3 options I opt to keep all info, although it would be a good idea to remove 1 option for each category, this should become appearent in data exploration.

# In[21]:




