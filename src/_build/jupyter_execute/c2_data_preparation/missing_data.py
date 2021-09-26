#!/usr/bin/env python
# coding: utf-8

# ## Missing Data
# 
# In this notebook we will look at a few datasets where values from columns are missing.
# It is crucial for data science and machine learning to have a dataset where no values are missing as algorithms are usually not able to handle data with information missing.
# 
# For python, we will be using the pandas library to handle our dataset.

# In[1]:


import pandas as pd


# ### Kamyr digester
# 
# The first dataset we will be looking at is taken from a psysical device equiped with numerous sensors, each timepoint (1 hour) these sensors are read out and the data is collected. Let's have a look at the general structure

# In[2]:


kamyr_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/kamyr-digester.csv')
kamyr_df.head()


# Interesting, there seem to be 22 sensor values and 1 timestamp for each record. As mechanical devices are prone to noise and dropouts of sensors we would be foolish to assume no missing values are present.

# In[3]:


kamyr_df.isna().sum().divide(len(kamyr_df)).round(4)*100


# As expected, the datapoint 'AAWhiteSt-4' even has 46% of data missing!
# It seems we only have 300 datapoints and presumably these missing values occur in different records our dataset will be decimated if we just drop all rows with missing values.

# In[4]:


kamyr_df.shape


# In[5]:


kamyr_df.dropna().shape


# As we drop all rows with missing values, we are left with only 131 records.
# Whilst this might be good enough for some purposes, there are more viable options.
# 
# Perhaps we can first remove the column with the most missing values and then drop all remaining

# In[6]:


kamyr_df.drop(columns=['AAWhiteSt-4 ','SulphidityL-4 ']).dropna().shape


# Significantly better, although we lost the information of 2 sensors we now have a complete dataset with 263 records. For purposes where those 2 sensors are irrelevant this is a viable option, keep in mind that this dataset is still 100% truthful, as we have not imputed any values.
# 
# Another option, where we retain all our records would be using the timely nature of our dataset, each record is a measurement with an interval of 1 hour. I have no knowledge of this dataset but one might make the assumption that the interval of 1 hour is taken as the state of the machine does not alter much in 1 hour. Therefore we could do what is called a forward fill, where we fill in the missing values with the same value of the sensor for the previous measurement.
# 
# This would solve nearly all nan values as there might be a problem where the first value is missing. This is shown below.

# In[7]:


kamyr_df.fillna(method='ffill')['SulphidityL-4 ']


# Although our dataset is not fully the truth, we can see that little to no changes occur in the sensor and using a forward fill is arguably the most suitable option.
# 
# ### Travel times
# 
# Another dataset from the same source contains a collection of recorded travel times and specific information about the travel itself as e.g.: the day of the week, where they were going, ...

# In[8]:


travel_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/travel-times.csv')
travel_df


# we have a total of 205 records and we can already see that the FuelEconomy column seems pretty bad, let's quantify that.

# In[9]:


travel_df.isna().sum().divide(len(travel_df)).round(4)*100


# In the end, it doesn't seem that bad, but there are comments and nearly none of them are filled in. Which in perspective is understandable. Let's see what the comments look like

# In[10]:


travel_df[~travel_df.Comments.isna()].Comments


# As you would expect, these comments are text based. Now imagine we would like to run some Natural Language Processing (NLP) on these, it would be a pain to perform string operations on it when it is riddled with missing values.
# 
# Here a simple example where we select all records containing the word 'rain', with no avail.

# In[11]:


travel_df[travel_df.Comments.str.lower().str.contains('rain')]


# The last line of the python error traceback gives us the reason it failed, because there were NaN values present.
# 
# Luckily the string variable has more or less it's on 'null' value, being an empty string, this way these operations are still possible, most of the comments will just contain nothing.

# In[12]:


travel_df.Comments = travel_df.Comments.fillna('')


# In[13]:


travel_df[travel_df.Comments.str.lower().str.contains('rain')]


# Fixed! now we can use the comments for analysis.
# 
# We still have to fix the FuelEconomy, let us take a look at the non NaN values

# In[14]:


travel_df[~travel_df.FuelEconomy.isna()]


# It seems that aside NaN values there are also other intruders, a quick check on the data type (Dtype) reveils it is not recognised as a number!

# In[15]:


travel_df.info()


# The column is noted as an object or string type, meaning that these numbers are given as '9.24' instead of 9.24 and numerical operations are not possible.
# We can cast them to numeric but have to warn pandas to coerce errors, meaning errors will be converted to NaN values.
# Later we'll handle the NaN's.

# In[16]:


travel_df.FuelEconomy = pd.to_numeric(travel_df.FuelEconomy, errors='coerce')
travel_df.info()


# Wonderful, now the column is numerical and we can see 2 more missing values have popped up!
# We could easily drop these 19 records and have a complete dataset.

# In[17]:


travel_df.dropna()


# However im leaving them as an excercise for you to apply a technique we will see in the next part

# In[ ]:





# ### Material properties
# 
# Another dataset from the same source contains the material properties from 30 samples, this time there is not timestamp as the samples are not related in time with each other.

# In[18]:


material_df = pd.read_csv('https://raw.githubusercontent.com/LorenzF/data-science-practical-approach/main/src/c2_data_preparation/data/raw-material-properties.csv')
material_df


# let us quantify the amount of missing data

# In[19]:


material_df.isna().sum().divide(len(material_df)).round(4)*100


# Unfortunately that is a lot of missing data, covered in all records, dropping here seems almost impossible if we want to keep a healthy amount of records.
# 
# Here it would be wise to go for a more elaborate method of imputation, I opted for the K-nearest neighbours method, which looks at the K most similar records in the dataset to make an educated guess on what the missing value could be, this because we can assume that records with similar data are also similar over all the properties (columns).
# 
# Im using the sklearn library for this, which has more imputation techniques such as MICE.
# More info can be found [here](https://scikit-learn.org/stable/modules/impute.html)

# In[20]:


from sklearn.impute import KNNImputer


# im creating an imputer object and specify that i want to use the 5 most similar records and weigh them by distance from the to imputed record, meaning closer neighbours are more important.

# In[21]:


imputer = KNNImputer(n_neighbors=5, weights="distance")


# As the imputer only takes numerical values I had to do some pandas magic and drop the first column, which I then added again. The result is a fully filled dataset, you can recognise the new values as they are not rounded.

# In[22]:


pd.DataFrame(
    imputer.fit_transform(material_df.drop(columns=['Sample'])), 
    columns=material_df.columns.drop('Sample')
)


# This concludes the part of missing values, perhaps you can try yourself and impute the missing values for the FuelEconomy using the SimpleImputer or even the IterativeImputer.
