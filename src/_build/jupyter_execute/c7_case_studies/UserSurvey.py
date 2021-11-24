#!/usr/bin/env python
# coding: utf-8

# # Case Study: User survey
# 
# In this case study we figure out how to analyse the responses from a user survey form kaggle
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
# - What influences salary?
# - Can we deduce common skills for job titles?
# - Do higher paid jobs spend time differently?
# - Important: education or experience?
# 
# We'll (try to) keep these question in mind when performing the case study.

# ## Parsing
# 
# we start out by importing all necessary libraries

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
kaggle.api.dataset_download_files(dataset='kaggle/kaggle-survey-2018', path='./data', unzip=True)


# the csv files are now in the './data' folder, we can now read them using pandas, here is the list of all csv files in our folder

# In[4]:


os.listdir('./data')


# 
# The file of our interest is 'athlete_events.csv', it contains every contestant in every sport since 1896. Let's print out the top 5 events.

# In[5]:


choice_df = pd.read_csv('./data/multipleChoiceResponses.csv')
print('shape: ' + str(choice_df.shape))
choice_df.head()


# In[6]:


free_form_df = pd.read_csv('./data/freeFormResponses.csv')
print('shape: ' + str(free_form_df.shape))
free_form_df.head()


# I saw that the first row of our choice dataframe contains the questions, to let's extract that.

# In[7]:


questions = choice_df.iloc[0]
choice_df = choice_df.drop(0)


# In[8]:


questions.head(20)


# ## Preparation
# 
# here we perform tasks to prepare the data in a more pleasing format.

# ### Data Types
# 
# Before we do anything with our data, it is good to see if our data types are in order

# In[9]:


choice_df.info()


# Seems there are to many too show, so we have to do some manual work, The first 10 questions seem to be about personal info, where the first one is about gender

# In[10]:


print(questions.Q1)
choice_df.Q1.value_counts()


# In[11]:


print(questions.Q1_OTHER_TEXT)
choice_df.Q1_OTHER_TEXT.unique()


# Hmm the self-describe seems to already been encoded, as there are so many different answers I would opt to ignore those results as they only take up 79 answers of all 24k.
# For the second question I am going to convert it to an ordinal value, this way we know the order of the categories.

# In[12]:


choice_df.Q2 = choice_df.Q2.astype(pd.api.types.CategoricalDtype(categories=['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70-79', '80+'], ordered=True))
print(questions.Q2)
choice_df.Q2


# Next we have a few very important questions that signify the situation of each user in our survey. I chose for nominal categories as I don't want to be biased.

# In[13]:


print(questions.Q6)
choice_df.Q6.value_counts()


# In[14]:


print(questions[['Q3', 'Q4', 'Q5', 'Q6', 'Q7']])
choice_df[['Q3', 'Q4', 'Q5', 'Q6', 'Q7']] = choice_df[['Q3', 'Q4', 'Q5', 'Q6', 'Q7']].astype('category')


# Question 8 is about experience, or as they call it tenure. Not as a numerical value but in categories, so again I create an ordinal category from it.

# In[15]:


print(questions.Q8)
choice_df.Q8.value_counts()


# In[16]:


choice_df.Q8 = choice_df.Q8.astype(pd.api.types.CategoricalDtype(categories=['0-1', '1-2', '2-3', '3-4', '4-50', '5-10', '10-15', '15-20', '20-25', '25-30', '30+'], ordered=True))
print(questions.Q8)
choice_df.Q8


# And not to forget we have the salary, again as a category, which is unfortunate since we could have been able to create a more accurate prediction in the end.
# Here I opt for an ordinal category.

# In[17]:


choice_df.Q9.value_counts()


# In[18]:


choice_df.Q9 = choice_df.Q9.astype(pd.api.types.CategoricalDtype(categories=['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000', '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000', '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000', '300-400,000', '400-500,000', '500,000+',], ordered=True))
choice_df.Q9


# ### Missing values
# 
# for each dataframe we apply a few checks in order to see the quality of data

# In[19]:


print(100*choice_df.isna().sum().head(20)/choice_df.shape[0])


# You can clearly see that there are a lot of missing values, for questions 11 and onwards this is just because they did not check that answer on a question, but for 1-10 this is a problem as these are 'mandatory' questions. I have no idea how to fill this in and salary is missing about 35%, pretty disastrous, but this is to be expected with user surveys.
# 
# Another problem we have here is trolls, there might have been persons that would just fill this in to mess with our data collection, I thought they might have been funny and answered a high salary.

# In[20]:


choice_df[choice_df.Q9=='500,000+'].Q2.value_counts()


# you can see there are 13 persons between 25-29 that earn more than 500k annually, which i think is near impossible. Let us see what they are upto.

# In[21]:


choice_df[(choice_df.Q9=='500,000+') & (choice_df.Q2=='25-29')]


# No way they are this succesfull, i'm not yet going to remove them, but i'm definitely going to keep this in mind, this might break our predictions!
# 
# Later on I will remove the entries without salaries, but im going to keep them in a prediction dataframe, so we could perhaps predict their salary, we don't have a reference but still might be interesting. For the rest of the preparation im going to keep them in here so the final format of both train and prediction are the same.

# ### Duplicates
# 
# It is very highly unlikely but just to check if no one has entered the same survey twice, we check for duplicates

# In[22]:


choice_df[choice_df.duplicated()]


# I take back my words, seems there are some faulty entries, perhaps we should even improve our bad entry detection? For now im just going to remove duplicates

# In[23]:


choice_df = choice_df.drop_duplicates()


# At this point im going to seperate the non salary entries from our training dataframe. resulting in 2 partitions:
# - train_df
# - prediction_df

# In[24]:


prediction_df = choice_df[(choice_df.Q9.isna()) | (choice_df.Q9=='I do not wish to disclose my approximate yearly compensation')]
train_df = choice_df.drop(prediction_df.index)
print('prediction shape:' + str(prediction_df.shape))
print('remaining shape:' + str(train_df.shape))


# ## Processing
# 
# For other questions I selected a few that caught my interest, here is the list that made it. Notice that I did not perform any preparation on these question as they mostly are checkmarks on a survey, yet in processing I am going to create a more convenient method to store them.

# In[25]:


print(questions.Q11_Part_1)
#print(questions.Q12_Part_1_TEXT)
print(questions.Q13_Part_1)
print(questions.Q16_Part_1)
print(questions.Q17)
print(questions.Q19_Part_1)
print(questions.Q21_Part_1)
print(questions.Q31_Part_1)
print(questions.Q34_Part_1)
print(questions.Q42_Part_1)
print(questions.Q49_Part_1)


# ### One hot encoding questions
# What I will do here is create a makeshift database, not in SQL as usually just to keep it simple, but in a dictionary of dataframes. For each question I will take the answers and create a one hot encoded table from them, for each user we will know which checkmarks they marked and which they didn't. This view makes it easier to apply statistics and machine learning to the data.

# In[26]:


answer_dfs = {}
for question in ['Q11', 'Q13', 'Q16', 'Q19', 'Q21', 'Q31', 'Q34', 'Q42', 'Q49']:
  
  choices = train_df[train_df.columns[train_df.columns.str.contains(question)][:-1]].notnull().astype(int)
  choices.columns = questions[questions.index.str.contains(question)][:-1].str.split(' - ').apply(lambda x: x[-1]).values
  answer_dfs[question] = choices


# an example of a question, Q13: Which IDE's have you used in the last 5 years?

# In[27]:


answer_dfs['Q13']


# for some reason they did Q17 differently, so we have to one hot encode it in another method.

# In[28]:


answer_dfs['Q17'] = pd.get_dummies(train_df[train_df.columns[train_df.columns.str.contains('Q17')][:-1]])
answer_dfs['Q17']


# That was for our choices data, where the questions are based on choices, for generic info we do it a bit different, we create a general dataframe containing all info.

# In[29]:


info_df = train_df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']]
info_df.columns = questions[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']]


# In[30]:


info_df


# ### Mean choice Matrix
# As we have so much information to process, I opted to keep it dynamic, the following function helps in that, it calculates for a question from our choice database the mean occurence for each group in a feature of the info dataframe.
# Let's say we want to know the average amount of persons that know a specific language for each role/job title. We would have to match Q16 (known languages) with Q6 (job description). This is performed below, notice how it both performs a merge (join) and a groupby to get the result.

# In[31]:


def mean_matrix(info, question):
  return info_df[[questions[info]]].join(answer_dfs[question]).groupby(questions[info]).mean()


# In[32]:


mean_matrix('Q6','Q16')


# We can see that for each combination of job title and programming language an average between 0 and 1 persons have checked this option, e.g. the combination of data scientist and python equals 0.86, meaning that 86% of data scientists know python. 
# 
# Similarly we can also calculate correlation between choices from our choice database, here we did it again for Question 16.

# In[33]:


answer_dfs['Q16'].corr()


# Here we see thich answers are checked usually together or not, as an example we see that python and SQL have a correlation of 19% whilst Python and R have a correlation of 7.7% which is logical as Python and R have a similar purpose and SQL is complementary. Obviously None is always negatively correlated, a good example of obsolete information!

# ### Count matrix
# to correlate information between 2 questions of the info dataframe, we create a function that counts the occurence of each combination. An example is given for question 2 (age) and Question 7 (industry). With this information we can find out if there is a correlation between information of our users in the survey, not specifically their choices on the multiple choice answers.

# In[34]:


def count_matrix(q1, q2):
  return info_df[[questions[q1], questions[q2]]].groupby([questions[q1], questions[q2]]).size().unstack()


# In[35]:


count_matrix('Q2', 'Q7')


# In[35]:




