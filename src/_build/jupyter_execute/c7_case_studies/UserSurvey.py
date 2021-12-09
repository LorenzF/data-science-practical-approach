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
plt.rcParams['figure.figsize'] = [10, 10]


# in order to download datasets from kaggle, we need an API key to access their API, we'll make that here

# In[2]:


if not os.path.exists(os.path.expanduser('~/.kaggle')):
    os.mkdir(os.path.expanduser('~/.kaggle'))

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
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


# The file of our interest is 'multipleChoiceResponses.csv', it contains the multiple choice responses of our subjects. Let's print out the top 5 events.

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


choice_df.Q8 = choice_df.Q8.astype(pd.api.types.CategoricalDtype(categories=['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30 +'], ordered=True))
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
#info_df.columns = questions[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']]


# In[30]:


info_df


# ### Mean choice Matrix
# As we have so much information to process, I opted to keep it dynamic, the following function helps in that, it calculates for a question from our choice database the mean occurence for each group in a feature of the info dataframe.
# Let's say we want to know the average amount of persons that know a specific language for each role/job title. We would have to match Q16 (known languages) with Q6 (job description). This is performed below, notice how it both performs a merge (join) and a groupby to get the result.

# In[31]:


def mean_choice_matrix(info, question):
    return info_df[[info]].join(answer_dfs[question]).groupby(info).mean()


# In[32]:


mean_choice_matrix('Q6','Q16')


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
    return info_df[[q1, q2]].groupby([q1, q2]).size().unstack()
def mean_matrix(q1, q2):
    return info_df[[q1, q2]].groupby([q1, q2]).size().unstack().apply(lambda x: x/x.sum(), axis='columns')


# In[35]:


count_matrix('Q2', 'Q7')


# ## Exploration
# 
# To start of our exploration I would like to know what influences our salary, to do so I created a count_matrix function that counts the occurences of each option with information questions, to illustrate an example with Q4: which degree?

# In[36]:


count_matrix('Q4', 'Q9')


# By using a contingency chi squared test we can find out which degrees are under- and overrepresented for which salary ranges.

# In[37]:


F, p, df, exp = scipy.stats.chi2_contingency(count_matrix('Q4','Q9'))
F, p


# With such significance we already know this is not a coincidence and the correlation will propably be large. comparing true and expected values we can see where the difference is.

# In[38]:


degree_diff = count_matrix('Q4', 'Q9')-exp
degree_diff


# It would be very hard to analyse this difference using the complete matrix, I propose we take the sum of differences for the high paying jobs and compare those.
# As a threshold of high-paying I chose to go for those who 'earn six figures'.

# In[39]:


degree_diff.loc[:,'100-125,000':'400-500,000'].sum(axis='columns').sort_values()


# By the looks of it, it pays of to study longer and get more degrees, as a Masters degree is overrepresented by 57 persons and Doctoral degrees even by 181 persons. On the other side Bachelors or Professional degrees are underrepresented whilst no formal education is not particularly underperforming. 
# 
# We can do the same for Q5: which field? were we analyse in which field the person works compared with their salary.

# In[40]:


df = count_matrix('Q5', 'Q9')
df = df.loc[~(df==0).all(axis=1)]
F, p, deg, exp = scipy.stats.chi2_contingency(df)
print(f'F: {F}, p: {p}')
diff = df-exp
diff.loc[:,'100-125,000':'400-500,000'].sum(axis='columns').sort_values()


# We have a clear loser here, for some reason the computer science department seems to be underpayed or either not worth their money. on the other side there is a more gradual increase and most fields are over represented in the region of highly paid jobs.
# 
# What about Q6: your job description?

# In[41]:


df = count_matrix('Q6', 'Q9')
df = df.loc[~(df==0).all(axis=1)]
F, p, deg, exp = scipy.stats.chi2_contingency(df)
print(f'F: {F}, p: {p}')
prof_diff = df-exp
prof_diff.loc[:,'100-125,000':'400-500,000'].sum(axis='columns').sort_values()


# Highly expected students score very bad here, which is a good confirmation. Something remarkable here is the difference between Data Analyst and Data Scientist, two jobs that seem to be similar have such a difference in representation in the high paid region. 
# 
# To complete the analysis we also chose Q7: which sector?

# In[42]:


df = count_matrix('Q7', 'Q9')
df = df.loc[~(df==0).all(axis=1)]
F, p, deg, exp = scipy.stats.chi2_contingency(df)
print(f'F: {F}, p: {p}')
diff = df-exp
diff.loc[:,'100-125,000':'400-500,000'].sum(axis='columns').sort_values()


# Here we can again see the students, this time acompanied by the Academics/Education sector, which is understandable as it usually is a non-profit governmental oranization.
# Leading the charts we have the Computers/Technology sector which is currently booming.
# 
# ### Common skills
# 
# Aside from salary we also are interested in most common skills for a specific job title, therefore I took the averages of each choice for a multiple choice question. Here as example the combination of Q6: which job and Q16: what languages?

# In[43]:


mean_choice_matrix('Q6', 'Q16')


# This does give us a lot of information, e.g. that 86% of all data scientists use python, yet it does not show correlation between answers, therefore we would need to go back to our original one hot encoded data and merge with the necessary info, it look like this.

# In[44]:


df = info_df[['Q6','Q9']].join(answer_dfs['Q16'])
df.head()


# To deduce the correlation for all persons, we would not need Q6 or Q9, this will become necessary when we want to select subgroups. For now we calculate the percentage of all persons that have chosen each option

# In[45]:


df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# We can see that options as Python, SQL and R are very popular, yet how do they correlate? Are the same persons who pick python also those who pick R? We use the numerical correlation to calculate this. Notice that I use the Spearman Rank as our data consists of 0 and 1, being non-normal distributed.

# In[46]:


all_jobs_corr = df.corr('spearman')
all_jobs_corr


# As an example you can see that for those who chose python (the column called python) there is a 7.7% correlation with R and 19.1% with SQL, so a person who uses python is more likely to also know SQL (or Bash) rather than R. This is understandable as those languages are similar in usage.
# 
# Now we want to change things so we don't look towards all persons, but only data scientists, as I am a data scientist and want to know which languages I should learn more about.

# In[47]:


df = df[df['Q6']=='Data Scientist']
data_science_corr = df.corr('spearman')
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# For the case of percentage that have chosen languages, things do not drastically change, although all percentages are up much more. You can see that the shell scripting language Bash has shifted upwards so a basic knowledge in Bash would not hurt.
# 
# For the correlation I opted to show the difference with the all persons correlation.

# In[48]:


data_science_corr-all_jobs_corr


# In the Python column we can see that generic non Data Science languages such as C/C++ and Java are falling, yet the correlation with Bash is also negative, this indicates by selecting Data Science profiles we have on average more people choosing for Bash, but NOT in combination with Python.
# Although results are somewhat expected, there do not seem to be any drastic changes.
# 
# To shake things up more, we apply a second filter, where we only take the persons who earn more than 100k.

# In[49]:


df = df.loc[('100-125,000'<df.Q9) & (df.Q9<'500,000+')]
high_paying_job_corr = df.corr('spearman')
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# For the average of choices we can now see that Scala - a language used for big data - shootsj up the ranks, indicating that having a data engineering language in your knowledge base is good for your salary.
# 
# To compare the correlation of high paying data science jobs I took the difference with correlation of all jobs.

# In[50]:


high_paying_job_corr-all_jobs_corr


# Again as I mainly use python I will be looking at the Python column, you can see that Scala is indeed correlated with Python and Java or C/C++ is not a must at all.
# 
# In a similar fashion we evaluate the influence of machine learning toolkits, where we first see the average choice of all persons.

# In[51]:


df = info_df[['Q6','Q9']].join(answer_dfs['Q19'])
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# Scikit-learn or sklearn (the one we sometimes use) is chosen the most often, problably because of it's ease of use and effectiveness. Now we would like to see the choice of data scientists

# In[52]:


df = df[df['Q6']=='Data Scientist']
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# No particular shifts although None has dropped to the last place, indicating that knowledge of Machine Learning is essential for a Data Scientist.
# 
# What happens when we only ask the high paying data scientists?

# In[53]:


df = df.loc[('100-125,000'<df.Q9) & (df.Q9<'500,000+')]
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# Nothing in particular, except that all percentages have increased, to conclude your choice of machine learning library is not that important!
# 
# ### Time spend
# 
# I would also like to know how other data scientists spend their time, in the same fashion we analyse this

# In[54]:


df = info_df[['Q6','Q9']].join(answer_dfs['Q34'])
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# Looks like we made a mistake, we one hot encoded all questions but this is a numerical question, we need som more manipulations.

# In[55]:


df = train_df[train_df.columns[train_df.columns.str.contains('Q34')][:-1]].fillna(0).astype(float)
df = info_df[['Q6','Q9']].join(df).rename(columns=questions[questions.index.str.contains('Q34')].apply(lambda x: x.split(' - ')[-1]).to_dict())
df


# This looks better, now for each answer we have a value between 0 and 100%, we need to check if they have filled in this answer though

# In[56]:


df = df[~(df.drop(columns=['Q6', 'Q9']).sum(axis='columns')==0)]
df


# much better! we have the percentages and dropped the rows where nothing was filled in

# In[57]:


time_all = df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)
time_all


# In the beginning of my course I showed a graph on how a data scientists time is divided, this should give another view on it, most of it is data cleaning and model selection, visualization and insights are equally important but get more hands-on time.
# 
# How are these relations when looking at Data Scientists?

# In[58]:


df = df[df['Q6']=='Data Scientist']
time_scientist = df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)
time_scientist


# I would not say things have changed much, as expected as many of the persons are data scientists.
# Does this stay when we filter on the higher paid jobs?

# In[59]:


df = df.loc[('100-125,000'<df.Q9) & (df.Q9<'500,000+')]
time_high_pay = df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)
time_high_pay


# There seems to be a little change, we can see that data visualization is less important, this is understandable as this is rather a task for a data analyst that creates reports using graphs.
# 
# So if I want to specialize myself in Data Science I should not put the focus on data visualizations.
# 
# To end this analysis I would like to pick Q42: Quality control of products. Again we do the same analysis

# In[60]:


df = info_df[['Q6','Q9']].join(answer_dfs['Q42'])
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# In[61]:


df = df[df['Q6']=='Data Scientist']
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# In[62]:


df = df.loc[('100-125,000'<df.Q9) & (df.Q9<'500,000+')]
df.drop(columns=['Q6','Q9']).mean().sort_values(ascending=False)


# You can see that Data Scientists focus more on Metrics that consider unfair bias, as this is often an issue in Data Science, when reporting data biases might not be that critical (or might even help you) but in Data Science - when exploring new ideas - it is important to not have a bias that might disrupt your machine learning algorithm.
# 
# ### Age vs experience
# 
# Something we can really do much about, but it would be nice to see if it is never too late to change careers.
# For both age and experience we create a cross-tabulation and calculate a contingency test.

# In[63]:


age_crosstab_df = info_df.groupby(['Q9', 'Q2']).size().unstack()
age_crosstab_df


# just by looking at it you can see a correlation, but just for significance we do the statistics

# In[64]:


F, p, deg, exp = scipy.stats.chi2_contingency(age_crosstab_df)
print(f'F: {F}, p: {p}')
diff = age_crosstab_df-exp
age_diff = diff.loc['100-125,000':'400-500,000'].sum()#.sort_values()
age_diff


# Again I took the high paying jobs and you can see that from the age of 30 there is an overrepresentation in high paying jobs. We can safely say that by increasing age you are more likely to end up in the high paying salary sector although it reverts back around the age of 55.
# 
# Now for the experience

# In[65]:


exp_crosstab_df = info_df.groupby(['Q9', 'Q8']).size().unstack()
exp_crosstab_df


# A less obvious correlation, we can use the F values to compare.

# In[66]:


F, p, deg, exp = scipy.stats.chi2_contingency(exp_crosstab_df)
print(f'F: {F}, p: {p}')
diff = exp_crosstab_df-exp
exp_diff = diff.loc['100-125,000':'400-500,000'].sum()#.sort_values()
exp_diff


# The F value is indeed lower, indicating that the correlation between age and salary is stronger than age and experience. The expected experience level to reach the high paying jobs seems to be around the 5 year mark.

# ## Visualisation
# 
# Although data scientists spend less time visualizing, I'm still going to make the effort here, a little refreshment, we created a mean matrix between 2 informative questions.

# In[67]:


mean_matrix('Q6','Q9')


# What I was thinking about would be a bar chart where each job title is a row and the distribution of each salary is shown, below the example

# In[68]:


df = mean_matrix('Q6', 'Q9').dropna().cumsum(axis='columns')
for idx, col in enumerate(df.columns[::-1]):
    sns.barplot(x=df[col], y=df.index, color=sns.color_palette('colorblind')[idx%10])

plt.xlabel('distribution of salary')
print(df.columns.tolist())
plt.show()


# The colors are awful but it displays the salary distribution, you can see that students clearly are in the lower parts similar to research assistants, notice that jobs with low statistical count can create a distortion as e.g. data journalist only has about 20 records. Jobs such as Manager and Principal Investigator seem to have a very even distribution indicating a faster climbing up the salary ladder.
# 
# In a similar fashion for other questions you could construct the same graph.
# 
# Another way to look at these things would be to use the difference between true and expected values, we already created the degree for differences, let's turn this into a bar plot.

# In[69]:


df = degree_diff.loc[:,'100-125,000':'400-500,000'].sum(axis='columns').sort_values()
df


# In[70]:


sns.barplot(x = df.index.astype('str'), y=df, color='b')
plt.xticks(rotation=90)
plt.show()


# There are a lot of things you can still do to beautify this graph, but that's not our main interest, it shows the under- and overrepresented groups in high paying jobs. It would be wise however to create a relative version of this, as e.g. bachelor's degrees might be much more prevalent than others.
# 
# The same can be done with groupings of profession/job

# In[71]:


df = prof_diff.loc[:,'100-125,000':'400-500,000'].sum(axis='columns').sort_values()
sns.barplot(x = df.index.astype('str'), y=df, color='b')
plt.xticks(rotation=90)
plt.show()


# To keep things consistent and because people love bar charts, we can use them to also display the disparity of choices of programming languages between high paying data scientists and all persons

# In[72]:


df = (high_paying_job_corr-all_jobs_corr).Python.sort_values()
sns.barplot(x = df.index.astype('str'), y=df, color='b')
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# you can see that the correlation between python and C/C++ is 15% less likely for high paid data scientists, indicating that it is not a good choice to learn next, in contrast languages such as Scala and SAS are a good option!
# 
# As far as my knowledge goes, the increase in correlation with None is because they are both negative and Python is more often chosen for data scientists, therefore the option 'not Python, not None' (but another language) is less often chosen, resulting in a higher correlation.
# 
# If you would want to make things a bit more fancy, you could use a clustermap, underlaying an algorithm will cluster your parameters into groups, here we cluster the correlation between common languages.

# In[73]:


df = info_df[['Q6','Q9']].join(answer_dfs['Q16'])
sns.clustermap(df.corr('spearman'))
plt.show()


# The algorithm was able to group languages such as Python, Bash, SQL and Scala, indicating that there is some correlation, but this graph makes things rather complicated in my opinion.
# 
# Now about time spending, we could visualize this by showing the difference between high paid scientists and regular persons

# In[74]:


df = time_high_pay-time_all
sns.barplot(x = df.index.astype('str'), y=df, color='b')
plt.xticks(rotation=90)
plt.show()


# we can see they spend more time on cleaning data, communication and production readiness, but less on visualization.
# Efficient time handling can be crucial for a good career!
# 
# At last we discussed age vs experience, as we cannot use histograms and overlapping is not possible with different categories (age vs exp) we are stuck with a bar chart. The repetivity of our dataset is reflected in our visualization.

# In[75]:


df = age_diff
sns.barplot(x = df.index, y=df, color='b')
plt.show()


# Although simple it clearly shows the surplus of older persons in the high paying jobs.

# In[76]:


df = exp_diff
sns.barplot(x = df.index, y=df, color='b')
plt.show()


# And as known before, experience gets your salary going from the 5 years and onwards

# ## Summary

# - Degrees and Job title strongly influences job salary
# - Job sectors as Academics are underpayed
# - For a data Scientist using python aim for other skills such as Scala and forget C/C++
# - Your choice of Machine Learning library is of no importance
# - Data Scientists spend less time visualizing and more cleaning, communicating and production 
# - Data Scientists are more worried about biases in their analysis
# - Although both relevant, Age is more an indicator of a higher salary than experience, never to late to chase your dreams!

# In[ ]:




