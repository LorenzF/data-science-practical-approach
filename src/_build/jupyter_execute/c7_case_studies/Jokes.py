#!/usr/bin/env python
# coding: utf-8

# # Case Study: Jokes
# 
# In this case study we find out if we can make ourselves funnier by analysing jokes from a database.
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
# - What jokes are funny?
# - Can we find types of jokes?
# - Would a joke recommender work?
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
kaggle.api.dataset_download_files(dataset='pavellexyr/one-million-reddit-jokes', path='./data', unzip=True)


# the csv files are now in the './data' folder, we can now read them using pandas, here is the list of all csv files in our folder

# In[4]:


os.listdir('./data')


# With only one file in the dataset, we import it.

# In[5]:


reddit_jokes_df = pd.read_csv('./data/one-million-reddit-jokes.csv')
print('shape: ' + str(reddit_jokes_df.shape))
reddit_jokes_df.head()


# Already we can see a lot of unnecessary information, so cleanup is important. It seems the joke is divided in a title and selftext where often the punchline is present.

# ## Preparation
# 
# here we perform tasks to prepare the data in a more pleasing format.

# ### Cleanup
# 
# First thing I would like to do see which columns are useless, by printing the amount of unique values

# In[6]:


for col in reddit_jokes_df.columns:
    print(col)
    print(reddit_jokes_df[col].nunique())
    print()


# a few columns only have 1 value, also the links are not important for our case, so we drop them too.

# In[7]:


reddit_jokes_df = reddit_jokes_df.drop(columns=['type', 'id', 'subreddit.id', 'subreddit.name', 'subreddit.nsfw', 'permalink', 'url'])
reddit_jokes_df.head()


# much cleaner already!

# ### Data Types
# 
# Before we do anything with our data, it is good to see if our data types are in order

# In[8]:


reddit_jokes_df.info()


# the created_utc feature is encoded in an unix timestamp, it would be more usefull to transform it to a timestamp

# In[9]:


reddit_jokes_df['created'] = pd.to_datetime(reddit_jokes_df['created_utc'], unit='s')
del reddit_jokes_df['created_utc']
reddit_jokes_df.head()


# ### Missing values
# 
# for each dataframe we apply a few checks in order to see the quality of data

# In[10]:


print(100*reddit_jokes_df.isna().sum()/reddit_jokes_df.shape[0])


# it looks like some jokes are missing the selftext field, we show a few here.

# In[11]:


reddit_jokes_df[reddit_jokes_df.selftext.isna()].sort_values(by='score', ascending=False)


# as far as I can see here the jokes are so short they are only one sentence, so we can fill in the missing values with an empty text.

# In[12]:


reddit_jokes_df.selftext = reddit_jokes_df.selftext.fillna('')


# This does not mean we are done, earlier I noticed the words [removed] and [deleted] in the selftext feature, indicating the joke was removed or deleted, these are missing values!

# In[13]:


reddit_jokes_df[reddit_jokes_df.selftext.isin(['[removed]', '[deleted]'])].head()


# I am going to remove these jokes as they are not complete anymore, it might have been that these jokes have been removed as they have already been posted.

# In[14]:


reddit_jokes_df = reddit_jokes_df[~reddit_jokes_df.selftext.isin(['[removed]', '[deleted]'])]
reddit_jokes_df.shape


# seems we have kept about 578k jokes, not bad!

# ### Duplicates
# 
# As formatting of text might be different i'm not expecting a lot of duplicates, let's see what we can find.

# In[15]:


reddit_jokes_df[reddit_jokes_df.duplicated(subset=['title', 'selftext'])]


# A fair amount of jokes are reposted, so we keep the ones with the highest score.

# In[16]:


reddit_jokes_df = reddit_jokes_df.sort_values('score').drop_duplicates(subset=['title', 'selftext'], keep='last').reset_index()


# ### Text formatting
# 
# Before we can analyze the text in the jokes we have to format it. We can do this by removing all special character and changing it all to lowercase

# In[17]:


for col in ['selftext', 'title']:
    reddit_jokes_df[col] = reddit_jokes_df[col].replace(to_replace="[^a-zA-Z,.!? ]", value="", regex=True).str.lower()

reddit_jokes_df.head()


# Next we create a single joke by combining the title and selftext, this makes it easier to operate.

# In[18]:


reddit_jokes_df['joke'] = reddit_jokes_df.title + ' ' + reddit_jokes_df.selftext
reddit_jokes_df = reddit_jokes_df.drop(columns=['title', 'selftext'])
reddit_jokes_df.head()


# ## Processing

# ### Timing of joke
# 
# I would like to know if the timing of the jokes makes an impact on how funny the joke is, so i grouped based on both the weekday as well as the hour of day.

# In[19]:


reddit_jokes_weekday = reddit_jokes_df.groupby(reddit_jokes_df.created.dt.weekday).score.agg(['mean', 'count'])
reddit_jokes_weekday


# In[20]:


reddit_jokes_hour = reddit_jokes_df.groupby(reddit_jokes_df.created.dt.hour).score.agg(['mean', 'count'])
reddit_jokes_hour


# ### Bag of words
# To be able to work with the words in our joke, we create a bag of words dataframe, where for each word and joke combination a count is kept of how many times the word is present in that joke. Notice that stopwords are removed.
# 
# First we split each joke up in words

# In[21]:


joke_words = reddit_jokes_df.joke.str.split(' ')
joke_words.head()


# Next we use the nltk toolkit to get a list of english stopwords.

# In[22]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')[:5]


# We remove all the stopwords from the jokes, now the jokes have a handicapped grammar.

# In[23]:


joke_words = joke_words.head().apply(lambda x : [word for word in x if word not in stopwords.words('english')])
joke_words.head()


# Finally we are going to use sklearn and the CountVectorizer to create the BoW vector, this is a sparse matrix as most words are not appearing in most jokes.
# This means we cannot visualise the matrix, or our computer would explode.

# In[24]:


from sklearn.feature_extraction.text import CountVectorizer

cnt_vect = CountVectorizer(analyzer="word", stop_words=stopwords.words('english'), max_features=20000) 

bow_jokes = cnt_vect.fit_transform(reddit_jokes_df.joke.values)


# In[25]:


bow_jokes


# But we can fetch the vocabulary of our bag, which starts with a lot of weird words, indicating we might have chosen too many features

# In[26]:


cnt_vect.get_feature_names_out()[:10]


# ### Term Frequency - Inverse Document Frequency
# Another interesting method is the tf-idf matrix, where each occurence is weighted by the overall frequency of that word. If a word is used often over all jokes, it won't be as important, but if a word is used infrequent it is more important.
# 
# Again we use sklearn to vectorize our jokes

# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
tfidf_jokes = tfidf_vect.fit_transform(reddit_jokes_df.joke.values)
tfidf_jokes


# we can create a quick dataframe to interpret the result, for each word in our dataset we retrieve the inverse document frequency, a high idf means a unique word.

# In[28]:


idf = pd.DataFrame(
    {
      'term': tfidf_vect.get_feature_names_out(),
      'idf': tfidf_vect.idf_,
    }
)
idf.head()


# When we sort them by idf we can find the most unique words, yet it doesn't seem to be useful at the moment.

# In[29]:


idf.sort_values(by='idf', ascending=False).head(10)


# ## Exploration

# In[30]:


good_jokes = reddit_jokes_df[reddit_jokes_df.score>10000].copy()
good_jokes


# In[ ]:





# In[31]:


good_jokes_word_cnt = pd.concat(
    [
        pd.Series(cnt_vect.get_feature_names_out()),
        pd.Series(np.asarray(bow_jokes[good_jokes.index].sum(axis=0)).squeeze()),
    ], axis='columns', keys=['word', 'count']
)
good_jokes_word_cnt


# In[32]:


good_jokes_word_cnt.sort_values('count', ascending=False).head(20)


# In[33]:


for joke in good_jokes[good_jokes.joke.str.contains(' man ')].tail(5).joke:
    print(joke)
    print()


# In[34]:


good_jokes_word_idf = pd.concat(
    [
        pd.Series(tfidf_vect.get_feature_names_out()),
        pd.Series(np.asarray(tfidf_jokes[good_jokes.index].sum(axis=0)).squeeze()),
    ], axis='columns', keys=['word', 'count']
)
good_jokes_word_idf


# In[35]:


good_jokes_word_idf[~good_jokes_word_idf.word.isin(stopwords.words('english'))].sort_values('count', ascending=False).head(20)


# In[36]:


tfidf_good_vect = TfidfVectorizer()
tfidf_good_jokes = tfidf_good_vect.fit_transform(good_jokes.joke.values)
tfidf_good_jokes


# In[37]:


good_jokes_word_idf = pd.concat(
    [
        pd.Series(tfidf_good_vect.get_feature_names_out()),
        pd.Series(np.asarray(tfidf_good_jokes.sum(axis=0)).squeeze()),
    ], axis='columns', keys=['word', 'count']
)
good_jokes_word_idf


# In[38]:


good_jokes_word_idf[~good_jokes_word_idf.word.isin(stopwords.words('english'))].sort_values('count', ascending=False).head(20)


# In[39]:


from sklearn.cluster import KMeans


# In[40]:


kmeans = KMeans(n_clusters=100)
kmeans.fit(tfidf_good_jokes)


# In[41]:


good_jokes['label'] = kmeans.labels_


# In[42]:


good_jokes.head()


# In[43]:


jokes_cluster_counts = good_jokes.label.value_counts()
jokes_cluster_counts


# In[44]:


for joke in good_jokes[good_jokes.label==jokes_cluster_counts.index[0]].sort_values('score', ascending=False).joke.head():
    print(joke)
    print()


# In[45]:


for joke in good_jokes[good_jokes.label==jokes_cluster_counts.index[-1]].sort_values('score', ascending=False).joke.head():
    print(joke)
    print()


# In[ ]:





# In[ ]:





# In[46]:


kaggle.api.dataset_download_files(dataset='vikashrajluhaniwal/jester-17m-jokes-ratings-dataset', path='./data', unzip=True)


# In[47]:


jester_jokes_df = pd.read_csv('./data/jester_items.csv')
print('shape: ' + str(jester_jokes_df.shape))
jester_jokes_df.head()


# In[48]:


jester_ratings_df = pd.read_csv('./data/jester_ratings.csv')
print('shape: ' + str(jester_ratings_df.shape))
jester_ratings_df.head()


# In[ ]:





# In[ ]:





# In[49]:


jester_ratings_df.groupby('jokeId').rating.mean()


# In[50]:


jester_sorted = jester_ratings_df.groupby('jokeId').rating.mean().to_frame().join(jester_jokes_df).sort_values('rating', ascending=False)
jester_sorted.head()


# In[51]:


for joke in jester_sorted.head().jokeText:
    print(joke)
    print('---')


# In[52]:


jester_ratings_pivot_df = jester_ratings_df.pivot(index='userId', columns='jokeId', values='rating')
jester_ratings_pivot_df.head()


# In[53]:


kmeans = KMeans(n_clusters=100)
kmeans.fit(jester_ratings_pivot_df.fillna(0))


# In[54]:


user_clusters = pd.Series(kmeans.labels_, index=jester_ratings_pivot_df.index)
user_clusters


# In[55]:


user_cluster_counts = user_clusters.value_counts()
user_cluster_counts


# In[56]:


users_set = set(user_clusters[user_clusters==user_cluster_counts.index[0]].index)
print(users_set)


# In[57]:


jester_group_sorted = jester_ratings_df[jester_ratings_df.userId.isin(users_set)].groupby('jokeId').rating.mean().to_frame().join(jester_jokes_df).sort_values('rating', ascending=False)
jester_group_sorted.head()


# In[58]:


users_set = set(user_clusters[user_clusters==user_cluster_counts.index[-5]].index)
jester_group_sorted = jester_ratings_df[jester_ratings_df.userId.isin(users_set)].groupby('jokeId').rating.mean().to_frame().join(jester_jokes_df).sort_values('rating', ascending=False)
jester_group_sorted.head()


# In[59]:


kmeans.inertia_


# In[60]:


kmeans = KMeans(n_clusters=200)
kmeans.fit(jester_ratings_pivot_df.fillna(0))


# In[61]:


kmeans.inertia_


# In[62]:


user_clusters = pd.Series(kmeans.labels_, index=jester_ratings_pivot_df.index)
user_clusters.value_counts()


# In[63]:


users_set = set(user_clusters[user_clusters==49].index)
jester_group_sorted = jester_ratings_df[jester_ratings_df.userId.isin(users_set)].groupby('jokeId').rating.mean().to_frame().join(jester_jokes_df).sort_values('rating', ascending=False)
jester_group_sorted.head()


# In[64]:


elbow_dict = {}
for k in [5, 10, 50, 100, 200, 500]:
    print(k)
    elbow_dict[k] = {}
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(jester_ratings_pivot_df.fillna(0))
    
    elbow_dict[k]['kmeans'] = kmeans
    elbow_dict[k]['inertia'] = kmeans.inertia_
    elbow_dict[k]['user_cluster'] = pd.Series(kmeans.labels_, index=jester_ratings_pivot_df.index)


# In[65]:


for k, clustering in elbow_dict.items():
    print(clustering['inertia'])


# In[66]:


inertia = pd.Series([clustering['inertia'] for k, clustering in elbow_dict.items()], index=elbow_dict.keys())
sns.lineplot(x=inertia.index, y=inertia)


# In[ ]:





# In[67]:


elbow_dict[100]['user_cluster'].value_counts()


# In[68]:


elbow_dict[500]['user_cluster'].value_counts()


# In[ ]:





# In[69]:


from sklearn.neighbors import NearestNeighbors


# In[70]:


nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(jester_ratings_pivot_df.fillna(0))


# In[71]:


jester_ratings_pivot_df.fillna(0).loc[[1]]


# In[72]:


dist, neighbours = nbrs.kneighbors(jester_ratings_pivot_df.fillna(0).loc[[1]])
neighbours[0].tolist()


# In[73]:


neighbours_ratings = jester_ratings_pivot_df.iloc[neighbours[0].tolist()[1:]]
neighbours_ratings


# In[74]:


approriate_jokes = neighbours_ratings.mean()[neighbours_ratings.mean()>7].index.tolist()
approriate_jokes


# In[75]:


recommended_jokes = jester_ratings_pivot_df.loc[1,approriate_jokes]
recommended_jokes


# In[76]:


recommended_jokes[recommended_jokes.isna()].index.tolist()


# In[77]:


for joke in jester_jokes_df[jester_jokes_df.jokeId.isin(recommended_jokes[~recommended_jokes.isna()].index.tolist())].jokeText:
    print(joke)
    print('---')


# In[78]:


for joke in jester_jokes_df[jester_jokes_df.jokeId.isin(recommended_jokes[recommended_jokes.isna()].index.tolist())].jokeText:
    print(joke)
    print('---')


# In[ ]:




