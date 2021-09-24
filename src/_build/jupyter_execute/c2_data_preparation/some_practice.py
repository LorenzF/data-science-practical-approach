#!/usr/bin/env python
# coding: utf-8

# ## Some practice
# 
# Now that you have learned techniques in data preparation, why don't you put them to use in this wonderfully horrifying dataset. Good luck!

# In[1]:


import os
import json

import pandas as pd
import kaggle


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


# In[3]:


kaggle.api.dataset_download_files(dataset='PromptCloudHQ/us-jobs-on-monstercom', path='./data', unzip=True)


# In[4]:


df = pd.read_csv('./data/monster_com-job_sample.csv')


# In[5]:


df.head()


# In[ ]:





# Need some inspiration? perhaps [this](https://www.kaggle.com/ankkur13/perfect-dataset-to-get-the-hands-dirty) might work.
