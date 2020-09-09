#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[4]:


df_train = pd.read_csv('train.csv')


# In[5]:


#drop unnecessary fields
df_train = df_train.drop(["pickup_time", "drop_time"], axis = 1)


# In[6]:


features = df_train.columns[1:11]
features


# In[7]:


df_train.fillna(0, inplace=True)


# In[16]:


#look for NA values
dfColumns=[col.strip().upper() for col in df_train.columns]
df_train.columns=dfColumns
print("Data Fields with NA values:")
print(df_train.columns[df_train.isna().any()].tolist())


# In[8]:


df_train


# In[10]:


#convert correct,incoreect into 1,0
y = (df_train['label'] == 'correct').astype(int)    
print(y[:21])


# In[ ]:


clf = RandomForestClassifier(n_jobs = 2, random_state=40)
clf.fit(df_train[features], y)


# In[12]:


#get test data
df_test =  pd.read_csv('test.csv')
df_test = df_test.drop(["pickup_time", "drop_time"], axis = 1)
df_test


# In[13]:


test_features = df_test.columns[1:11]
test_features


# In[14]:


prediction = clf.predict(df_test[test_features])


# In[15]:


df_test['prediction'] = prediction
df_test


# In[16]:


header = ["tripid", "prediction"]
df_test.to_csv('D:/test1.csv', columns = header)


# In[ ]:




