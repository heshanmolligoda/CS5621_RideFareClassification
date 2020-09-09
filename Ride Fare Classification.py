#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[12]:


df_train = pd.read_csv('train.csv')


# In[13]:


df_train.head(10)


# In[14]:


#look for NA values
dfColumns=[col.strip().upper() for col in df_train.columns]
df_train.columns=dfColumns
print("Data Fields with NA values:")
print(df_train.columns[df_train.isna().any()].tolist())


# In[15]:


df_train = df_train.dropna()


# In[19]:


#look for NA values
dfColumns=[col.strip().upper() for col in df_train.columns]
df_train.columns=dfColumns
print("Data Fields with NA values:")
print(df_train.columns[df_train.isna().any()].tolist())


# In[17]:


#drop unnecessary fields

df_train = df_train.drop(["PICKUP_TIME", "DROP_TIME"], axis = 1)
features = df_train.columns[1:11]
features


# In[53]:


#convert correct,incoreect into 1,0
y = (df_train['LABEL'] == 'correct').astype(int)    
print(y[:21])


# In[24]:


clf = RandomForestClassifier(n_jobs = 2, random_state=0)
clf.fit(df_train[features], y)


# In[36]:


df_test =  pd.read_csv('test.csv')
df_test = df_test.drop(["pickup_time", "drop_time"], axis = 1)
df_test.head(10)


# In[38]:


#look for NA values
dfColumns=[col.strip().upper() for col in df_test.columns]
df_test.columns=dfColumns
print("Data Fields with NA values:")
print(df_test.columns[df_test.isna().any()].tolist())


# In[54]:


prediction = clf.predict(df_test[features])


# In[46]:


print(df_test["TRIPID"])
print(len(prediction))


# In[47]:


df_test['prediction'] = prediction


# In[48]:


df_test


# In[52]:


df_test.to_csv('D:/test.csv')


# In[ ]:




