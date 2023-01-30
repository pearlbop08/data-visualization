#!/usr/bin/env python
# coding: utf-8

# In[37]:


import seaborn as sns


# In[17]:


df=sns.load_dataset("titanic")


# In[19]:


df.head()


# In[26]:


df.isnull().sum()


# In[28]:



nan_deck=df["deck"].value_counts(dropna=False)


# In[24]:


nan_deck


# In[29]:


df.shape


# In[30]:


df_age=df.dropna(subset=["age"],how="any",axis=0)
len(df_age)


# In[31]:


mean_age=df["age"].mean()


# In[33]:


mean_age


# In[34]:


df["age"].fillna(mean_age,inplace=True)


# In[36]:


df.head(10)


# In[39]:


mode_embark=df["embark_town"].value_counts(dropna=True).idxmax()
mode_embark


# In[41]:


df["embark_town"].fillna(mode_embark,inplace=True)


# In[42]:


df.isnull().sum()


# In[43]:


df["embark_town"][825:830]


# In[ ]:




