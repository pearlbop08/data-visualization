#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/auto-mpg.csv")


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


duplicate=df.duplicated()
print(duplicate.sum())


# In[6]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[7]:


df.shape


# In[8]:


df.boxplot(column=["mpg"])


# In[9]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-1.5*IQR
    upper_range=Q3+1.5*IQR
    return lower_range,upper_range


# In[10]:


low_mpg,high_mpg=remove_outlier(df["mpg"])
df["mpg"]=np.where(df["mpg"]>high_mpg,high_mpg,df["mpg"])
df["mpg"]=np.where(df["mpg"]<low_mpg,low_mpg,df["mpg"])


# In[11]:


df.boxplot(column=["mpg"])


# In[12]:


df.head()


# In[14]:


median1=df["horsepower"].median
median1


# In[19]:


df["horsepower"]=df['horsepower'].replace("?",np.nan)


# In[20]:


df["horsepower"]=df['horsepower'].astype(float)


# In[21]:


df.dtypes


# In[23]:


df["horsepower"].replace(np.nan,median1,inplace=True)


# In[24]:


df.isnull().sum()


# In[ ]:




