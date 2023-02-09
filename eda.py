#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/pearlbop08/Datasets/main/hotel_bookings.csv")


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df=df.drop(["company"],axis=1)


# In[7]:


df.shape


# In[8]:


median1=df["children"].median()
median1


# In[9]:


mean1=df["children"].mean()
mean1


# In[10]:


df[("children")].replace(np.nan,median1,inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


median2=df["agent"].median()
median2


# In[13]:


df["agent"].replace(np.nan,median2,inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


mode1=df["country"].mode().values[0]
mode1


# In[16]:


df["country"].replace(np.nan,mode1,inplace=True)


# In[17]:


df.isnull().sum()


# In[18]:


#check for duplicates
duplicate=df.duplicated()
print(duplicate.sum())


# In[19]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[20]:


df.shape


# In[21]:


#outliers


# In[22]:


df.boxplot(column=["lead_time"])


# In[23]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-1.5*IQR
    upper_range=Q3+1.5*IQR
    return lower_range,upper_range


# In[24]:


low_leadtime,high_leadtime=remove_outlier(df["lead_time"])
df["lead_time"]=np.where(df["lead_time"]>high_leadtime,high_leadtime,df["lead_time"])
df["lead_time"]=np.where(df["lead_time"]<low_leadtime,low_leadtime,df["lead_time"])


# In[25]:


df.boxplot(column=["lead_time"])


# In[26]:


df.boxplot(column=["adults"])


# In[27]:


pd.get_dummies(df)


# In[ ]:





# In[28]:


dummies=pd.get_dummies(df[['hotel','arrival_date_month','reservation_status_date','customer_type','deposit_type','reserved_room_type','reservation_status','assigned_room_type','market_segment','distribution_channel','meal','country']])


# In[29]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:




