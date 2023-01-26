#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas
from matplotlib import pyplot as plt
import scipy
import seaborn as sns
print(sns.get_dataset_names())
df=sns.load_dataset('car_crashes')
print(df.head())





# In[20]:


plt.scatter(df.speeding,df.alcohol)
plt.show()


# In[25]:



plt.scatter(df.speeding,df.alcohol)
sns.set_style('whitegrid')
plt.show()


# In[26]:


tips=sns.load_dataset('tips')
tips.head()


# In[27]:


sns.relplot(data=tips,x="total_bill", y='tip')


# In[28]:


sns.relplot(data=tips,x="total_bill", y='tip', hue="day")


# In[34]:


sns.relplot(data=tips,x="total_bill", y='tip' , hue="sex", col='day',col_wrap=3)


# In[36]:


#histogram
df=sns.load_dataset('iris')
sns.displot(df['petal_length'],kde=False)


# In[ ]:




