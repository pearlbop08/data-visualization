#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/titanic-training-data.csv")


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


sns.pairplot(df,diag_kind="kde")


# In[7]:


df.dtypes


# In[8]:


df.drop("Cabin",axis=1,inplace=True)


# In[9]:


df.shape


# In[10]:


mean1=df["Age"].mean()
mean1


# In[11]:


df[("Age")].replace(np.nan,mean1,inplace=True)


# In[12]:


median2=df["Embarked"].mode()
median2


# In[13]:


mode1=df["Embarked"].mode().values[0]
mode1


# In[14]:


df.head()


# In[15]:


df["Sex"]=df["Sex"].replace({1:"female", 2:"male"})
df.sample(10)


# In[16]:


df=pd.get_dummies(df,columns=["Sex"])
df.sample(10)


# In[17]:


df["Embarked"]=df["Embarked"].replace({1:"S", 2:"C", 3:"Q"})
df.sample(10)


# In[18]:


df=pd.get_dummies(df,columns=["Embarked"])
df.sample(10)


# In[19]:


df.drop("Name",axis=1,inplace=True)


# In[20]:


df.drop("Ticket",axis=1,inplace=True)


# In[21]:


df.drop("Fare",axis=1,inplace=True)


# In[22]:


df.drop("PassengerId",axis=1,inplace=True)


# In[23]:


df.head()


# In[24]:


df["Pclass"]=df["Pclass"].replace({1:"1", 2:"2", 3:"3"})
df.sample(10)


# In[25]:


df=pd.get_dummies(df,columns=["Pclass"])
df.sample(10)


# In[26]:


X = df.drop(["Survived"], axis=1)
#dependent variable
Y= df[["Survived"]]


# In[27]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y ,test_size=0.30, random_state=1)


# In[28]:


model_1 = LogisticRegression()
model_1.fit(X_train,Y_train)


# In[29]:


model_1.score(X_train, Y_train)
model_1.score(X_test, Y_test)


# In[30]:


predictions=model_1.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)


# In[32]:


from sklearn import metrics


# In[33]:


print(metrics.classification_report(Y_test,predictions))


# In[34]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,predictions)


# In[35]:


from sklearn.metrics import classification_report
classification_report(Y_test,predictions)


# In[36]:


from sklearn import metrics


# In[37]:


cm=metrics.confusion_matrix(Y_test,predictions,labels=[1,0])
df_cm=pd.DataFrame(cm,index=[i for i in["1","0"]],
columns=[i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:





# In[ ]:




