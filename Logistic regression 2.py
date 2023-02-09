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


df=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/pima-indians-diabetes-2.csv")


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


sns.pairplot(df,diag_kind="kde")


# In[7]:


df.head()


# In[40]:


X = df.drop(["class"], axis=1)
#dependent variable
Y= df[["class"]]


# In[41]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y ,test_size=0.30, random_state=1)


# In[42]:


model_1 = LogisticRegression()
model_1.fit(X_train,Y_train)


# In[43]:


model_1.score(X_train, Y_train)
model_1.score(X_test, Y_test)


# In[44]:


predictions=model_1.predict(X_test)


# In[45]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)


# In[46]:


from sklearn import metrics


# In[47]:


print(metrics.classification_report(Y_test,predictions))


# In[48]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,predictions)


# In[49]:


from sklearn.metrics import classification_report
classification_report(Y_test,predictions)


# In[50]:


from sklearn import metrics


# In[51]:


cm=metrics.confusion_matrix(Y_test,predictions,labels=[1,0])
df_cm=pd.DataFrame(cm,index=[i for i in["1","0"]],
columns=[i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:




