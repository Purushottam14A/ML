#!/usr/bin/env python
# coding: utf-8

# In[59]:


import sklearn


# In[3]:


import pandas as pd


# In[4]:


total_data=pd.read_csv(r"C:\Users\kshat\Downloads\archive.zip")


# In[5]:


total_data.describe


# In[6]:


X=total_data.iloc[:,0:7]


# In[83]:


X.info


# In[8]:


y=total_data.iloc[:,7]


# In[9]:


y.describe


# In[60]:


from sklearn import svm


# In[61]:


from sklearn.svm import SVC


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.8)


# In[64]:


X_test.size


# In[65]:


from sklearn.preprocessing import StandardScaler


# In[66]:


y_test.size


# In[68]:


sc=StandardScaler()


# In[69]:


X_train=sc.fit_transform(X_train)


# In[70]:


X_test=sc.transform(X_test)


# In[71]:


clf=svm.SVC()


# In[72]:


clf.fit(X_train,y_train)


# In[81]:


pred_clf=clf.predict(X_test)


# In[82]:


pred_clf


# In[79]:


sklearn.metrics.accuracy_score(y_test,pred_clf)


# In[80]:


print(sklearn.metrics.classification_report(y_test,pred_clf))


# In[ ]:




