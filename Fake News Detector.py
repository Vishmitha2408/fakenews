#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords # the for of in with
from nltk.stem.porter import PorterStemmer # loved loving == love
from sklearn.feature_extraction.text import TfidfVectorizer # loved = [0.0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[25]:


news_df = pd.read_csv('train.csv')


# In[26]:


news_df.head()


# In[27]:


news_df.shape


# In[28]:


news_df.isna().sum()


# In[29]:


news_df = news_df.fillna(' ')


# In[30]:


news_df.isna().sum()


# In[31]:


news_df['content'] = news_df['author']+" "+news_df['title']


# In[32]:


news_df


# In[33]:


news_df['content']


# In[34]:


# stemming
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[35]:


news_df['content'] = news_df['content'].apply(stemming)


# In[37]:


news_df['content']


# In[38]:


X = news_df['content'].values
y = news_df['label'].values


# In[39]:


print(X)


# In[40]:


vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)


# In[41]:


print(X)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1)


# In[43]:


X_train.shape


# In[45]:


X_test.shape


# In[46]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[47]:


train_y_pred = model.predict(X_train)
print("train accurracy :",accuracy_score(train_y_pred,y_train))


# In[48]:


test_y_pred = model.predict(X_test)
print("train accurracy :",accuracy_score(test_y_pred,y_test))


# In[56]:


# prediction system

input_data = X_test[20]
prediction = model.predict(input_data)
if prediction[0] == 1:
    print('Fake news')
else:
    print('Real news')


# In[55]:


news_df['content'][20]


# In[ ]:




