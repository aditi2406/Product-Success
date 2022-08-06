#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import random
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[2]:


products = pd.read_csv('flipkart_com-ecommerce_sample.csv')


# In[3]:


products.head()


# In[4]:


len(products['product_name'])
products = products[products['description'].notna()]
print(products.isnull().any())
print(products['overall_rating'].value_counts())
len(products['product_name'])


# In[5]:


#filling the missing datas
ratings = products['overall_rating']
list1 = np.arange(1,5,0.1)
for index, values in ratings.iteritems():
    if values == "No rating available":
        ratings[index] = round(random.choice(list1),1)
print(ratings.value_counts())


# In[6]:


#visualize the ratings count
ratings.value_counts().hist(bins=[0,1,2,3,4,5])


# In[7]:


#extracting keywords from product category
import csv        
cat_df = products['product_category_tree'].to_numpy()
category = []
def listToString(s): 
    str1 = ""   
    for ele in s: 
        str1 += ele  
    return str1 


for i in range(len(cat_df)):
    temp = cat_df[i].split(">>")
    temp = listToString(temp)
    category.append(temp[2:len(temp)-2])
print(category[0])


# In[8]:


#merging product name and description
name_df = products['product_name'].to_numpy()
des_df = products['description'].to_numpy()
inputs_words = []
print(len(name_df))
print(len(des_df))
print(len(category))
for i in range(len(name_df)):
    temp = name_df[i].split()
    temp1 = des_df[i].split() 
    temp2 = category[i].split()
    temp_list = list(set().union(temp,temp1,temp2))
    inputs_words.append(temp_list)
print(inputs_words[0])
print(len(inputs_words[0]))


# In[9]:


#reviews setup
#review less than 3 will be counted as bad product (0) orelse good product (1)
score_df = ratings.to_numpy()
ratings_score = []
for i in range(len(score_df)):
    temp = score_df[i]
    if int(float(temp)) < 3 :
        ratings_score.append(0)
    else:
        ratings_score.append(1)
print(ratings_score[1])


# In[10]:


#stopwords and stemming
data = []
for i in range(len(inputs_words)):
    review = ' '
    review = review.join(inputs_words[i])
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
print(data[0])


# In[11]:


#countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()
pickle.dump(cv, open("vector.pickel", "wb"))
y = np.array(ratings_score)
print(x[0])
print(y[1])
print(type(y))


# In[12]:


print(x.shape)
print(type(x))


# In[13]:


#model building
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 100 ,activation = "relu",batch_size=2048))
model.add(Dense(units = 200 ,activation = "relu",batch_size=2048))

model.add(Dense(units = 1 ,activation = "sigmoid",batch_size=2048))
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
model.fit(x_train,y_train,epochs  = 10000)


# In[14]:


x_test.shape


# In[15]:


#predicting the y values
y_pred = model.predict(x_test)
y_pred = (y_pred>0.5)


# In[16]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
acc


# In[17]:


y_pred


# In[18]:


y_test


# In[19]:


model.save("productsentiment.h5")


# In[20]:


from keras.models import load_model
model1 = load_model("productsentiment.h5")


# In[21]:


input_word = ['in', 'Key', 'shorts', 'Clothing', 'Features', 'Lukewarm', 'the', 'Contents', "Women's", 'Ideal', 'Pack', 'of', 'Do', 'Gentle', 'Style', 'Code', 'Shorts', 'Additional', 'Number', 'In', 'Care', 'Swimwear', 'ALTHT_3P_21', 'Lingerie,', 'Wash', 'Water,', 'Alisha', 'Cotton', 'Package', 'Not', 'Sleep', 'Red,', 'Details', 'Fabric', 'Pattern', 'Machine', 'Box', 'Bleach', 'Cycling', 'Navy,', 'General', 'For', 'Lycra', '&', 'Type', '3', 'Sales', 'Solid', 'Navy,Specifications']


# In[22]:


review = ' '
review = review.join(input_word)
review = re.sub('[^a-zA-Z]', ' ',review)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
input_word = review
data1 = []
data1.append(input_word)


# In[23]:


data1[0]


# In[24]:


vectorizer = pickle.load(open("vector.pickel", "rb"))
x1 = vectorizer.transform(data1).toarray()


# In[25]:


x1.shape


# In[26]:


y_pred = model1.predict(x1)
y_pred = (y_pred>0.5)


# In[27]:


y_pred


# In[ ]:




