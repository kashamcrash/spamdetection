#!/usr/bin/env python
# coding: utf-8

# In[67]:


# SMS Spam Classification
# Create a model that can predict whether the given sms is a spam or ham sms.


# In[68]:


import numpy as np
import pandas as pd


# In[69]:


data = pd.read_csv('C:\Datasets\SMSSpamCollection', sep='\t', names=['label','message'])


# In[70]:


data.info()


# In[71]:


data.head()


# In[72]:


# Balanced and Unbalanced Dataset in Classification

# Balanced Dataset  -> no of observations for each unique label is same.
#                      e.g. n(ham) = n(spam) ---> Balanced Dataset

# Unbalanced Dataset -> no of observations of each unique label is different.
#                      e.g. n(ham) not equal to n(spam) ---> Unbalanced Dataset

data.label.value_counts()


# In[73]:


# The given dataset is an Unbalanced Dataset. 


# In[74]:


# Preprocess Feature Column since given feature column (message) is Pure String

# 1. Perform Text Preprocessing
# 2. Create BoW
# 3. Apply TF-IDF on BoW


# In[75]:


# 1. Perform Text Preprocessing
#
# a. Remove Punctuations
# b. Convert Sentences to Words
# c. Remove Stopwords
# d. Normalize the words


# In[76]:


import nltk
from nltk.corpus import stopwords
import string
def textPreprocessor(feature):
    # a. Remove Punctuations
    removePunctuations = [character for character in feature if character not in string.punctuation]
    sentencesWithoutPunct = ''.join(removePunctuations)
    
    # b. Convert Sentences to Words -- Tokenization
    words = sentencesWithoutPunct.split(" ")
    
    # c. Normalize the words
    wordNormalized = [ word.lower() for word in words ]
    
    # d. Remove Stopwords
    finalWords = [word for word in wordNormalized if word not in stopwords.words('english')]
    
    return finalWords


# In[77]:


# Seperate data as features and label

features = data.iloc[:,[1]].values
features


# In[78]:


label = data.iloc[:,[0]].values
label


# In[79]:


# Creating BOW -- scikit-learn

# feature ---> textPreprocessor ---> Creating BOW( Vocab , Contigency Matrix)

from sklearn.feature_extraction.text import CountVectorizer
wordVector = CountVectorizer(analyzer=textPreprocessor)

# Building the Vocabulary
finalWordVectorVocab = wordVector.fit(features)


# In[80]:


finalWordVectorVocab.vocabulary_


# In[81]:


# Building BOW
bagOfWords = finalWordVectorVocab.transform(features)


# In[82]:


demo = bagOfWords.toarray()
demo


# In[83]:


len(finalWordVectorVocab.vocabulary_)


# In[84]:


demo.shape


# In[85]:


# Applying TFIDF on BOW

from sklearn.feature_extraction.text import TfidfTransformer

tfIdfObject = TfidfTransformer().fit(bagOfWords)


# In[86]:


finalFeatureArray = tfIdfObject.transform(bagOfWords)


# In[87]:


finalFeatureArray


# In[88]:


# import sys
# import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
# demo1 = finalFeatureArray.toarray()
# demo1


# In[89]:


# pd.DataFrame(finalFeatureArray.toarray()).to_csv('TFIDF.csv', index=False)


# In[90]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(finalFeatureArray,
                                                label,
                                                test_size=0.2,
                                                random_state=6)


# In[91]:


# Building the model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[92]:


# Model score
model.score(X_train,y_train)


# In[93]:


model.score(X_test,y_test)


# In[94]:


# Creating a function for the same
import warnings
warnings.filterwarnings("ignore")
for i in range(1,101):
    X_train,X_test,y_train,y_test = train_test_split(finalFeatureArray,
                                                label,
                                                test_size=0.2,
                                                random_state=i)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    trainScore = model.score(X_train,y_train)
    testScore = model.score(X_test,y_test)
    
    if testScore > trainScore and testScore > 0.9:
        print("Testing {} , Training {}, RS {}".format(testScore,trainScore,i))


# In[95]:


# The dataset is an unbalanced dataset, we need to check for precision and recall

from sklearn.metrics import confusion_matrix
confusion_matrix(label , model.predict(finalFeatureArray))


# In[96]:


from sklearn.metrics import classification_report
print(classification_report(label , model.predict(finalFeatureArray)))


# In[97]:


# Avg of Precision(Spam ) and Recall(Ham) --> 1 which is greater than CL --- Thus ACCEPTABLE !!!!


# In[98]:


#Deployment Check

smsInput = input("Enter SMS: ")

#Preprocess
preProcessedFeature = textPreprocessor(smsInput)

#BOW

bowFeature = finalWordVectorVocab.transform(preProcessedFeature)

#TFIDF

tfIDFFeature = tfIdfObject.transform(bowFeature)

#Pred

predLabel = model.predict(tfIDFFeature)[0]

print("Given SMS is {}".format(predLabel))


# In[99]:


# Thank You :)


# In[ ]:




