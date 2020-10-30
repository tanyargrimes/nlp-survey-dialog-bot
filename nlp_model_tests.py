# !/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
PROG8420 - Programming for Big Data

Group 3 Assignment

Created on Sat Aug  8 10:22:50 2020

@authors: 
    Jaibir Singh
    Tanya Grimes
"""

import nltk
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


nltk.download('stopwords')


#--------------------------------- Load Data


# Please ensure the current working directory is set to the location of this project folder


data = pd.read_csv('yelp.csv', names=['business_id','date','review_id','stars','text','type','user_id','cool','useful','funny'], sep=',')

# remove first row with labels
data = data.drop(data.index[0])


# Initial exploring of data
# print(data.head())
# print(data['stars'].value_counts())
# print('Columns', data.columns)
# print('Column data types', data.dtypes)
# print('No. of elements', data.size)
# print('No. of records', data.shape[0])


#--------------------------------- Data Clean up and Transformation


# Step 1: filter data to stars 1 and 5 for extreme values
#data = data[data.stars.isin(['1','5'])]
# validation dataset dominated by 5s, 3:1

# Step 2: combine stars 1 and 2 to balance out with 5s
data = data[data.stars.isin(['1','2','5'])]
data = data.replace({'stars':'2'}, '1')
# validation still dominated by 5s, 2:1

# Step 3: Filter out 5-star results with funny greater than 0.
data = data.drop(data[(data['stars'] == '5') & (data['funny'] != '0')].index)

print(data.head())
print(data['stars'].value_counts())


# add functions to clean, tokenize and stem the data
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer() #

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    
    return text


# #####  Split Data into train, validate and test datasets

# create separate datasets for the X and y data
X = data[['text']]
y = data['stars']

# split first to get a training dataset of 90% of the data
X_train, X_group, y_train, y_group = train_test_split(X, y, test_size = 0.1, random_state = 1, stratify = y)

# split group into validate and test datasets, each having 5% of the overall data
X_validate, X_test, y_validate, y_test = train_test_split(X_group, y_group, test_size = 0.5, random_state = 1, stratify = y_group)

# confirms 90-5-5 split in datsets
print(y_train.shape[0])
print(y_validate.shape[0])
print(y_test.shape[0])


#####  Vectorize text


# create Tf Idf vectorizer that stores the clean_text function 
tfidf_vectorizer = TfidfVectorizer(analyzer = clean_text)

# fit the X training data to the vectorizer and will trigger cleaning of text
tfidf_vfit = tfidf_vectorizer.fit(X_train['text'])

# generate vecotrized data for each dataset
tfidf_train = tfidf_vfit.transform(X_train['text'])
tfidf_validate = tfidf_vfit.transform(X_validate['text'])
tfidf_test = tfidf_vfit.transform(X_test['text'])



#---------------------------------  Validation of models

# 1. Random Forest Classifier

# higher estimator, the better the accuracy so far. 20 vs. 200
class_rnf_val = RandomForestClassifier(n_estimators = 200, max_depth = None, n_jobs = -1)

rnf_val_model = class_rnf_val.fit(tfidf_train, y_train)

y_val_rnf_pred = rnf_val_model.predict(tfidf_validate)

precision, recall, fscore, train_support = score(y_validate, y_val_rnf_pred, pos_label='5', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_validate, y_val_rnf_pred), 3)))


# Making the Confusion Matrix
cm = confusion_matrix(y_validate, y_val_rnf_pred)
class_label = ['1', '5']
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Star')
plt.ylabel('Actual Star')
plt.show()


# 2. Logistic Regression 

class_lgt_val = LogisticRegression()

lgt_val_model = class_lgt_val.fit(tfidf_train, y_train)

y_val_lgt_pred = lgt_val_model.predict(tfidf_validate)

precision, recall, fscore, train_support = score(y_validate, y_val_lgt_pred, pos_label='5', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_validate, y_val_lgt_pred), 3)))


# Making the Confusion Matrix
cm = confusion_matrix(y_validate, y_val_lgt_pred)
class_label = ['1', '5']
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Star')
plt.ylabel('Actual Star')
plt.show()


# 3. Naive Bayes Classifier

class_nbc_val = MultinomialNB()

lgt_nbc_model = class_nbc_val.fit(tfidf_train, y_train)

y_val_nbc_pred = lgt_nbc_model.predict(tfidf_validate)

precision, recall, fscore, train_support = score(y_validate, y_val_nbc_pred, pos_label='5', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_validate, y_val_nbc_pred), 3)))

# Making the Confusion Matrix
cm = confusion_matrix(y_validate, y_val_nbc_pred)
class_label = ['1', '5']
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Star')
plt.ylabel('Actual Star')
plt.show()



#---------------------------------  Final evaluation of selected model


class_lgt_test = LogisticRegression()

lgt_test_model = class_lgt_test.fit(tfidf_train, y_train)

y_test_lgt_pred = lgt_test_model.predict(tfidf_test)

precision, recall, fscore, train_support = score(y_test, y_test_lgt_pred, pos_label='5', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_test, y_test_lgt_pred), 3)))


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_test_lgt_pred)
class_label = ['1', '5']
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Star')
plt.ylabel('Actual Star')
plt.show()


# Export the model to current path (renamed to prevent an overwrite)
joblib.dump(tfidf_vfit, 'vectorizerFit_New.joblib')
joblib.dump(lgt_test_model, 'logisticRegressionModel_New.joblib')



#---------------------------------  Test model classification results

print('Please ensure the current working directory is set to the location of this project folder')


# Creating response
def bot_answer(q):
    process_text = tfidf_vfit.transform([q]).toarray()
    prob = lgt_val_model.predict_proba(process_text)[0]
    print(prob)
    # returns index of maxium probability
    max_ = np.argmax(prob)
    print(max_)
      
    
# Some answer to test model responses    
bot_answer('I hated the atmosphere. It was so gloomy...') 
bot_answer('Disgusting! I felt sick afterwards!')
bot_answer('Produce quite bad quality today.')
bot_answer('The food was very bland. Would not be returning any time soon.')
bot_answer('''Disgusting!  Had a Groupon so my daughter and I tried it out.  
           Very outdated and gaudy 80\'s style interior made me feel like I was in
           an episode of Sopranos.  The food itself was pretty bad.  
           We ordered pretty simple dishes but they just had no flavor at all!  
           After trying it out I\'m positive all the good reviews on here are 
           employees or owners creating them.''')    
bot_answer('A courtroom sketch artist obviously has clear manga influences.')
bot_answer('I had a great time on my birthday! The waiters were amazing!')







