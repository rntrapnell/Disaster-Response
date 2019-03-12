#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[4]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)
categories = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
X = df['message']
Y = df[categories]


# ### 2. Write a tokenization function to process your text data

# In[13]:


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    
    urls = re.findall(url_regex, text)
    for url in urls:
        text.replace(url, 'linktext')
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    cleaned_lemmed_list = []
    for word in words: 
        cleaned_lemmed_word = lemmatizer.lemmatize(word).lower().strip()
        cleaned_lemmed_list.append(cleaned_lemmed_word)
    return cleaned_lemmed_list


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[14]:


pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator = RandomForestClassifier()))
        ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)


# In[20]:



pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[21]:


y_pred = pipeline.predict(X_test)


# In[39]:


df_yp = pd.DataFrame(y_pred)
df_yt = pd.DataFrame(y_test)


# In[23]:


y_true = y_test.values


# In[31]:


#for truth, prediction in np.ndindex(y_true), np.ndindex(y_pred):
warnings.filterwarnings("error")
for cat in np.arange(0, 36):
    in_out = ['Not {}'.format(categories[cat]), '{}'.format(categories[cat])]
    try:
        class_report = classification_report(y_true[cat], y_pred[cat],target_names = in_out)
    except:
        class_report = 'UndefinedMetricWarning'
    print ('Score for {}:\n{}'.format(categories[cat],class_report))
    


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[33]:


pipeline.get_params()


# In[15]:


from sklearn.metrics import make_scorer, recall_score
parameters = {
        'vect__stop_words': [None, 'english'],
        'clf__estimator__n_estimators': [5, 10, 20],
        'clf__estimator__min_samples_split': [2, 3]
    }
#scorer = make_scorer(recall_score)
cv = GridSearchCV(pipeline, param_grid=parameters)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[16]:


y_true = y_train.values
gs = cv.fit(X_train, y_true)


# In[17]:


best = gs.best_estimator_


# In[18]:


bp = best.predict(X_test)


# In[19]:


y_truth = y_test.values


# In[20]:


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
#https://stackoverflow.com/questions/5644836/in-python-how-does-one-catch-warnings-as-if-they-were-exceptions
warnings.filterwarnings("error")
acc_list = []

for cat in np.arange(0, 36):
    no_samp = 'No Predicted Samples'
    acc = accuracy_score(y_truth[cat], bp[cat])
    try:
        pre = precision_score(y_truth[cat],bp[cat])
        rec = recall_score(y_truth[cat],bp[cat])
    except: 
        #UndefinedMetricWarning:
        pre = no_samp
        rec = no_samp
    
    print ('Scores for {}: Accuracy-{}, Prescision-{}, Recall-{}'.format(categories[cat], acc, pre, rec))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# ### 9. Export your model as a pickle file

# In[21]:


import pickle
pickle_file = open('model_save', 'wb')
pickle.dump(best, pickle_file)
pickle_file.close()


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




