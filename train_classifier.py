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

import numpy as np
import warnings
import pickle
import argparse

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

parser = argparse.ArgumentParser()

parser.add_argument("db_path" )
parser.add_argument("model_path" )

args = parser.parse_args()

# load data from database
engine = create_engine('sqlite:///{}'.format(args.db_path))
df = pd.read_sql_table('table1', engine)
categories = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
X = df['message']
Y = df[categories]

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

pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(estimator = RandomForestClassifier()))
        ])

X_train, X_test, y_train, y_test = train_test_split(X, Y)

parameters = {
        'vect__stop_words': [None, 'english'],
        'clf__estimator__n_estimators': [5, 10, 20],
        'clf__estimator__min_samples_split': [2, 3]
    }
cv = GridSearchCV(pipeline, param_grid=parameters)

y_true = y_train.values
gridsearch = cv.fit(X_train, y_true)

best = gridsearch.best_estimator_
b_pred = best.predict(X_test)
y_truth = y_test.values

warnings.filterwarnings("error")
for cat in np.arange(0, 36):
    in_out = ['Not {}'.format(categories[cat]), '{}'.format(categories[cat])]
    try:
        class_report = classification_report(y_truth[cat], b_pred[cat],target_names = in_out)
    except:
        class_report = 'UndefinedMetricWarning'
    print ('Score for {}:\n{}'.format(categories[cat],class_report))

pickle_file = open(args.model_path, 'wb')
pickle.dump(best, pickle_file)
pickle_file.close()
