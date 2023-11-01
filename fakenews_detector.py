#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!pip install streamlit nltk


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load data
data = pd.read_csv("fake_or_real_news.csv")

# Text preprocessing: Remove stopwords and apply lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

data["title"] = data["title"].apply(preprocess_text)

x = np.array(data["title"])
y = np.array(data["label"])

# Use TfidfVectorizer
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(xtrain, ytrain)

# Evaluate model on test set
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(ytest, ypred))

# Streamlit application
st.title("Fake News Detection System")

def fakenewsdetection():
    user = st.text_area("Enter Any News Headline: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = preprocess_text(user)
        data = vectorizer.transform([sample]).toarray()
        prediction = model.predict(data)
        probability = model.predict_proba(data)
        confidence = np.max(probability)
        st.title(f"Prediction: {prediction[0]}, Confidence: {confidence:.2f}")

fakenewsdetection() 


# In[ ]:




