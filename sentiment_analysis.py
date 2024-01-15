
import streamlit as st
#from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import re,string,unicodedata
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
import pickle
nlp = spacy.load("en_core_web_sm")

user_input = st.text_input("Enter a Sentence", "Default Text")
button_clicked = st.button('Predict')

class TextCleaner():
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == str:
            return self.clean_text(X)
        else:
            return [self.clean_text(text) for text in X]
    def clean_text(self, text):
        text = str(text).lower()  # Make text lowercase
        text = re.sub('\[.*?\]', '', text)  # Remove any sequence of characters in square brackets
        text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove links
        text = re.sub('<.*?>+', '', text)  # Remove HTML tags
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
        text = re.sub('\n', '', text)  # Remove newline characters
        text = re.sub('\w*\d\w*', '', text)  # Remove words containing numbers
        text = re.sub(r'[^a-z/A-Z/0-9/ ]', '', text)  # Remove special characters
        return text

class stopwords():
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        if type(X) == str:
            return self.stopw(X)
        else:
            return [self.stopw(text) for text in X]
    
    def stopw(self,text):
        from nltk.corpus import stopwords
        stopwords = stopwords.words('english')
        stopwords=stopwords+['s','m','u','im','ye','id','atg','na','ta','gon','wan']
        text= ' '.join([x for x in text.split() if x not in stopwords])
        return text

class lemma():
    
    def __init__(self,lemma_model):
        self.lemma_model=lemma_model
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if type(X) == str:
            return [self.lemmatise(X)]
        else:
            return [self.lemmatise(text) for text in X]
    
    def lemmatise(self,text):
        return " ".join([token.lemma_ for token in self.lemma_model(text)])
        
# import platform 
# import pathlib 
# plt = platform.system() 
# if plt == 'Linux':
#     pathlib.WindowsPath = pathlib.PosixPath
    
from pathlib import Path
# Load the pickled model
model_path =  Path('sentiment_analysis.pkl')
with open(model_path , 'rb') as file:
    model = pickle.load(file)

# with open('sentiment_analysis.pkl', 'rb') as file:
#     model = pickle.load(file)
    
if button_clicked:
    pred=model.predict(user_input)
    if pred==1:
        st.markdown("# Positive")
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1C4VPejYDvywKmk12MHyeH1z0ubr0E1A8lg&usqp=CAU')
    else:
        st.markdown("# Negative")
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSbbuDRvaFBgko-Kox-TUykBQFIqGU7p5SWt5kFoKK1p9B_LQWlPbswDfiJH6RpEGfqQbY&usqp=CAU')

