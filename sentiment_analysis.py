
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
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('popular')
nltk.download('wordnet')
import gzip

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

class lemma:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == str:
            return [self.lemmatize(X)]
        else:
            return [self.lemmatize(text) for text in X]

    def lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_text = " ".join(lemmatized_tokens)
        return lemmatized_text

# class lemma():
    
#     def __init__(self,lemma_model):
#         self.lemma_model=lemma_model
        
#     def fit(self,X,y=None):
#         return self
    
#     def transform(self,X):
#         if type(X) == str:
#             return [self.lemmatise(X)]
#         else:
#             return [self.lemmatise(text) for text in X]
    
#     def lemmatise(self,text):
#         return " ".join([token.lemma_ for token in self.lemma_model(text)])
        
# import platform 
# import pathlib 
# plt = platform.system() 
# if plt == 'Linux':
#     pathlib.WindowsPath = pathlib.PosixPath
# import pathlib
# temp = pathlib.WindowsPath
# pathlib.WindowsPath = pathlib.PosixPath
# from pathlib import Path
# # Load the pickled model
# model_path =  Path('sentiment_analysis.pkl')
# with open(model_path , 'rb') as file:
#     model = pickle.load(file)

txt_clean=pickle.load(gzip.open('Text_preprocessing_3.pkl','rb'))
tf=pickle.load(gzip.open('vectorizer_3.pkl','rb'))
model=pickle.load(gzip.open('Sentiment_detector_3.pkl','rb'))


# with open('sentiment_analysis.pkl', 'rb') as file:
#     model = pickle.load(file)
    
if button_clicked:
    X=txt_clean.transform(user_input)
    X=tf.transform(X)   
    pred=model.predict(X)
    if pred==1:
        st.markdown("# The Sentence Seems to be POSITIVE")
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1C4VPejYDvywKmk12MHyeH1z0ubr0E1A8lg&usqp=CAU')
    elif pred==0:
        st.markdown("# The Sentence Seems to be NEGATIVE")
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSbbuDRvaFBgko-Kox-TUykBQFIqGU7p5SWt5kFoKK1p9B_LQWlPbswDfiJH6RpEGfqQbY&usqp=CAU')
    else:
        st.markdown("# The Sentence Seems to be NEUTRAL")
        st.image('https://assets-global.website-files.com/5bd07788d8a198cafc2d158a/61c49a62dccfe690ca3704be_Screen-Shot-2021-12-23-at-10.44.27-AM.jpg')


