import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ✅ Cache data loading function
@st.cache_data
def load_data():
    cols = ['author', 'title', 'label']
    # Limit rows for faster loading
    news_df = pd.read_csv('train.csv', usecols=cols, nrows=5000) 
    news_df = news_df.fillna(' ')
    news_df['content'] = news_df['author'] + ' ' + news_df['title']
    return news_df

# Load data
news_df = load_data()

# ✅ Preload stopwords and stemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ✅ Stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# Apply stemming to content column
news_df['content'] = news_df['content'].apply(stemming)

# ✅ Cache the model and vectorizer
@st.cache_resource
def train_model():
    X = news_df['content'].values
    y = news_df['label'].values

    # ✅ Reduce vector size for faster processing
    vector = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)
    X = vector.fit_transform(X)

    # ✅ Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # ✅ Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    return vector, model

# Train the model and vectorizer
vector, model = train_model()

# 🌐 Streamlit UI
st.title('📰 Fake News Detector')

# ✅ User input
input_text = st.text_input('Enter news article')

# ✅ Prediction function
def predict_news(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# ✅ Display Prediction Result
if input_text:
    pred = predict_news(input_text)
    if pred == 1:
        st.write('❌ **The News is Fake**')
    else:
        st.write('✅ **The News is Real**')

# ✅ Run Command:
# streamlit run app.py --server.enableCORS false --server.enableWebsocketCompression false
