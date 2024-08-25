import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Website styling
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #1e1e2f;  /* Set background to a dark theme */
        color: #FFFFFF;  /* Set font color to white */
    }
    .main {
        background-color: #2e2e3f;  /* Dark background for the main content */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
    }
    .stTextInput > div > div > input {
        background-color: #3e3e4f;  /* Darker background for input */
        border-radius: 5px;
        padding: 10px;
        font-size: 1.2em;
        color: #FFFFFF;  /* Set input text color to white */
    }
    .stButton > button {
        background-color: #007BFF;  /* Primary button color */
        color: white;
        font-size: 1.2em;
        border-radius: 8px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
    }
    .stTitle {
        color: #FFFFFF;  /* Set title color to white */
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        font-weight: bold;
        font-size: 2.5em;
    }
    .stMarkdown {
        font-size: 1.1em;
        color: #d0d0d0;  /* Slightly lighter text color */
    }
    .stAlert {
        border-radius: 8px;
    }
    hr {
        border: 1px solid #444;
    }
    </style>
    """, unsafe_allow_html=True)

# Website title
st.title('📰 Fake News Detector')

# Input area
st.subheader("Enter the News Article Below:")
input_text = st.text_input('', placeholder='Type or paste your news article here...')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# Prediction and display
if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.error('🚨 The News is FAKE')
    else:
        st.success('✅ The News Is REAL')

# Footer
st.markdown("""
    <hr>
    <div style="text-align: center;">
        <small>Built with ❤️ using Streamlit | Powered by Shreyas</small>
    </div>
    """, unsafe_allow_html=True)
