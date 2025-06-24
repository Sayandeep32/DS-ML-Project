import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')
data = df.where(pd.notnull(df), '')

data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

x = data['Message']
y = data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)
# Transforming the text data into numerical data using TF-IDF Vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

Model = LogisticRegression()
Model.fit(X_train_features, Y_train)
# Making predictions and calculating accuracy(on traing data)
prediction_on_training_data = Model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
# Making predictions and calculating accuracy(on testing data)
prediction_on_testing_data = Model.predict(X_test_features)
accuracy_on_testing_data = accuracy_score(Y_test, prediction_on_testing_data)
#Details of the model
st.title("Spam/Ham Mail Classifier")
st.write(f"Training Accuracy: {accuracy_on_training_data*100:.2f}%")
st.write(f"Testing Accuracy: {accuracy_on_testing_data*100:.2f}%")
# User input for spam/ham classification
user_input = st.text_input("Enter the message to check if it is spam or ham:")

if user_input:
    input_features = feature_extraction.transform([user_input])  # Pass as list
    prediction = Model.predict(input_features)
    if prediction[0] == 1:
        st.write("It is a HAM mail")
    else:
        st.write("It is a SPAM mail")
