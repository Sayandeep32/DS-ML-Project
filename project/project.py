import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('mail_data.csv')
data=df.where(pd.notnull(df),'')

data.loc[data['Category']=='spam','Category']=0
data.loc[data['Category']=='ham','Category']=1
# Splitting the dataset into training and testing sets
x=data['Message']
y=data['Category']

#split dataset
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#Transform text data into feature vectors that can be used as input to the logistic regression
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
#Train the model with Training data
Model=LogisticRegression()
Model.fit(X_train_features,Y_train)
# Prediction of data (for training data)
prediction_on_training_data=Model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)

# Prediction of data (for testing data)
prediction_on_testing_data=Model.predict(X_test_features)
accuracy_on_testing_data=accuracy_score(Y_test,prediction_on_testing_data)

input=["Free Msg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, Â£1.50 to rcv"]
input_features=feature_extraction.transform(input)
# predict
prediction=Model.predict(input_features)
# print(prediction)
if prediction[0]==1:
    print("Ham mail")
else:
    print("Spam mail")