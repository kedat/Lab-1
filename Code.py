import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Loading data from csv file 
data = pd.read_csv("iris1.csv")
#Get first 150 rows of data
print(data.head(150))
#LGet information by take from the data at first column to the column before the last 
x_data = data.iloc[:,0:-1]
#Get label of training data, this is the final column of each data sample
y_data = data['class_type'].values

#Encode the data to numeric format
le = LabelEncoder()
x_data = x_data.apply(le.fit_transform)
#Spilt data to training sample and test sample

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=1)
#Apply classify models, it can be Gaussian, Bernoulli, Multinomial, ...
model = GaussianNB()
model.fit(X_train, y_train)
y_du_bao = model.predict(X_test)

#Evaluate the accuracy and print the confusion matrix

print('accuracy = ', accuracy_score(y_test, y_du_bao))
cnf_matrix = confusion_matrix(y_test, y_du_bao)
print('Ma trận nhầm lẫn:')
print(cnf_matrix)

