import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
#doc du lieu tu file csv co tieu de cac cot
#data = pd.read_csv("data\student-por.csv")
#data = pd.read_csv("data\zoo1.csv")
data = pd.read_csv("play_tennis.csv")
#Lay thong tin mo ta
x_data = data.iloc[:,0:-1]
#Lay nhan lop cua bo du lieu, ‘class_type’ la tieu de cot phan lop trong Bo du lieu iris1.csv
y_data = data['class_type'].values
y_data_trimmed = [value.strip() for value in y_data]
#chuyen gia trị du lieu ve dang so
le = LabelEncoder()
x_data = x_data.apply(le.fit_transform)
#Chia du lieu huan luyen (X_train, y_train)
X_train= x_data[:10]
X_test= x_data[10:14]
y_train=y_data_trimmed[:10]
y_test= y_data_trimmed[10:14]
# X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=1)
# print(X_test)
# print(X_test)
print(y_train)
print(y_test)
# Ap dung mo hinh phan lop
gaussian = GaussianNB()
bernoulli = BernoulliNB()
model.fit(X_train, y_train) 
y_du_bao = model.predict(X_test)

#Xac dinh hieu nang bo phan lop theo cac chi so

from sklearn.metrics import accuracy_score, confusion_matrix
print('accuracy = ', accuracy_score(y_test, y_du_bao))

