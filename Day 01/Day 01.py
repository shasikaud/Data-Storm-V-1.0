#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the pre-processed datasets
dataset_train = pd.read_csv('credit_card_default_train.csv')
dataset_test = pd.read_csv('credit_card_default_test.csv')

 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Endoing and labelling the training set
labelencoder_1 = LabelEncoder()
dataset_train.iloc[:,2] = labelencoder_1.fit_transform(dataset_train.iloc[:,2])
labelencoder_2 = LabelEncoder()
dataset_train.iloc[:,3] = labelencoder_2.fit_transform(dataset_train.iloc[:,3])
labelencoder_3 = LabelEncoder()
dataset_train.iloc[:,4] = labelencoder_3.fit_transform(dataset_train.iloc[:,4])
labelencoder_4 = LabelEncoder()
dataset_train.iloc[:,5] = labelencoder_4.fit_transform(dataset_train.iloc[:,5])

#Endoing and labelling the test set
labelencoder_1a = LabelEncoder()
dataset_test.iloc[:,2] = labelencoder_1a.fit_transform(dataset_test.iloc[:,2])
labelencoder_2a = LabelEncoder()
dataset_test.iloc[:,3] = labelencoder_2a.fit_transform(dataset_test.iloc[:,3])
labelencoder_3a = LabelEncoder()
dataset_test.iloc[:,4] = labelencoder_3a.fit_transform(dataset_test.iloc[:,4])
labelencoder_4a = LabelEncoder()
dataset_test.iloc[:,5] = labelencoder_4a.fit_transform(dataset_test.iloc[:,5])

#Splitting the training set into dependent and independent sets
x_train = dataset_train.iloc[:, 1:24].values
y_train = dataset_train.iloc[:, 24].values

#Independent variable set for the test set
x_test = dataset_test.iloc[:, 1:24].values

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Feature scaling training set
x_train = sc.fit_transform(x_train)
#Feature scaling test set
x_test = sc.fit_transform(x_test)

#XG boost classifier
import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(x_train, y_train)

#Predicting the training set
y_pred = classifier.predict(x_train)
#Predicting the test set
y_pred_test = classifier.predict(x_test)

#Calculating the conusion matrix for the training set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

#Calculating the f1 score for the training set
from sklearn.metrics import classification_report
repo = classification_report(y_train,y_pred)
