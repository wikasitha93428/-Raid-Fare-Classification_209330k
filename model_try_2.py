# -*- coding: utf-8 -*-
"""

@author: wikasitha
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV



#read traing data
df = pd.read_csv("E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\train.csv")

#read test data
df_test = pd.read_csv("E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\test.csv")

#plotting histoghrams 
df.hist(alpha=0.5, figsize=(20, 10))
plt.tight_layout()
plt.show()


#Removing NaN /Null values from the training data set 
df= df.dropna()

#Clencing data 
X = df.drop(['label','pickup_time','drop_time'], axis=1)


y= df['label'].map({'correct': 1, 'incorrect': 0 })
df_test = df_test.drop(['pickup_time','drop_time'], axis=1)

#Split the data into training and test sets (0.33 for test with random state =0 )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#Making model
rfc = RandomForestClassifier(n_estimators=1000, max_depth=100, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(df_test)

#print(classification_report(y_test, rfc_predict))
print('Accuracy  :',rfc.score(X_test, y_test))

df_test['prediction'] = rfc_predict

df_test = df_test.drop(['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare'] ,axis=1)
print(df_test)
#save output into csv
df_test.to_csv('E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\csewiks_ml_2_Prediction.csv', index=False)
        

    