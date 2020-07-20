# -*- coding: utf-8 -*-
"""
@author: wikasitha
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#read traing data
data = pd.read_csv('E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\train.csv')  

columns_feature = ['additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare','meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']
for each in columns_feature:
    data[each]= data[each].fillna(data[each].mean())


data = data[['tripid', 'additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare', 'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare',
       'label']]

label_encoder_no_of_class = LabelEncoder()

label_encoder_no_of_class  =label_encoder_no_of_class.fit(data['label'])

data["label"] = label_encoder_no_of_class.transform(data["label"])

#Making model
rfc = RandomForestClassifier(oob_score=True)


features = data.loc[:,'additional_fare':'fare']
labels = data.loc[:,'label']


rfc.fit(features,labels)
    
test_data = pd.read_csv('E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\test.csv')

test_data = test_data[['tripid', 'additional_fare', 'duration', 'meter_waiting','meter_waiting_fare',
                       'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']]
test = test_data.loc[:,'additional_fare':'fare']
 
prediction = rfc.predict(test)

prediction_original = list(label_encoder_no_of_class.inverse_transform(prediction))



prediction_result = pd.read_csv('E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\sample_submission.csv')
prediction_result['prediction'] = prediction_original
prediction_result = prediction_result.replace({'prediction': {'correct': 1,'incorrect': 0}})

#save output into csv
prediction_result.to_csv('E:\\msc\\2_sem\\ML\\kegalle_Ass\\wikis\\csewiks_ml_1_Prediction.csv', index=False)







