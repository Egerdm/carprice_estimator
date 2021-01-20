import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("car_data_km_1000.csv", sep=';')




one_cekis = df["CEKIS"].values.reshape(-1,1)

onehot_encoder_cekis = OneHotEncoder(sparse=False)

OHE_Cekıs = onehot_encoder_cekis.fit_transform(one_cekis)

df["Arkadan"] = OHE_Cekıs[:,0]
df["Onden"] = OHE_Cekıs[:,1]
df["4CEKER"] = OHE_Cekıs[:,2]
print(df)

one_Gear = df["Gear"].values.reshape(-1,1)

onehot_encoder_gear = OneHotEncoder(sparse=False)

OHE_Gear = onehot_encoder_gear.fit_transform(one_Gear)

df["Otomatik"] = OHE_Gear[:,0]
df["Yarı"] = OHE_Gear[:,1]
df["Duz"] = OHE_Gear[:,2]
print(OHE_Cekıs)

one_Fuel = df["Fuel"].values.reshape(-1,1)

onehot_encoder_fuel = OneHotEncoder(sparse=False)

OHE_Fuel = onehot_encoder_fuel.fit_transform(one_Fuel)

df["Benzin"] = OHE_Fuel[:,0]
df["Dizel"] = OHE_Fuel[:,1]
df["LPG"] = OHE_Fuel[:,2]
print(df)

import pickle  # Initialize the flask App

output_1 = open('OHE_cekıs.pkl', 'wb')
output_2 = open('OHE_gear.pkl', 'wb')
output_3 = open('Ohe_fuel.pkl', 'wb')
pickle.dump(onehot_encoder_cekis, output_1)
pickle.dump(onehot_encoder_gear, output_2)
pickle.dump(onehot_encoder_fuel, output_3)

print(onehot_encoder_gear)


from sklearn import preprocessing
import pandas as pd

le_serie = preprocessing.LabelEncoder()
le_brand = preprocessing.LabelEncoder()
le_color = preprocessing.LabelEncoder()
df[['Brand']] = df[['Brand']].apply(le_brand.fit_transform)
df[['Serie']] = df[['Serie']].apply(le_serie.fit_transform)
df[['Color']] = df[['Color']].apply(le_color.fit_transform)
# df['city'] = le.fit(df['city'])


import pickle  # Initialize the flask App

output_1 = open('Brand_Encoder.pkl', 'wb')
output_2 = open('Serie_Encoder.pkl', 'wb')
output_3 = open('Color_Encoder.pkl', 'wb')
pickle.dump(le_brand, output_1)
pickle.dump(le_serie, output_2)
pickle.dump(le_color, output_3)

print(le_color.classes_)
pd.set_option('display.max_columns', 500)
print(df)
X = df.loc[:, ['Brand', 'Serie', 'Color', 'Year', 'KM', 'CC', 'HP',
               'Galeriden', 'GARANTI',
               'Onden', 'Otomatik', 'Yarı','Dizel',
               'LPG']]

y = df.loc[:, ['Price']]


from sklearn.model_selection import train_test_split

y_test: object
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train.values.ravel())
pickle.dump(rf_reg, open('model.pkl','wb'))
print(X_train)
'''
y_pred = rf_reg.predict(X_test)
print("Accuracy on Traing set: ",rf_reg.score(X_train,y_train))
print("Accuracy on Testing set: ",rf_reg.score(X_test,y_test))
'''
