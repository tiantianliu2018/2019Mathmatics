import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("train/train_merge.csv",header = None)
data.columns = ['Cell Index','Cell X','Cell Y','Height','Azimuth','Electrical Downtilt','Mechanical Downtilt',
                'Frequency Band','RS Power','Cell Altitude','Cell Building Height','Cell Clutter Index',
                'X','Y','Altitude','Building Height','Clutter Index','RSRP']
 
 ## Feature Engineering              
data['relative_x'] = (data['X']-data['Cell X'])
data['relative_y'] = (data['Y']-data['Cell Y'])
data['distance'] = (data['relative_x']**2 + data['relative_y']**2)**0.5
data.drop(['relative_x','relative_y'],axis = 1,inplace=True)
## 
data['angle'] = data['Electrical Downtilt'] + data['Mechanical Downtilt']
## 
data['relative_altitude'] = data['Altitude'] - data['Cell Altitude']
data['deltaHv'] = data['Height'] - np.tan(data['angle']/180.0 * np.pi) * data['distance'] + data['Cell Altitude'] - data['Altitude']
data['h_theta'] = 90 - np.arctan2((data['Y']-data['Cell Y']), (data['X']-data['Cell X']))/np.pi * 180.0 - data['Azimuth']
data['h_distance'] = 2*abs(np.sin(data['h_theta']/np.pi*180.0/2)) * data['distance']
## 
data.loc[data['Height'] == 0,['Height']] = 22.0
## 
data.loc[data['distance'] == 0,['distance']] = 1
data.loc[data['h_distance'] == 0,['h_distance']] = 1
## 
data['Height'] = np.log10(data['Height'])
data['Frequency Band'] = np.log10(data['Frequency Band'])
data['distance'] = np.log10(data['distance'])
data['h_distance'] = np.log10(data['h_distance'])

data['angle_level']=np.arctan2(data['X']-data['Cell X'],data['Y']-data['Cell Y'])*180/np.pi
data['angle_level']=np.abs(data['Azimuth']-data['angle_level'])
data['angle_level']= np.where(data['angle_level']<=180, data['angle_level'], 360.0-data['angle_level'])

data['hd'] = data['distance'] * data['Height']
data['f_sub_h'] = data['Frequency Band']-data['Height']
data['f_add_d'] = data['Frequency Band']+data['distance']
data['f_sub_hd'] = data['Frequency Band']-data['hd']

data['f_sub_h_add_d'] = data['f_sub_h']+data['distance']
data['f_sub_h_sub_hd'] = data['f_sub_h']-data['hd']
data['f_add_d_sub_hd'] = data['f_add_d']-data['hd']

data['f_sub_h_add_d_sub_hd'] = data['f_sub_h_add_d'] - data['hd']
data2 = data.drop(['Cell Index','Cell X', 'Cell Y','Azimuth','Electrical Downtilt',
                   'Mechanical Downtilt','Cell Altitude','Cell Building Height',
                   'angle','X','Y','Altitude'],axis=1)

#############################
train = lasso_data = data2.drop(['RSRP'], axis = 1)
label = data2['RSRP']


x_train, x_test, y_train, y_test = train_test_split(train, label, test_size = 0.1, random_state = 1)
model = RandomForestRegressor()   # LinearRegression()  # DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)  
score = model.score(x_test,y_test)
print(score)
       
mse = mean_squared_error(y_test,y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print('MSE: ',mse)
print('RMSE:',rmse)
print('MAE:',mae)
print('R2:',r2)