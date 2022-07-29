#model script
#imports 
from IPython.display import Image
import datetime
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, model_selection, ensemble

import plotly.figure_factory as ff
import plotly.offline as py
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import mlflow
import xgboost as xgb
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import warnings
warnings.filterwarnings('ignore')
py.init_notebook_mode(connected=True)

import pickle

#---------- read data ---------
data = pd.read_csv('hour_v2.csv')

#--------- prepocess data ---------
data['dteday'] = pd.to_datetime(data['dteday'])
data['day'] = data['dteday'].map(lambda x:x.day)
data['month'] = data['dteday'].map(lambda x:x.month)
data['year'] = data['dteday'].map(lambda x:x.year)
data['hour'] = data['dteday'].map(lambda x:x.hour)

data.drop(labels=['dteday'], axis=1, inplace=True)

data.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

data.month.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'},inplace = True)
data.drop(['casual','registered', 'year', 'hour', 'day', 'workingday', 'holiday','instant','season','month'],axis=1,inplace=True)

#----------- Scaling --------------
min_max=MinMaxScaler()
scaled=pd.DataFrame(min_max.fit_transform(data), columns=data.columns)
scaled.head()

#-------- preparedatasets ---------
X = scaled.drop(labels=['cnt'], axis=1)
y = scaled['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#--------- Model fit and train ---------
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)

#--------- Save model ---------

with open('service_demo/model.bin', 'wb' ) as f_out:
    pickle.dump(rfr, f_out)
