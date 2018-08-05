# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 03:00:08 2018

@author: Rahul
"""
import joblib
import pandas as pd
import numpy as np
import os
from sklearn import cross_validation 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


os.chdir("C:/Users/Rahul/Desktop/edwisor/bike")
import bike_preprocess as prep

def mape(y_true,y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape
train_data=prep.finalDF
dpendntcolumns=["casual","registered","cnt"]
X=np.array(prep.removeColumns(dpendntcolumns,train_data))
dpendntcolumns.remove("cnt")
y=np.array(prep.finalDF.loc[:,dpendntcolumns])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2,random_state=25)
y_train_cas=[]
y_test_cas=[]
y_train_reg=[]
y_test_reg=[]

for i in y_train:
    y_train_cas.append(i[0])
    y_train_reg.append(i[1])
    
y_train_cas_arr=np.asarray(y_train_cas,dtype=np.int64)
y_train_reg_arr=np.asarray(y_train_reg,dtype=np.int64)


for i in y_test:
    y_test_cas.append(i[0])
    y_test_reg.append(i[1])
    
y_test_cas_arr=np.asarray(y_test_cas,dtype=np.int64)
y_test_reg_arr=np.asarray(y_test_reg,dtype=np.int64)

##############################multiple linear regression###################################

model_reg = LinearRegression().fit(X_train, y_train_reg_arr)
model_cas = LinearRegression().fit(X_train, y_train_cas_arr)
confidence_mlr_reg=(model_reg.score(X_test, y_test_reg))*100
confidence_mlr_cas=(model_cas.score(X_test, y_test_cas))*100
y_pred_reg_mlr=model_reg.predict(X_test).astype(int)
y_pred_cas_mlr=model_cas.predict(X_test).astype(int)
mape_mlr_reg= mape(y_test_reg,y_pred_reg_mlr)
mape_mlr_cas= mape(y_test_cas,y_pred_cas_mlr)
###############################DecisionTreeRegressor Model#################################################

regressor_reg=DecisionTreeRegressor(random_state = 25).fit(X_train, y_train_reg_arr)
regressor_cas=DecisionTreeRegressor(random_state = 25).fit(X_train, y_train_cas_arr)
confidence_dt_reg=(regressor_reg.score(X_test, y_test_reg))*100
confidence_dt_cas=(regressor_cas.score(X_test, y_test_cas))*100
y_pred_reg_dt=regressor_reg.predict(X_test).astype(int)
y_pred_cas_dt=regressor_cas.predict(X_test).astype(int)
mape_dt_reg= mape(y_test_reg,y_pred_reg_dt)
mape_dt_cas= mape(y_test_cas,y_pred_cas_dt)
##############################Random forest###############################################

def showbestestimator():
    estimators = np.arange(10, 501, 10)
    scores_reg = {}
    scores_cas = {}
    for n in estimators:
        model_rf_reg=RandomForestRegressor(n_estimators=n,random_state=25,n_jobs=-1)
        model_rf_cas=RandomForestRegressor(n_estimators=n,random_state=25,n_jobs=-1)
        mod_rf_reg=model_rf_reg.fit(X_train, y_train_reg_arr)
        mod_rf_cas=model_rf_cas.fit(X_train, y_train_cas_arr)
        scores_reg.update({n:mod_rf_reg.score(X_test, y_test_reg)})
        scores_cas.update({n:mod_rf_cas.score(X_test, y_test_cas)})
    x=[]
    y=[]
    z=[]
    for k,v in scores_reg.items():
        x.append(k)
        y.append(v)
    for k,v in scores_cas.items():
        z.append(v)
    plt.figure(figsize=(20,10))
    plt.plot(x, y)
    plt.grid(True)  
    plt.xticks(np.arange(10,501,10))
    plt.xlabel('Number_of_trees')
    plt.ylabel('R_squared_reg')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.plot(x,z)
    plt.grid(True)  
    plt.xticks(np.arange(10,501,10))
    plt.xlabel('Number_of_trees')
    plt.ylabel('R_squared_cas') 
    plt.show()
showbestestimator()

final_model_reg=RandomForestRegressor(n_estimators=310,random_state=25,n_jobs=-1)
final_model_reg.fit(X_train, y_train_reg_arr)
final_model_cas=RandomForestRegressor(n_estimators=130,random_state=25,n_jobs=-1)
final_model_cas.fit(X_train, y_train_cas_arr)
confidence_rf_reg=(final_model_reg.score(X_test, y_test_reg))*100
confidence_rf_cas=(final_model_cas.score(X_test, y_test_cas))*100
y_pred_reg_rf=final_model_reg.predict(X_test).astype(int)
y_pred_cas_rf=final_model_cas.predict(X_test).astype(int)
mape_rf_reg= mape(y_test_reg,y_pred_reg_rf)
mape_rf_cas= mape(y_test_cas,y_pred_cas_rf)

############################polynomial regression########################################
X_ = PolynomialFeatures(degree=2).fit_transform(X_train)
predict_ = PolynomialFeatures(degree=2).fit_transform(X_test)
clf_poly_reg=LinearRegression()
clf_poly_cas=LinearRegression()
clf_poly_reg.fit(X_, y_train_reg_arr)
clf_poly_cas.fit(X_, y_train_cas_arr)
y_pred_reg_pl=clf_poly_reg.predict(predict_)
y_pred_cas_pl=clf_poly_cas.predict(predict_)
rvalue_poly_reg=(r2_score(y_test_reg,y_pred_reg_pl))*100
rvalue_poly_cas=(r2_score(y_test_cas,y_pred_cas_pl))*100
mape_pl_reg= mape(y_test_reg,y_pred_reg_pl)
mape_pl_cas= mape(y_test_cas,y_pred_cas_pl)

summary_reg= pd.DataFrame({'Model_name': ["MLR","DTR","RFR","PLR"], 'R_squared_value': [confidence_mlr_reg,confidence_dt_reg,confidence_rf_reg,rvalue_poly_reg], 'mape': [mape_mlr_reg,mape_dt_reg,mape_rf_reg,mape_pl_reg]})

fig = plt.figure(figsize=(9,6)) 


ax = fig.add_subplot(111)
ax2 = ax.twinx() 

width = 0.4

summary_reg.R_squared_value.plot(kind='bar', color='red', ax=ax, width=width, position=1)
summary_reg.mape.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.set_ylabel('R_squared %')
ax.set_xticklabels(summary_reg.Model_name)
ax2.set_ylabel('MAPE')



plt.show()

summary_cas= pd.DataFrame({'Model_name': ["MLR","DTR","RFR","PLR"], 'R_squared_value': [confidence_mlr_cas,confidence_dt_cas,confidence_rf_cas,rvalue_poly_cas], 'mape': [mape_mlr_cas,mape_dt_cas,mape_rf_cas,mape_pl_cas]})

fig = plt.figure(figsize=(9,6)) 


ax = fig.add_subplot(111)
ax2 = ax.twinx() 

width = 0.4

summary_cas.R_squared_value.plot(kind='bar', color='red', ax=ax, width=width, position=1)
summary_cas.mape.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.set_ylabel('R_squared %')
ax.set_xticklabels(summary_cas.Model_name)
ax2.set_ylabel('MAPE')



plt.show()
joblib.dump(final_model_reg,'RF.pkl')
joblib.dump(final_model_cas,'RF_cas.pkl')

