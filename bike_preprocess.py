# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:48:48 2018

@author: Rahul
"""

import pandas as pd
import os
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import OneHotEncoder
   


os.chdir("C:/Users/Rahul/Desktop/edwisor/bike")
raw_data=pd.read_csv("day.csv")
dpendentcolumnsvif=["casual","registered","cnt"]
missing_val=pd.DataFrame(raw_data.isnull().sum())
def chngtocat(dataframe,coltochange):
    for i in coltochange:
        dataframe[i]=dataframe[i].astype('category')
    return dataframe
def removeColumns(listofcolumns,dataframe):
    for i in listofcolumns:
        dataframe=dataframe.drop([i],axis=1) 
    return dataframe

def vif_calculator(X, thresh,dependent_col):
    vifdependent=[]
    X=removeColumns(dependent_col,X)
    X = add_constant(X)
    dropped=True
    while dropped:
        variables = X.columns 
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns ]
        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            vifdependent.append(X.columns.tolist()[maxloc])
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped=True
    if "const" in vifdependent:
        vifdependent.remove("const")
    return vifdependent
    
class ChiSquaretest:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _store_chisquare_result(self, colX, alpha):
        k=0
        if self.p>alpha:
            k=1
        return k
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        return self._store_chisquare_result(colX,alpha)

learning,validation=np.split(raw_data.sample(frac=1), [int(.95* len(raw_data))])
validation=removeColumns(dpendentcolumnsvif,validation)
validation.to_csv("Submission_data.csv",index=False)
coldrop=['instant','dteday']
learning=removeColumns(coldrop,learning)
categorical_columns=["season","yr","mnth","holiday","weekday","weathersit","workingday"]
learning=chngtocat(learning,categorical_columns)
sns.boxplot(x="workingday", y="registered", data=learning)
sns.barplot(x="workingday", y="registered", data=learning)
sns.boxplot(x="workingday", y="casual", data=learning)
sns.barplot(x="workingday", y="casual", data=learning)
sns.boxplot(x="weekday", y="registered", data=learning)
sns.barplot(x="weekday", y="registered", data=learning)
sns.boxplot(x="weekday", y="casual", data=learning)
sns.barplot(x="weekday", y="casual", data=learning)
sns.boxplot(x="holiday", y="registered", data=learning)
sns.barplot(x="holiday", y="registered", data=learning)
sns.boxplot(x="holiday", y="casual", data=learning)
sns.barplot(x="holiday", y="casual", data=learning)
sns.boxplot(x="yr", y="registered", data=learning)
sns.barplot(x="yr", y="registered", data=learning)
sns.boxplot(x="yr", y="casual", data=learning)
sns.barplot(x="yr", y="casual", data=learning)
sns.boxplot(x="season", y="registered", data=learning)
sns.barplot(x="season", y="registered", data=learning)
sns.boxplot(x="season", y="casual", data=learning)
sns.barplot(x="season", y="casual", data=learning)
sns.boxplot(x="mnth", y="registered", data=learning)
sns.barplot(x="mnth", y="registered", data=learning)
sns.boxplot(x="mnth", y="casual", data=learning)
sns.barplot(x="mnth", y="casual", data=learning)
sns.boxplot(x="weathersit", y="registered", data=learning)
sns.barplot(x="weathersit", y="registered", data=learning)
sns.boxplot(x="weathersit", y="casual", data=learning)
sns.barplot(x="weathersit", y="casual", data=learning)
sns.countplot(x="season",hue="weathersit",data=learning)

cT = ChiSquaretest(learning)
unimportantColumns=[]
dpendntcols=["weekday","holiday"]
for var in dpendntcols:
    if (cT.TestIndependence(colX=var,colY="workingday" )==0): 
        unimportantColumns.append(var)
if (cT.TestIndependence(colX="mnth",colY="season" )==0): 
    unimportantColumns.append("mnth")

transformedDF=removeColumns(unimportantColumns,learning)
numerical_columns=["temp","atemp","hum","windspeed","casual","registered","cnt"]
df_corr = transformedDF.loc[:,numerical_columns]
f, ax = plt.subplots(figsize=(9, 6))
corr = df_corr.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)
sns.regplot("temp","atemp",data=transformedDF,fit_reg=False)

dpendentmulticor=vif_calculator(df_corr,10,dpendentcolumnsvif)

finalDF=removeColumns(dpendentmulticor,transformedDF)
transfrmcol=["weathersit","season"]
enc = OneHotEncoder(sparse=False)
for i in transfrmcol:
    temp = enc.fit_transform(finalDF[[i]])
    temp=pd.DataFrame(temp,columns=[(i+"_"+str(k)) for k in finalDF[i].value_counts().index],dtype='int64')
    temp=chngtocat(temp,list(temp.columns))
    finalDF=finalDF.join(temp)
    finalDF=finalDF.drop(i,axis=1)    






