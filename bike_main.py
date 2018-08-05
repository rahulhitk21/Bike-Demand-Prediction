# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 21:39:19 2018

@author: Rahul
"""

import joblib
import os, argparse
import pandas as pd
import numpy as np
os.chdir("C:/Users/Rahul/Desktop/edwisor/bike")
import bike_preprocess as prep



class bike(object):
	def getLoadOption(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--Data_File', action='store', dest='Data_File')
		
		self.result_op = parser.parse_args()
		
		return self.result_op

def main():
    print('Execution started')
    cli = bike()
    cli_line= cli.getLoadOption()
    data_file = cli_line.Data_File
    test_data=pd.read_csv(data_file)
    predict_data=test_data.copy()
    test_data=prep.removeColumns(prep.coldrop,test_data)
    test_data=prep.chngtocat(test_data,prep.categorical_columns)
    test_data=prep.removeColumns(prep.unimportantColumns,test_data)
    test_data=prep.removeColumns(prep.dpendentmulticor,test_data)
    transfrmcol=prep.transfrmcol
    for i in transfrmcol:
        temp = prep.enc.fit_transform(test_data[[i]])
        temp=pd.DataFrame(temp,columns=[(i+"_"+str(k)) for k in test_data[i].value_counts().index],dtype='int64')
        temp=prep.chngtocat(temp,list(temp.columns))
        test_data=test_data.join(temp)
        test_data=test_data.drop(i,axis=1) 
    train_col=prep.finalDF.columns.tolist()
    for i in prep.dpendentcolumnsvif:
        train_col.remove(i)
    test_col=test_data.columns.tolist()
    for j in train_col:
        if j not in test_col:
            dttype=prep.finalDF[j].dtype
            test_data[j]=0
            test_data[j]=test_data[j].astype(dttype)
    test_data=test_data[train_col]
    predict_x=np.array(test_data.values)
    model_rf=joblib.load('RF.pkl')
    model_rf_cas=joblib.load('RF_cas.pkl')
    predict_y_reg=model_rf.predict(predict_x).astype(int)
    predict_y_cas=model_rf_cas.predict(predict_x).astype(int)
    predict_data['casual']=predict_y_cas
    predict_data['registered']=predict_y_reg
    predict_data['cnt']=predict_y_cas+predict_y_reg
    predict_data.to_csv("Predic_submission.csv",index=False)
    print("prediction has been made successfully")
if __name__ == "__main__":
	main()
    