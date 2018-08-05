setwd("C:/Users/Rahul/Desktop/edwisor/bike")
source("bike_preprocess.R")
library("randomForest")

args = commandArgs(trailingOnly=TRUE)
print(args[1])
predict_raw_data=read.csv(args[1],header = TRUE)
predict_data=removeCols(coldrop,predict_raw_data)
predict_data = changecat(categorical_columns,predict_data)
predict_data =removeCols(x,predict_data)
predict_data=removeCols(vifremovecol,predict_data)

necscolumn=c('X1',"X2","X3","X4")

maindummyencoding=function(X,dataframe,newname){
  temp_data = subset(dataframe, select = X)
  oenc = OneHotEncoder.fit(temp_data)
  z = transform(oenc,temp_data,sparse=FALSE)
  df=data.frame(z)
  if(length(necscolumn)!=length(newname)){
    necscolumn=setdiff(necscolumn, "X4")
  }
  for(i in necscolumn){
    if((i %in% names(df))==FALSE){
      df[,i] = 0
    }
  }
  colnames(df) = newname
  newdataframe=cbind(df,dataframe)
  return(newdataframe)
}

predict_data=maindummyencoding("season",predict_data,seasoncol)
predict_data=maindummyencoding("weathersit",predict_data,weathersitcol)

finalDF_predict=removeCols(transfrmcol,predict_data)

RF_reg_model=get(load(file = "RF.rda"))
RF_cas_model=get(load(file = "RF_cas.rda"))
Predictions_reg = as.integer(predict(RF_reg_model,finalDF_predict))
Predictions_cas = as.integer(predict(RF_cas_model,finalDF_predict))

predict_raw_data$casual=Predictions_cas
predict_raw_data$registered=Predictions_reg
predict_raw_data$cnt=predict_raw_data$registered+predict_raw_data$casual


write.csv(predict_raw_data, file = "predict_r_submission_final.csv",row.names=FALSE)

print("prediction has been made successfully")

