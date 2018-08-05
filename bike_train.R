setwd("C:/Users/Rahul/Desktop/edwisor/bike")
source("bike_preprocess.R")
library(rpart)
library(FNN)
library("randomForest")


set.seed(123)
train_data_index = sample(1:nrow(finalDF), 0.8 * nrow(finalDF))
train = finalDF[train_data_index,]
test = finalDF[-train_data_index,]

dpendentcolumnsreg=c("casual","cnt")
dpendentcolumnscas=c("registered","cnt")


train_reg=removeCols(dpendentcolumnsreg,train)
train_cas=removeCols(dpendentcolumnscas,train)

test_reg=removeCols(dpendentcolumnsreg,test)
test_cas=removeCols(dpendentcolumnscas,test)

calculatersqured =function(actual,predict){
 rsquared= (1 - (sum((actual-predict )^2)/sum((actual-mean(actual))^2)))*100
 return(rsquared)
}

MAPE = function(actual, predict){
  mape=(mean(abs((actual - predict)/actual)))*100
  return(mape)
}
#Decision tree regression
set.seed(123)
dtfit1 = rpart(registered ~ ., data = train_reg, method = "anova")
dtfit2 = rpart(casual ~ ., data = train_cas, method = "anova")
predictions_DT_reg = as.integer(predict(dtfit1, subset(test_reg, select = -registered)))
predictions_DT_cas=as.integer(predict(dtfit2, subset(test_cas, select = -casual)))
rsquareddtreg_dt=calculatersqured(test_reg$registered,predictions_DT_reg)
rsquareddtcas_dt=calculatersqured(test_cas$casual,predictions_DT_cas)
mapedtreg_dt=MAPE(test_reg$registered,predictions_DT_reg)
mapedtcas_dt=MAPE(test_cas$casual,predictions_DT_cas)

#randomforestregression
set.seed(123)
ntrees=c(seq(10,500,10))
rsquared_rf_reg= vector(mode="numeric", length=0)
rsquared_rf_cas= vector(mode="numeric", length=0)
rfbestestimator=function(){
  for(n in ntrees){
    set.seed(123)
    RF_model1 = randomForest(registered ~ ., data = train_reg, importance = TRUE, ntree = n)
    RF_model2 = randomForest(casual ~ ., data = train_cas, importance = TRUE, ntree = n)
    predictions_rf_reg = as.integer(predict(RF_model1, subset(test_reg, select = -registered)))
    predictions_rf_cas=as.integer(predict(RF_model2, subset(test_cas, select = -casual)))
    rsquared_rf_reg=append(rsquared_rf_reg,calculatersqured(test_reg$registered,predictions_rf_reg))
    rsquared_rf_cas=append(rsquared_rf_cas,calculatersqured(test_cas$casual,predictions_rf_cas))
    
  }
  plot(ntrees,rsquared_rf_reg,type = "l",col='red',xlab = "no. of trees",ylab = "Rsquared",main = "RF Rsquared Plot for registered",xaxp  = c(10, 500, 49))
  axis(1, at = c(seq(10,500,10)), tck = 1, lty = 2, col = "grey", labels = NA)
  
  plot(ntrees,rsquared_rf_cas,type = "l",col='red',xlab = "no. of trees",ylab = "Rsquared",main = "RF Rsquared Plot for casual ",xaxp  = c(10, 500, 49))
  axis(1, at = c(seq(10,500,10)), tck = 1, lty = 2, col = "grey", labels = NA)
  
}
rfbestestimator()
#10 is the best number for casual users
#60is the best number for registered users
set.seed(123)
RF_model_final1 = randomForest(registered ~ ., data = train_reg, importance = TRUE, ntree =60 )
RF_model_final2 = randomForest(casual ~ ., data = train_cas, importance = TRUE, ntree =10 )

RF_Predictions_reg = as.integer(predict(RF_model_final1, subset(test_reg, select = -registered)))
RF_Predictions_cas=as.integer(predict(RF_model_final2, subset(test_cas, select = -casual)))

rsquareddtreg_RF=calculatersqured(test_reg$registered,RF_Predictions_reg)
rsquareddtcas_RF=calculatersqured(test_cas$casual,RF_Predictions_cas)
mapedtreg_RF=MAPE(test_reg$registered,RF_Predictions_reg)
mapedtcas_RF=MAPE(test_cas$casual,RF_Predictions_cas)

#Multiple linear regression
set.seed(123)
train_reg=changecat(seasoncol,train_reg)
train_reg=changecat(weathersitcol,train_reg)
train_cas=changecat(seasoncol,train_cas)
train_cas=changecat(weathersitcol,train_cas)

lm_model1 = lm(registered ~ ., data = train_reg)
lm_model2 = lm(casual ~ ., data = train_cas)
predictions_lm_reg = as.integer(predict(lm_model1, subset(test_reg, select = -registered)))
predictions_lm_cas=as.integer(predict(lm_model2, subset(test_cas, select = -casual)))
rsquareddtreg_lm=calculatersqured(test_reg$registered,predictions_lm_reg)
rsquareddtcas_lm=calculatersqured(test_cas$casual,predictions_lm_cas)
mapedtreg_lm=MAPE(test_reg$registered,predictions_lm_reg)
mapedtcas_lm=MAPE(test_cas$casual,predictions_lm_cas)

#polynomial regression

set.seed(123)
poly_model1=lm(registered ~ polym(temp,hum,windspeed, degree=2,raw=TRUE)+weathersit1+
weathersit2+weathersit3+season1+season2+season3+season4+yr+workingday,data = train_reg)

poly_model2=lm(casual ~ polym(temp,hum,windspeed, degree=2,raw=TRUE)+weathersit1+
weathersit2+weathersit3+season1+season2+season3+season4+yr+workingday,data = train_cas)

poly_Predictions_reg = as.integer(predict(poly_model1, subset(test_reg, select = -registered)))
poly_Predictions_cas=as.integer(predict(poly_model2, subset(test_cas, select = -casual)))

rsquareddtreg_poly=calculatersqured(test_reg$registered,poly_Predictions_reg)
rsquareddtcas_poly=calculatersqured(test_cas$casual,poly_Predictions_cas)
mapedtreg_poly=MAPE(test_reg$registered,poly_Predictions_reg)
mapedtcas_poly=MAPE(test_cas$casual,poly_Predictions_cas)

tot_rsquare_reg=c(rsquareddtreg_dt,rsquareddtreg_lm,rsquareddtreg_RF,rsquareddtreg_poly)
tot_rsquare_cas=c(rsquareddtcas_dt,rsquareddtcas_lm,rsquareddtcas_RF,rsquareddtcas_poly)
tot_mape_reg=c(mapedtreg_dt,mapedtreg_lm,mapedtreg_RF,mapedtreg_poly)
tot_mape_cas=c(mapedtcas_dt,mapedtcas_lm,mapedtcas_RF,mapedtcas_poly)

namesalgo=c("DT","LM","RF","Poly")
barplot(tot_rsquare_reg,
        main = "Rsquared for registered",
        xlab = "Algorithm",
        ylab = "Rsquared",
        names.arg = namesalgo,
        col = "darkred")
barplot(tot_rsquare_cas,
        main = "Rsquared for casual",
        xlab = "Algorithm",
        ylab = "Rsquared",
        names.arg = namesalgo,
        col = "darkred")
barplot(tot_mape_reg,
        main = "MAPE for registered",
        xlab = "Algorithm",
        ylab = "MAPE",
        names.arg = namesalgo,
        col = "blue")
barplot(tot_mape_cas,
        main = "MAPE for casual",
        xlab = "Algorithm",
        ylab = "MAPE",
        names.arg = namesalgo,
        col = "blue")


save(RF_model_final2, file = "RF_cas.rda")
save(RF_model_final1, file = "RF.rda")