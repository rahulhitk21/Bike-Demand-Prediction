setwd("C:/Users/Rahul/Desktop/edwisor/bike")
library(dplyr)
library("corrgram")
library(caret)
library(car)
library(CatEncoders)

removeCols=function(listofcols,dataframe){
  dataframe=dataframe[ , !(names(dataframe) %in% listofcols)]
  return(dataframe)
}

dpendentcolumnsvif=c("casual","registered","cnt")
raw_data=read.csv("day.csv",header = TRUE)
is.null(raw_data)
set.seed(123)
train_index = sample(1:nrow(raw_data), 0.95 * nrow(raw_data))
learning = raw_data[train_index,]
submission = raw_data[-train_index,]
submission=removeCols(dpendentcolumnsvif,submission)
write.csv(submission, file = "predict_r_submission.csv",row.names=FALSE)


changecat=function(listofcols,dataframe){
  for (i in listofcols){
    dataframe[,i]=as.factor(dataframe[,i])}
  return(dataframe)
}

changenum=function(listofcols,dataframe){
  for (i in listofcols){
    dataframe[,i]=as.numeric(dataframe[,i])}
  return(dataframe)
}

chisqtst=function(X,dataframe,Y){
    k=0
    a=(chisq.test(table(dataframe[,Y],dataframe[,X])))
    if(a$p.value>0.05)
    {
      
      k=1
  
    }
    
    return(k)
  
}

coldrop=c('instant','dteday')
learning=removeCols(coldrop,learning)
categorical_columns=c("season","yr","mnth","holiday","weekday","weathersit","workingday")
learning=changecat(categorical_columns,learning)
boxplot(registered ~ workingday, data = learning, xlab = "workingday",
        ylab = "registered", main = "")
regworkdist=split(learning$registered,learning$workingday)
regworkdistmean=sapply(regworkdist,mean)
barplot(regworkdistmean,xlab = "workingday",ylab = "registered", main = "")
boxplot(casual ~ workingday, data = learning, xlab = "workingday",
        ylab = "casual", main = "")
casworkdist=split(learning$casual,learning$workingday)
casworkdistmean=sapply(casworkdist,mean)
barplot(casworkdistmean,xlab = "workingday",ylab = "casual", main = "")
boxplot(registered ~ weekday, data = learning, xlab = "weekday",
        ylab = "registered", main = "")
regweekdist=split(learning$registered,learning$weekday)
regweekdistmean=sapply(regweekdist,mean)
barplot(regweekdistmean,xlab = "weekday",ylab = "registered", main = "")
boxplot(casual ~ weekday, data = learning, xlab = "weekday",
        ylab = "casual", main = "")
casweekdist=split(learning$casual,learning$weekday)
casweekdistmean=sapply(casweekdist,mean)
barplot(casweekdistmean,xlab = "weekday",ylab = "casual", main = "")
boxplot(registered ~ holiday, data = learning, xlab = "holiday",
        ylab = "registered", main = "")
regholidist=split(learning$registered,learning$holiday)
regholidistmean=sapply(regholidist,mean)
barplot(regholidistmean,xlab = "holiday",ylab = "registered", main = "")
boxplot(casual ~ holiday, data = learning, xlab = "holiday",
        ylab = "casual", main = "")
casholidist=split(learning$casual,learning$holiday)
casholidistmean=sapply(casholidist,mean)
barplot(casholidistmean,xlab = "holiday",ylab = "casual", main = "")
boxplot(registered ~ yr, data = learning, xlab = "yr",
        ylab = "registered", main = "")
regyrdist=split(learning$registered,learning$yr)
regyrdistmean=sapply(regyrdist,mean)
barplot(regyrdistmean,xlab = "yr",ylab = "registered", main = "")
boxplot(casual ~ yr, data = learning, xlab = "yr",
        ylab = "casual", main = "")
casyrdist=split(learning$casual,learning$yr)
casyrdistmean=sapply(casyrdist,mean)
barplot(casyrdistmean,xlab = "yr",ylab = "casual", main = "")
boxplot(registered ~ season, data = learning, xlab = "season",
        ylab = "registered", main = "")
regseasdist=split(learning$registered,learning$season)
regseasdistmean=sapply(regseasdist,mean)
barplot(regseasdistmean,xlab = "season",ylab = "registered", main = "")
boxplot(casual ~ season, data = learning, xlab = "season",
        ylab = "casual", main = "")
casseasdist=split(learning$casual,learning$season)
casseasdistmean=sapply(casseasdist,mean)
barplot(casseasdistmean,xlab = "season",ylab = "casual", main = "")
boxplot(registered ~ mnth, data = learning, xlab = "mnth",
        ylab = "registered", main = "")
regmnthdist=split(learning$registered,learning$mnth)
regmnthdistmean=sapply(regmnthdist,mean)
barplot(regmnthdistmean,xlab = "mnth",ylab = "registered", main = "")
boxplot(casual ~ mnth, data = learning, xlab = "mnth",
        ylab = "casual", main = "")
casmnthdist=split(learning$casual,learning$mnth)
casmnthdistmean=sapply(casmnthdist,mean)
barplot(casmnthdistmean,xlab = "mnth",ylab = "casual", main = "")
boxplot(registered ~ weathersit, data = learning, xlab = "weathersit",
        ylab = "registered", main = "")
regwsitdist=split(learning$registered,learning$weathersit)
regwsitdistmean=sapply(regwsitdist,mean)
barplot(regwsitdistmean,xlab = "weathersit",ylab = "registered", main = "")
boxplot(casual ~ weathersit, data = learning, xlab = "weathersit",
        ylab = "casual", main = "")
caswsitdist=split(learning$casual,learning$weathersit)
caswsitdistmean=sapply(caswsitdist,mean)
barplot(caswsitdistmean,xlab = "weathersit",ylab = "casual", main = "")

dpendntcols=c("weekday","holiday")
x = vector(mode="character", length=0)
for (varbl in dpendntcols){
  if(chisqtst(varbl,learning,"workingday")==0){
    x=append(x,varbl)
    
  }
}

if(chisqtst("mnth",learning,"season")==0){
  x=append(x,"mnth")
  
}

transformedDF=removeCols(x,learning)


numeric_index = sapply(transformedDF,is.numeric)
numeric_data = transformedDF[,numeric_index]
cnames = colnames(numeric_data)
corrgram(transformedDF[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

plot(transformedDF$temp, transformedDF$atemp, main="", 
     xlab="atemp ", ylab="temp ", pch=19,col="blue")

model1 <- lm(cnt ~ ., data = numeric_data)
vif(model1)
aftrremoval=removeCols(c("atemp"),numeric_data)
model2 <- lm(cnt ~ ., data = aftrremoval)
vif(model2)

vifremovecol=c("atemp")
pfinalDF=removeCols(vifremovecol,transformedDF)

transfrmcol=c("weathersit","season")
seasoncol=c("season1","season2","season3","season4")
weathersitcol=c("weathersit1","weathersit2","weathersit3")


dummyencoding=function(X,dataframe,newname){
  temp_data = subset(dataframe, select = X)
  oenc = OneHotEncoder.fit(temp_data)
  z = transform(oenc,temp_data,sparse=FALSE)
  df=data.frame(z)
  colnames(df) = newname
  newdataframe=cbind(df,dataframe)
  return(newdataframe)
}
  
tfinalDF=dummyencoding("season",pfinalDF,seasoncol)
finalDF=dummyencoding("weathersit",tfinalDF,weathersitcol)


finalDF=removeCols(transfrmcol,finalDF)