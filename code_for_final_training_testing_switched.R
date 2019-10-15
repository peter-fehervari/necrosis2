library(kableExtra)
library(knitr)
library(captioner)
library(dplyr)
library(ggplot2)
library(party)
library(RcmdrMisc)
library(caret)
library(VIM)
library(PresenceAbsence)
library(randomForestSRC)
library(LiblineaR)
library(mlbench)
library(DMwR)
library(gbm)


############### data import & manipulation ###################################
testingData=read.table("Necrosis_short_all_data_GULASH.csv",sep=",",h=T)
str(testingData)
testingData$Necrosis=factor(testingData$Necrosis)
levels(testingData$Necrosis)=c("No_necrosis","Necrosis")
testingData$Necrosis=relevel(testingData$Necrosis,ref="No_necrosis")
testingData2=testingData[,-1:-4]
str(testingData2)


trainingData=read.table("Nekrozis_24parameter_1.0.csv",sep=",",h=T)
str(trainingData)
trainingData$Necrosis=factor(trainingData$Necrosis)
levels(trainingData$Necrosis)=c("No_necrosis","Necrosis")
trainingData$Necrosis=relevel(trainingData$Necrosis,ref="No_necrosis")
trainingData2=trainingData[,-1]
ncol(trainingData2)

str(trainingData2)

#preselect variables that have more than 35% NAs

varNAratio=aggr(trainingData2,bars=T,combined=T,sortVars=T,sortCombs=T,cex.numbers=1,varheight=T,only.miss=F,cex.axis=0.5)
varNAratio2=varNAratio$missings
varNAratio2$percent=round(varNAratio2$Count/nrow(trainingData2)*100,1)
varNAration2=varNAratio2[order(varNAratio2$percent),]
trainingData3=trainingData2[,row.names(varNAratio2[varNAratio2$percent<35,])]
str(trainingData3)
#data imputation

testingData4=VIM::kNN(testingData2,k=5,imp_var = F)
trainingData4=VIM::kNN(trainingData3,k=5,imp_var = F)

ncol(testingData4)
dim(trainingData4)

################# SMOTE ################################
trainingData4SMOTE=SMOTE(Necrosis~.,data=trainingData4,perc.over = 500, perc.under = 300)

sum(table(trainingData4SMOTE$Necrosis))
table(trainingData4$Necrosis)
table(trainingData4SMOTE$Necrosis)

################### CART ##########################x

ctl=ctree_control(testtype = "Teststatistic",minbucket = 10, maxdepth = 2,mtry=10)
fa1=ctree(Necrosis~.,data=trainingData4SMOTE,control=ctl)
plot(fa1)

##################### CARET pred mods ##############################

# initialization

fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = T)

########## Bayes Log Reg #############
BayesLogRegFit <- train(Necrosis ~ ., data = trainingData4SMOTE, 
                        method = "bayesglm", 
                        trControl = fitControl,
                        na.action = na.pass,
                        metric="ROC")

BayesLogRegFit
confusionMatrix(predict(BayesLogRegFit,newdata = testingData4,type="raw"),reference = testingData4$Necrosis)

str(testingData4)
######## CART CARET ################x

cart2Grid <-  expand.grid(maxdepth = c(5:15), mincriterion=c(0.01))

cartFit1 <- train(Necrosis ~ ., data = trainingData4SMOTE, 
                  method = "ctree2", 
                  trControl = fitControl,
                  na.action = na.pass,
                  metric="ROC",
                  tuneGrid = cart2Grid)

cartFit1
ggplot(cartFit1)
getTrainPerf(cartFit1)

confusionMatrix(predict(cartFit1,newdata = testingData4),reference = testingData4$Necrosis)
ctreeImp=varImp(cartFit1,scale = F)
plot(ctreeImp)

makeTransparent<-function(someColor, alpha=100)
{
  newColor<-col2rgb(someColor)
  apply(newColor, 2, function(curcoldata){rgb(red=curcoldata[1], green=curcoldata[2],
                                              blue=curcoldata[3],alpha=alpha, maxColorValue=255)})
}

plot(cartFit1$finalModel,type="simple",gp = gpar(fontsize = 2),
     inner_panel=node_inner(cartFit1$finalModel,
     abbreviate = FALSE,            # short variable names
     pval = FALSE,                 # no p-values
     id = FALSE),                  # no id of node
     terminal_panel=node_terminal(cartFit1$finalModel, 
                                  abbreviate = T,
                                  digits = 2,                   # few digits on numbers
                                  fill = makeTransparent("red",30),            # make box white not grey
                                  id = FALSE)
)


######################## Boosted CART ######################


adaGrid <-  expand.grid(iter = 3:15, maxdepth=8:13,nu=c(0.001,0.0001))

adaFit <- train(Necrosis ~ ., data = trainingData4SMOTE, 
                  method = "ada", 
                  trControl = fitControl,
                  na.action = na.pass,
                  metric="ROC",
                  tuneGrid = adaGrid)


adaFit
ggplot(adaFit)

confusionMatrix(predict(adaFit$finalModel,newdata = testingData4),reference = testingData4$Necrosis)

adaFit

############## RandomForest CARET ###############
cforestGrid <-  expand.grid(mtry = c(7:10))
cforestFit <- train(Necrosis ~ ., data = trainingData4SMOTE, 
                    method = "cforest", 
                    trControl = fitControl,
                    na.action = na.pass,
                    metric="ROC",
                    tuneGrid = cforestGrid)
cforestFit
ggplot(cforestFit)
confusionMatrix(predict(cforestFit,newdata = testingData4,type="raw"),reference = testingData4$Necrosis)
cforestImp=varImp(cforestFit,scale = F)
plot(cforestImp)

################## GBM CARET ################

gbmGrid <-  expand.grid(interaction.depth = c(8:13), 
                        n.trees = (20:30)*50, 
                        shrinkage = 0.05,
                        n.minobsinnode = 2)


gbmFit <- train(Necrosis ~ ., data = trainingData4SMOTE, 
                method = "gbm", 
                trControl = fitControl,
                na.action = na.pass,
                metric="ROC",
                tuneGrid = gbmGrid)

gbmFit
plot(gbmFit,metric="ROC")

confusionMatrix(predict(gbmFit,newdata = testingData4,type="raw"),reference = testingData4$Necrosis)
gbmImp=varImp(gbmFit,scale = F)
plot(gbmImp)


############################ SVM CARET ############################

svmGrid <-  expand.grid(C=c(0.00001,0.0001,0.001,0.1,0.25,0.5,1),
                        scale=c(0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5),
                        degree=2:4)

fitControl2 <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = T)
#       summaryFunction = twoClassSummary,
#       classProbs = T)

svmFit <- train(Necrosis ~ ., data = trainingData4SMOTE,
                method = "svmPoly", 
                trControl = fitControl2,
                na.action = na.pass,
                tuneGrid=svmGrid,
                metric="ROC")
svmFit
plot(svmFit)
confusionMatrix(predict(svmFit,newdata = testingData4,type="raw"),reference = testingData4$Necrosis)

svmImp=varImp(svmFit,scale = F)
plot(svmImp)

################# Neural Network ####################x

nnGrid <-  expand.grid(neurons=3:20)


nnFit <- train(Necrosis ~ ., data = trainingData4SMOTE, 
               method = "nnet", 
               trControl = fitControl,
               na.action = na.pass,
               metric="ROC")

nnFit
plot(nnFit,metric="ROC")

confusionMatrix(predict(nnFit,newdata = testingData4,type="raw"),reference = testingData4$Necrosis)
nnImp=varImp(nnFit,scale = F)
plot(nnImp)

####### Model selection ##############
testingData4pred=testingData4
testingData4pred$NNFit=predict(nnFit,newdata = testingData4,type="prob")[,2]
testingData4pred$svmFit=predict(svmFit$finalModel,newdata = testingData4,type="raw")[,2]
testingData4pred$gbmFit=predict(gbmFit$finalModel,newdata = testingData4,type="response",n.trees = 1300)

foo=predict(cforestFit$finalModel,newdata = testingData4)
foo2=unlist(foo)[seq(from=0,to=length(unlist(foo)),by=2)]
testingData4pred$cforestFit=foo2


fof=predict(adaFit$finalModel,newdata = testingData4,type="prob")
fof2=unlist(fof)[seq(from=0,to=length(unlist(fof)),by=2)]
testingData4pred$adaFit=fof2

oof=predict(cartFit1$finalModel,newdata = testingData4,type="prob")
oof2=unlist(oof)[seq(from=0,to=length(unlist(oof)),by=2)]
testingData4pred$cartFit1=oof2

ooL=predict(fa1 ,newdata = testingData4,type="prob")
ooL2=unlist(ooL)[seq(from=0,to=length(unlist(ooL)),by=2)]
testingData4pred$fa1=ooL2


# Predictions for large dataset without NA imputation!
ooLL=predict(fa1 ,newdata = testingData2,type="prob")
ooLL2=unlist(ooLL)[seq(from=0,to=length(unlist(ooLL)),by=2)]
testingData4pred$fa1_noimp=ooLL2

testingData4pred$BayesLogRegFit=predict(BayesLogRegFit$finalModel,newdata = testingData4,type="response")


str(testingData4pred)
DATA=data.frame(cbind(1:nrow(testingData4pred),
           as.numeric(testingData4pred$Necrosis)-1,
           testingData4pred$NNFit,
           testingData4pred$svmFit,
           testingData4pred$gbmFit,
           testingData4pred$cforestFit,
           testingData4pred$adaFit,
           testingData4pred$cartFit1,
           testingData4pred$BayesLogRegFit,
           testingData4pred$fa1))

othresholds=optimal.thresholds(DATA,req.spec =0.75)
othresholds
head(DATA)

for(i in 1:7)
{
b=cmx(DATA,i,threshold = othresholds[10,i+1])
print(b)
}
auc.roc.plot(DATA)

plot(fa1)

