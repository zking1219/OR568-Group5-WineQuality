library(corrplot)
library(mlbench)
library(ggpubr)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) # misc library including skewness function
library(dplyr)
library(ISLR)
library(pls)
library(elasticnet)
library(tidyr)
library(lattice)
library(randomForest)
library(MASS)
library(pROC)
library(glmnet)

#Pre Process----------------------------------------------------------------------------------------------------------------


    #Load Data
mid <- read.csv("/Users/saichindepalli/Desktop/Group_Project/winequality-red.csv", header = TRUE)
wineQuality <- as.data.frame(mid)

    #Raw Data Analysis 
wineQualityPre <- wineQuality[,1:11 ]
winePP <- preProcess(wineQualityPre, method = c("BoxCox", "center", "scale"))
wineTrans <- predict(winePP, wineQualityPre)

    #Correlation Analysis

    #With Quality
wineCor <- cor(wineQuality, method = "pearson")
# View(cor(wineCor) %>%
#        as.data.frame() %>%
#      mutate(var1 = rownames(.)) %>%
#        gather(var2, value, -var1) %>%
#         arrange(desc(value)) %>%
#         group_by(value) %>%
#       filter(row_number()==1))
    
  #Without Quality
wineCorWO <- cor(wineTrans, method = "pearson")
# View(cor(wineCorWO) %>%
#         as.data.frame() %>%
#         mutate(var1 = rownames(.)) %>%
#         gather(var2, value, -var1) %>%
#         arrange(desc(value)) %>%
#         group_by(value) %>%
#         filter(row_number()==1))
 
 #Data Partition - Linear Regression------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(555)
trainingRows <- createDataPartition(y = wineQuality$quality, p = .8,list= FALSE)

trainWine   <- wineQuality[trainingRows, ]
testWine <- wineQuality[-trainingRows, ]

#Linear Regression--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

     #Linear Model
set.seed(198)
lmModel <- train(quality ~.,data = trainWine, method = "lm", preProcess =c("BoxCox", "center", "scale"), 
                 trControl = trainControl(method = "repeatedcv", repeats = 5))
lmModel

#Predict
lmPredictLM <- predict(lmModel, testWine) 
lmValuesLM = data.frame(obs = testWine$quality, pred = lmPredictLM)
defaultSummary(lmValuesLM) 

      #Linear Model- Alcohol
set.seed(198)
lmModelAlc <-train(quality ~ alcohol, data = trainWine, method = "lm", preProcess =c("BoxCox", "center", "scale"), 
                   trControl = trainControl(method = "repeatedcv", repeats = 5))
lmModelAlc

#Predict
lmPredictAlc <- predict(lmModelAlc, testWine)
lmValuesAlc = data.frame(obs = testWine$quality, pred = lmPredictAlc)
defaultSummary(lmValuesAlc)

      #Linear Model - Sulfur
set.seed(209)
lmModelSulf <-train(quality ~ total.sulfur.dioxide + free.sulfur.dioxide  , data = trainWine, method = "lm", preProcess =c("BoxCox", "center", "scale"),
                    trControl = trainControl(method = "repeatedcv", repeats = 5))
lmModelSulf

#Predict
lmPredictSulf <- predict(lmModelSulf, testWine)
lmValuesSulf = data.frame(obs = testWine$quality, pred = lmPredictSulf)
defaultSummary(lmValuesSulf)

      #PLS
set.seed(100)
plsModel <- train(quality ~.,data = trainWine, method = "pls", preProcess =c("BoxCox", "center", "scale"),
                  trControl = trainControl(method = "repeatedcv", repeats = 5))
plsModel

plsPredPLS <- predict(plsModel, testWine) 
lmValuesPLS = data.frame(obs = testWine$quality, pred = plsPredPLS)
defaultSummary(lmValuesPLS) 

      #PLS Tuned
set.seed(334)
plsTune <- train(quality ~.,data = trainWine, method = "pls", preProcess =c("BoxCox", "center", "scale"), tuneGrid = expand.grid(ncomp = 1:8),
                 trControl =trainControl(method = "repeatedcv", repeats = 5))
plsTune

plsTunePred <- predict(plsTune, testWine) 
lmValuesPlsTune = data.frame(obs = testWine$quality, pred = plsTunePred)
defaultSummary(lmValuesPlsTune) 

#Elastic Net Penalty
enetGrid <- expand.grid(lambda = c(0, 0.01, .1), fraction = seq(.05, 1, length = 20))
set.seed(222)
enetTune <- train(quality ~.,data = trainWine, method = "enet",preProcess =c("BoxCox", "center", "scale"), tuneGrid = enetGrid,
                  trControl = trainControl(method = "repeatedcv", repeats = 5))
enetTune

##Classification Data Partition ------------------------------------------------------------------------------------------------------------------

#Convert Quality to factor for Classification
#removing highly correlated predictors 
  # free.sulfur.dioxide and citric acid

wineQualityClass <- wineQuality [ ,c(1:2, 4,5,7:12)]
wineQualityClass$quality <- as.factor(wineQualityClass$quality)
#lapply(wineQualityClass, class)

set.seed(555)
trainingRows <- createDataPartition(y = wineQualityClass$quality, p = .80,list= FALSE)

trainWineClass <- wineQualityClass[trainingRows, ]
testWineClass <- wineQualityClass[-trainingRows, ]

#CLassification Models--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

      #LDA--
set.seed(222)
ldaClass <- train(quality ~.,data = trainWineClass, method = "lda", preProcess =c("BoxCox", "center", "scale"), metric = "Kappa", trControl = trainControl(method = "LGOCV"))
ldaClass

ldaClassPred<- predict(ldaClass, testWineClass)
ldaClassPred

ldaClassVal = data.frame(pred=ldaClassPred, obs=testWineClass$quality)
ldaClassPR <- postResample(pred=ldaClassPred, obs=testWineClass$quality)
ldaClassPR

  #PLS--
set.seed(100)
plsClass <- train(quality ~.,data = trainWineClass, method = "pls",metric = "Kappa", preProcess =c("BoxCox", "center", "scale"),
                  trControl = trainControl(method = "LGOCV"))
plsClass

plsClassPred <- predict(plsClass, testWineClass) 
lmValuesPLSClass = data.frame(obs = testWineClass$quality, pred = plsClassPred)
defaultSummary(lmValuesPLSClass) 

      #PLS Tuned
set.seed(292)
plsClassTuned <- train(quality ~.,data = trainWineClass, method = "pls",tuneGrid = expand.grid(ncomp = 1:5), preProcess =c("BoxCox", "center", "scale"),
                      metric = "Kappa", trControl = trainControl(method = "LGOCV"))
plsClassTuned

plsClassTunedCM <- confusionMatrix(plsClassTuned, norm = "none")
plsClassTunedCM

plsClassTunedPred <- predict(plsClassTuned, testWineClass)
plsClassTunedPred

plsClassTunedVal <-data.frame(pred=plsClassTunedPred, obs=testWineClass$quality)
plsClassTunedPR <- postResample(pred=plsClassTunedPred, obs=testWineClass$quality)
plsClassTunedPR

      #Penzalied - Glmnet Model
glmnGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1), lambda = seq(.01, .2, length = 40))
set.seed(476)
glmnClass <- train(quality ~.,data = trainWineClass,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 preProc = c("BoxCox", "center", "scale"),
                 metric = "Kappa",
                 trControl = trainControl(method = "LGOCV"))
glmnClass

    

