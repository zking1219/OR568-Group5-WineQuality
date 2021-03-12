#Chindepalli1 ------------------------------------------------------------------------------------------------------------------------
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
library (ggplot2)

#Load Data----------------------------------------------------------------------------------------------------------------
    mid <- read.csv("data/winequality_center_scale_boxcox.csv", header = TRUE)
    wineQuality <- as.data.frame(mid)
 
#Chindepalli2--Data Partition - Linear Regression----------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------
    trainingRows <- createDataPartition(y = wineQuality$quality, p = .8,list= FALSE)

    trainWine   <- wineQuality[trainingRows, ]
    testWine <- wineQuality[-trainingRows, ]

#Linear Regression--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Linear Model
    set.seed(100)
    lmModel <- train(quality ~.,data = trainWine, method = "lm", 
                 trControl = trainControl(method = "repeatedcv", repeats = 5))
    lmModel

    lmPredictLM <- predict(lmModel, testWine) 
    lmValuesLM = data.frame(obs = testWine$quality, pred = lmPredictLM)
    defaultSummary(lmValuesLM) 

    #Linear Model - Alcohol
    set.seed(100)
    lmModelAlc <-train(quality ~ alcohol, data = trainWine, method = "lm", 
                   trControl = trainControl(method = "repeatedcv", repeats = 5))
    lmModelAlc

    lmPredictAlc <- predict(lmModelAlc, testWine)
    lmValuesAlc = data.frame(obs = testWine$quality, pred = lmPredictAlc)
    defaultSummary(lmValuesAlc)

    #Linear Model - Sulfur
    set.seed(100)
    lmModelSulf <-train(quality ~ total.sulfur.dioxide + free.sulfur.dioxide  , data = trainWine, method = "lm", preProcess =c("BoxCox", "center", "scale"),
                    trControl = trainControl(method = "repeatedcv", repeats = 5))
    lmModelSulf

    lmPredictSulf <- predict(lmModelSulf, testWine)
    lmValuesSulf = data.frame(obs = testWine$quality, pred = lmPredictSulf)
    defaultSummary(lmValuesSulf)

    #PLS - All Predictors 
    set.seed(100)
    plsModel <- train(quality ~.,data = trainWine, method = "pls",
                  trControl = trainControl(method = "repeatedcv", repeats = 5))
    plsModel
    
    plsPredPLS <- predict(plsModel, testWine) 
    lmValuesPLS = data.frame(obs = testWine$quality, pred = plsPredPLS)
    defaultSummary(lmValuesPLS) 

    #PLS - remove highly correlated predicor - free sulfur dioxide 
    set.seed(100)
    plsModel2 <- train(quality ~.,data = trainWine[, -7], method = "pls",
                  trControl = trainControl(method = "repeatedcv", repeats = 5))
    plsModel2

    plsPredPLS2 <- predict(plsModel2, testWine[, -7]) 
    lmValuesPLS2 = data.frame(obs = testWine$quality, pred = plsPredPLS2)
    defaultSummary(lmValuesPLS2) 

    #PLS - remove highly correlated predicor - fixed acidity 
    set.seed(100)
    plsModel3 <- train(quality ~.,data = trainWine[, -2], method = "pls",
                   trControl = trainControl(method = "repeatedcv", repeats = 5))
    plsModel3

    plsPredPLS3 <- predict(plsModel3, testWine[, -2]) 
    lmValuesPLS3 = data.frame(obs = testWine$quality, pred = plsPredPLS3)
    defaultSummary(lmValuesPLS3) 
  

    #PLS Tuned
    set.seed(100)
    plsTune <- train(quality ~.,data = trainWine[, -7], method = "pls", tuneGrid = expand.grid(ncomp = 1:8),
                 trControl =trainControl(method = "repeatedcv", repeats = 5))
    plsTune

    plsTunePred <- predict(plsTune, testWine[, -7]) 
    lmValuesPlsTune = data.frame(obs = testWine$quality, pred = plsTunePred)
    defaultSummary(lmValuesPlsTune) 

    #PLS Tuned - LGCOV
    set.seed(100)
    plsTune2 <- train(quality ~.,data = trainWine[, -7], method = "pls", tuneGrid = expand.grid(ncomp = 1:8),
                 trControl = trainControl(method = "LGOCV"))
    plsTune2

    plsTunePred2 <- predict(plsTune2, testWine[, -7]) 
    lmValuesPlsTune2 = data.frame(obs = testWine$quality, pred = plsTunePred2)
    defaultSummary(lmValuesPlsTune2) 
    


#Linear Model Comparison------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    methods = c("Linear Model - All Predictors, 10 Fold CV 5 Repeats",
            "PLS Model - All Predictors, 10 Fold CV 5 Repeats",
            "PLS Model - Remove Highly Correlated Predictor: Free Sulfur Dioxide, 10 Fold CV 5 Repeats",
            "PLS Model - Remove Highly Correlated Predictor: Fixed Acidity, 10 Fold CV 5 Repeats",
            "PLS Tuned with Ncomp, 10 Fold CV 5 Repeats",
            "PLS Tuned with Ncomp, LGCOV")

    rmses = c(lmModel$results[ ,2],plsModel$results[3,2],plsModel2$results[3,2],plsModel3$results[3,2],plsTune$results[5,2],plsTune2$results[5,2])
    r2s = c(lmModel$results[ ,3],plsModel$results[3,3],plsModel2$results[3,3],plsModel3$results[3,3],plsTune$results[5,3],plsTune2$results[5,3])
    rmseSDs = c(lmModel$results[ ,5],plsModel$results[3,5],plsModel2$results[3,5],plsModel3$results[3,5],plsTune$result[5,5],plsTune2$results[5,5])
    r2SDs = c(lmModel$results[ ,6],plsModel$results[3,6],plsModel2$results[3,6],plsModel3$results[3,6],plsTune$results[5,6], plsTune2$results[5,6])
    
    compare = data.frame(RMSE = rmses, RSquared = r2s, RMSESD = rmseSDs, RSquaredSD = r2SDs)
    row.names(compare) <- methods
    compare
    
    
#Chindepalli3--Plot -----------------------------------------------------------------------------------------------------
    
    #Linear Models
    models= c("Linear Model","PLS Model - All Predictors","PLS - Free Sulfur Dioxide",
              "PLS -RemoveFixed Acidity","Tuned w/Ncomp - Repeated CV", "Tuned w/Ncomp - LGCOV")
    
    RMSE= c("0.6450","0.6468","0.6480","0.6473","0.6463","0.6511")
    R2= c("0.3689","0.3654","0.3631","0.3642","0.3664","0.3591")
    
    graphData = data.frame(RMSE = RMSE, R2 = R2)
    row.names(graphData) <- models
    
    
    #RMSE Plot
    ggplot(graphData, aes(x = models, y = RMSE, label = RMSE)) + geom_point(col="blue") +
      labs(x="Regression Models",
           y="RMSE",title="Linear Regression RMSE Breakdown") + geom_text(aes(label=RMSE),hjust=-.2, vjust=0, size = 10) +theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5), text = element_text(size=20)) 
  
    
    #RSquared Plot
    ggplot(graphData, aes(x = models, y = R2)) + geom_point(col="blue") +
      labs(x="Regression Models",
           y="R-Squared",title="Linear Regression R-Squared Breakdown") + geom_text(aes(label=R2),hjust=-.2, vjust=0, size = 10) +
       theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5), text = element_text(size=20)) 
    
    
    
#Chindepalli4--Plot --------------------------------------------------------
    #PLS Tuned Plot
    plsTuneR2 <-  round(plsTune$results$Rsquared,4)
    ggplot(plsTune) + geom_point(col="blue") + labs(y = "R-Squared (Repeated Cross-Validation)", title="PLS Tuned with ncomp and Repeated CV") +
      geom_text(aes(label=plsTuneR2),hjust=1, vjust=2, size = 7)  + theme(text = element_text(size=20))
    
    ---------------------  
      


#Chindepalli5-- NonLinear Model -----------------------------------------------------------------------------------------------------

    
    #KNN - Repeated CV
    set.seed(100)
    knnModelRCV = train(x=trainWine, y=trainWine$quality, method="knn",
                        preProc=c("center","scale"),
                        tuneLength=10,
                        trControl = trainControl(method = "repeatedcv", repeats = 5))
     knnModelRCV
     knnModelRCV$results
    
    knnPred = as.data.frame(predict(knnModelRCV, newdata=testWine))
    knnRCVPR = postResample(pred=knnPred, obs=testWine$quality)
    knnRCVPR
    
    
    #KNN - bootstrap
    set.seed(100)
    knnModel2 = train(x=trainWine, y=trainWine$quality, method="knn",
                      preProc=c("center","scale"),
                      tuneLength=10)
    knnModel2
    
    knnPred2 = predict(knnModel2, newdata=testWine)
    knnPR2 = postResample(pred=knnPred2, obs=testWine$quality)
    knnPR2
    
    
#Plot Non Linear Models
    
    #KNN - RepeatedCV
    knnRmse <-  round(knnModelRCV$results$RMSE,4)
    ggplot(knnModelRCV) + geom_point(col="blue") + geom_text(aes(label= knnRmse),hjust=-.5, vjust=0, size = 10) + 
      labs(title="KNN Model Tuned with Optimal K and Resampled with 10 Fold Repeated CV ") + theme(text = element_text(size=20))
    
    
    #KNN - Bootstrapped
    knn2Rmse <-  round(knnModel2$results$RMSE,4)
    ggplot(knnModel2) + geom_point(col="blue") +geom_text(aes(label= knn2Rmse),hjust=-.5, vjust=0, size=10) +
      labs(title="KNN Model Tuned with Optimal K and Bootstrapped Resampling") + theme(text = element_text(size=20))
    
    


    
 
 

 






