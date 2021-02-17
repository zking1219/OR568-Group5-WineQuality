
library(corrplot)
library(mlbench)
library(ggpubr)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) # misc library including skewness function
library(dplyr)
#----------


#Load Data
mid <- read.csv("/Users/saichindepalli/Desktop/winequality-red.csv", header = TRUE)
winequality <- as.data.frame(mid)

#Raw Data Analysis 

nearZeroVar(winequality)
colSums(is.na(winequality))

skewValues <- apply(winequality, 2, skewness)
skewValues
hist(skewValues)

#Transform 
winePP <- preProcess(winequality, method = c("BoxCox", "center", "scale"))
wineTrans <- predict(winePP, winequality)
winequalityY <- winequality$quality

#Post Process Summary
skewValuesPost <- apply(wineTrans, 2, skewness)
skewValuesPost
hist(skewValuesPost)

        #PCA

pcaWine <- prcomp(winequality, center = TRUE, scale. = TRUE)
pcaWine$rotation
pcaVAR <- pcaWine$sd^2/sum(pcaWine$sd^2)*100
pcaVAR[1:5]

    #Check Correlation
wineCor <- cor(wineTrans, method = "pearson")
    corrplot(wineCor, type = "lower", order = "hclust", 
         tl.ccool = "black", tl.srt = 45)
    
    #Compare Box plots 
par(mfrow=c(2,2))
boxplot(winequality, horizontal = TRUE)
boxplot(wineTrans, horizontal = TRUE)
