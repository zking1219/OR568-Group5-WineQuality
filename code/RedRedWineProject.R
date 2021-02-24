#Abstract
# Took Zack's preproc information and ran two randomForest Models
#Accuracy     Kappa 
#0.8463950 0.6899165 






library(AppliedPredictiveModeling)
library(readr)
library(caret)
#install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
library(tidyverse)

setwd("C:/Users/plockett/Documents/Personal/GMU Folder/OR 568/Project")
wine_tibble <-  read.csv('winequality-red.csv')

df <- as.data.frame(wine_tibble)
# Response is quality


# investigate the imbalance in the dataset;
table(df$quality)

# quality
#   3   4   5   6   7   8 
#   10  53 681 638 199  18 

# investigate missing values
which(is.na(df))
# integer(0)

# investigate variables with low variance
nearZeroVar(df)
# integer(0)

# look at predictor distributions
install.packages("survival")
install.packages("Hmisc")
library(Hmisc)
hist.data.frame(df[,c(1:3)])
hist.data.frame(df[,c(4:6)])
hist.data.frame(df[,c(7:9)])
hist.data.frame(df[,c(10:12)])
# Most of the predictors look to be significantly skewed, we'll have to
# remember to apply BoxCox transforms before modeling

# Lets look at bewteen predictor correlations and correlations with the response
# Look at all the correlation between all predictors:
cormat <- cor( df ) 
cormat

library(reshape2)
cormat[lower.tri(cormat)] <- NA
melt_cor <- melt(cormat, na.rm=TRUE)
ggplot(data = melt_cor, aes(Var2, Var1, fill = value)) + 
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high= "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 12, hjust = 1)) + 
  coord_fixed()

# There aren't many correlations either between predictors or between predictors and response.
# Only fixed acidity correlates somewhat with other predictors (citric acid, density, pH).
# PCA doesn't seem to be necessary for this analysis. 

# Preprocess the data according to our finding that it needs BoxCox
library(caret)
wine_preproc <- preProcess(df[,c(-12)], method = c("BoxCox", "center", "scale"))
predictors_preproc <- predict(wine_preproc, df[,c(-12)])

# join the predictors back to the response
array_preproc <- cbind(quality = df$quality, predictors_preproc)
df_preproc <- as.data.frame(array_preproc)

write.csv(df_preproc, file = "winequality_center_scale_boxcox.csv",
          row.names = FALSE)


wine_tibble_preproc <- read_csv("winequality_center_scale_boxcox.csv")
df_pp <- as.data.frame(wine_tibble_preproc)

df_pp$goodbad[df_pp$quality < 5.5] <- "bad"
df_pp$goodbad[df_pp$quality > 5.5] <- "good"


# remove goodbad
df_pp$quality <- df_pp$goodbad
df_pp <- df_pp[,1:12]
df_pp$quality <- as.factor(df_pp$quality)


#Random Forest Classification Good/Bad
library(randomForest)
library(rpart)

set.seed(100)
trainingRows <- createDataPartition(y = df_pp$quality, p = .8, list=FALSE)

chart.Correlation(df_pp[-1], col = df_pp$quality)


#Better Results with 15 branches and higher tunelength
RFModel <- train(quality~.,df_pp[trainingRows,],
            method = 'rf', TuneLength = 7,
            trcontrol = trainControl(
              method = 'cv', number = 15,
              classProbs = TRUE))

RFModel$results


pred <- predict(RFModel,df_pp[-trainingRows,])

confusionMatrix(pred, df_pp[-trainingRows,]$quality,"good")

rfModel1 = randomForest( df_pp[trainingRows,]$quality ~ ., data = df_pp[trainingRows,], ntree=400 ) # ntree=500
rfModel1

# predict quality:
rf_yHat = predict(rfModel1,df_pp[-trainingRows,])
#rf_yHat
## performance evaluation
rfPR = postResample(pred=rf_yHat, obs=df_pp[-trainingRows,]$quality)
rfPR

#Accuracy     Kappa 
#0.8463950 0.6899165 

plot(rfModel1)


