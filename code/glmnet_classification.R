# Wine quality classificaton: good/bad with penalized logistic regression

# Read in the raw data
library(readr)
wine_tibble <- read_csv("/Users/iaa211/Documents/ContinuedEducation/GMU-Masters/OR568/Project/Data/winequality-red.csv")
df_raw <- as.data.frame(wine_tibble)

# Read in the pre-processed data
wine_tibble_preproc <- read_csv("/Users/iaa211/Documents/ContinuedEducation/GMU-Masters/OR568/Project/Data/winequality_center_scale_boxcox.csv")
df_pp <- as.data.frame(wine_tibble_preproc)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
library(caret)

# Penalized Model
#install.packages("glmnet")
library(glmnet)

# Now build classification models treating 3, 4, 5, as "bad"; 6, 7, 8 as "good"
########################

# Read in df_pp again (we don't want quality as a factor any longer)
df_pp <- as.data.frame(wine_tibble_preproc)

df_pp$goodbad[df_pp$quality < 5.5] <- "bad"
df_pp$goodbad[df_pp$quality > 5.5] <- "good"

# remove goodbad
df_pp$quality <- df_pp$goodbad
df_pp <- df_pp[,1:12]
df_pp$quality <- as.factor(df_pp$quality)


# Do the train/test split
set.seed(100)
trainingRows <- createDataPartition(y = df_pp$quality, p = .8, list=FALSE)

# Penalized Model

glmnGrid <- expand.grid(alpha = seq(0,1,length=15), lambda = seq(.0005, .05, length = 25))
set.seed(100)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
glmnFit <- train(quality ~ ., data = df_pp[trainingRows,],
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 trControl = trainControl(method = "cv", number=10),
                 metric="Kappa",
                 returnData=TRUE,
                 fitBest=FALSE)
glmnFit
# alpha = 0.1428571 and lambda = 0.0335
# Acc and Kappa
# 0.7390746  0.4767738

# Test set Acc, Kappa, (ROC)
preds <- predict(glmnFit, df_pp[-trainingRows,])
confusionMatrix(preds, df_pp[-trainingRows,]$quality, positive="good")
# Acc = .7712 and Kappa = 0.5393

# Remove correlated predictors and try again
X <- df_pp[,c(-1)]
tooHigh75 <- findCorrelation(cor(X), cutoff = .75)
colnames(X)[tooHigh75]
# Removed "free sulfur dioxide"

set.seed(100)
glmnFit75 <- train(quality ~ ., data = df_pp[trainingRows,c(-(tooHigh75 + 1))],
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 trControl = trainControl(method = "cv", number=10),
                 metric="Kappa",
                 returnData=TRUE,
                 fitBest=FALSE)
glmnFit75
# alpha     lambda     Accuracy   Kappa    AccuracySD    KappaSD
# 0.9285714 0.0066875 0.7390928 0.4767878 0.04492503 0.09017532

# Test set Acc, Kappa, (ROC)
preds75 <- predict(glmnFit75, df_pp[-trainingRows,c(-(tooHigh75 + 1))])
confusionMatrix(preds75, df_pp[-trainingRows,]$quality, positive="good")
# Acc = 0.768 and Kappa = 0.5328

X65 <-df_pp[,c(-1)]
tooHigh65 <- findCorrelation(cor(X65), cutoff = .65)
colnames(X65)[tooHigh65]
# Remove "fixed acidity" and "free sulfur dioxide"

set.seed(100)
glmnFit65 <- train(quality ~ ., data = df_pp[trainingRows,c(-(tooHigh65 + 1))],
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   trControl = trainControl(method = "cv", number=10),
                   metric="Kappa",
                   returnData=TRUE,
                   fitBest=FALSE)
glmnFit65
# glmnFit65$bestTune
# glmnFit65$results[169,]
# alpha       lambda     Accuracy   Kappa AccuracySD    KappaSD
# 0.4285714 0.037625 0.7375121 0.4731238 0.04866879 0.09729945

# Test set Acc, Kappa, (ROC)
preds65 <- predict(glmnFit65, df_pp[-trainingRows,c(-(tooHigh65 + 1))])
confusionMatrix(preds65, df_pp[-trainingRows,]$quality, positive="good")
# Acc = 0.768 and Kappa = 0.5324

X50 <-df_pp[,c(-1)]
tooHigh50 <- findCorrelation(cor(X50), cutoff = .5)
colnames(X50)[tooHigh50]
# Remove "fixed acidity"        "citric acid"          "total sulfur dioxide"

set.seed(100)
glmnFit50 <- train(quality ~ ., data = df_pp[trainingRows,c(-(tooHigh50 + 1))],
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   trControl = trainControl(method = "cv", number=10),
                   metric="Kappa",
                   returnData=TRUE,
                   fitBest=FALSE)
glmnFit50
# alpha       lambda     Accuracy   Kappa   AccuracySD    KappaSD
# 1           0.0396875 0.7359676 0.4703248 0.04393874 0.08866786
# LASSO

# Test set Acc, Kappa, (ROC)
preds50 <- predict(glmnFit50, df_pp[-trainingRows,c(-(tooHigh50 + 1))])
confusionMatrix(preds50, df_pp[-trainingRows,]$quality, positive="good")
# Acc = 0.7524 and Kappa = 0.5001

# Get SDs
glmnFit$results
glmnFit75$results
glmnFit65$results
glmnFit50$results

# log regression coefficients... pH is barely used, perhaps remove it
coef(glmnFit$finalModel, glmnFit$bestTune$lambda)
coef(glmnFit75$finalModel, glmnFit75$bestTune$lambda)
coef(glmnFit65$finalModel, glmnFit65$bestTune$lambda)
coef(glmnFit50$finalModel, glmnFit50$bestTune$lambda)

# varImp
varImp(glmnFit)
plot(varImp(glmnFit))
varImp(glmnFit75)
plot(varImp(glmnFit75))
varImp(glmnFit65)
plot(varImp(glmnFit65))
varImp(glmnFit50)
plot(varImp(glmnFit50))


# Produce ROC curves. Show/plot Kappa's and Accuracies.
# If there's a variable importance plot or coefficients from the penalized logistic regression,
# show those

glmnetCM <- confusionMatrix(glmnFit, norm = "none")
glmnetCM

glmnetCM75 <- confusionMatrix(glmnFit75, norm = "none")
glmnetCM75

glmnetCM65 <- confusionMatrix(glmnFit65, norm = "none")
glmnetCM65

glmnetCM50 <- confusionMatrix(glmnFit50, norm = "none")
glmnetCM50

library(MASS)
#install.packages("pROC")
library(pROC)

#preds <- predict(glmnFit, df_pp[,c(-1)])

roc_df <- df_pp
probs <- extractProb(list(glm=glmnFit), testX=df_pp[-trainingRows,c(-1)], testY=df_pp[-trainingRows,]$quality)
test_probs <- probs[probs$dataType == "Test",]

probs75 <- extractProb(list(glm=glmnFit75), testX=X[-trainingRows,c(-tooHigh75)],testY=df_pp[-trainingRows,]$quality)
test_probs75 <- probs75[probs75$dataType == "Test",]

probs65 <- extractProb(list(glm=glmnFit65), testX=X[-trainingRows,c(-tooHigh65)], testY=df_pp[-trainingRows,]$quality)
test_probs65 <- probs65[probs65$dataType == "Test",]

probs50 <- extractProb(list(glm=glmnFit50), testX=X[-trainingRows,c(-tooHigh50)], testY=df_pp[-trainingRows,]$quality)
test_probs50 <- probs50[probs50$dataType == "Test",]


glmnetRoc <- roc(response = df_pp[-trainingRows,]$quality,
                 predictor = test_probs$good)
                 #levels = rev(levels(df_pp$quality)))
glmnetRoc

glmnetRoc75 <- roc(response = df_pp[-trainingRows,]$quality,
                 predictor = test_probs75$good)
#levels = rev(levels(df_pp$quality)))
glmnetRoc75

glmnetRoc65 <- roc(response = df_pp[-trainingRows,]$quality,
                 predictor = test_probs65$good)
#levels = rev(levels(df_pp$quality)))
glmnetRoc65

glmnetRoc50 <- roc(response = df_pp[-trainingRows,]$quality,
                 predictor = test_probs50$good)
#levels = rev(levels(df_pp$quality)))
glmnetRoc50

# Plot the ROCs
plot.new()
plot(glmnetRoc, print.thres=TRUE)
plot(glmnetRoc75, print.thres=TRUE)
plot(glmnetRoc65, print.thres=TRUE)
plot(glmnetRoc50, print.thres=TRUE)


