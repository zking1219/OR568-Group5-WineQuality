wine <-read.csv("c:/Users/17246/Downloads/winequality-red.csv")

head(wine)


library(ISLR)
library(e1071)

smp_size <- floor(.70*nrow(wine))
smp_size

train_ind <- sample(seq_len(nrow(wine)), size = smp_size)

trainer <- wine[train_ind,]
tester <- wine[-train_ind,]
trainer$quality <- as.numeric(trainer$quality)

ctrl1 <- trainControl(method = "LOOCV",
                     classProbs = FALSE,
                     savePredictions = TRUE)
ctrl2 <- trainControl(method = "cv",
                     number = 10)

# No near variance among predictors. 
nearZeroVar(trainer)

pptrain <- preProcess(trainer[,-12], method = c("BoxCox", "center", "scale"))

trainmod <- predict(pptrain, trainer)

# skew fix comparisons
apply(trainer[,-12], 2, skewness)
apply(log(trainer[,-12]), 2, skewness)


pr <- prcomp(trainmod[,-12], scale = TRUE)

pv <- pr$sdev^2/sum(pr$sdev)*100
pv

plot(pv, xlab = "Component", ylab = "Percent Variance", type = "b")


library(ggcorrplot)
coral <- cor(trainer)
windows()
ggcorrplot(coral, type = "lower", lab = TRUE)


#Regression Models

trainer$quality <- as.numeric(trainer$quality)

lmfit1 <- train(trainer[,-12], y = trainer$quality,
                method = "lm",
                trControl = ctrl2)




#Basic Linear and Logistic Regression Models

#Need to Plot

lrm <- lrm(quality ~., data = trainer, x = TRUE, y = TRUE)
pred <- predict(lrm, trainer)

windows()
plot(pred)


trainer$quality <- as.factor(trainer$quality)
str(trainer)

multifit <- train(trainer[,-12], y = trainer$quality,
            method = "multinom",
            preProc = c("BoxCox", "center", "scale"),
            trControl = ctrl2)

pdafit <- train(trainer[,-12], y = trainer$quality,
                method = "pda",
                preProc = c("BoxCox", "center", "scale"),
                trControl = ctrl1)

trainer$quality <- as.numeric(trainer$quality)

glmfit <- train(trainer[,-12], y = trainer$quality,
                  method = "glm",
                  preProc = c("BoxCox", "center", "scale"),
                  trControl = ctrl1)


plsfit <- train(trainer[,-12], y = trainer$quality,
                method = "pls",
                preProc = c("BoxCox", "center", "scale"),
                trControl = ctrl1)

pdafit <- train(trainer[,-12], y = trainer$quality,
                method = "pda",
                preProc = c("BoxCox", "center", "scale"),
                trControl = ctrl1)

#no R^2 score given in LASSO
lassoreg <- train(trainer[,-12],
             y = trainer$quality,
             method = 'glmnet', 
             preProc = c("BoxCox", "center", "scale"),
             tuneGrid = expand.grid(alpha = 1, lambda = 1)) 

ridgereg <- train(trainer[,-12],
             y = trainer$quality,
             method = 'glmnet', 
             preProc = c("BoxCox", "center", "scale"),
             tuneGrid = expand.grid(alpha = 0, lambda = 1)) 



