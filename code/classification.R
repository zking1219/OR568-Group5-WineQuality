# Wine quality classificaton: 3-8 and good/bad

# Read in the raw data
wine_tibble <- read_csv("/Users/iaa211/Documents/ContinuedEducation/GMU-Masters/OR568/Project/Data/winequality-red.csv")
df_raw <- as.data.frame(wine_tibble)

# Read in the pre-processed data
wine_tibble_preproc <- read_csv("/Users/iaa211/Documents/ContinuedEducation/GMU-Masters/OR568/Project/Data/winequality_center_scale_boxcox.csv")
df_pp <- as.data.frame(wine_tibble_preproc)

# Begin by building classification models treating 3, 4, 5, 6, 7, 8 as classes
########################

# Logistic Regression
## multi class penalized linear model (multinom) - logistic regression 
library(nnet)
set.seed(1056)

# Convert quality to a factor for logistic regression
df_pp$quality <- as.factor(df_pp$quality)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
logisticReg <- train(quality ~ ., data = df_pp, 
                     method = "multinom",
                     trControl = trainControl(method = "cv", number=10),
                     metric="Kappa")
logisticReg

# LDA
set.seed(1056)
# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
ldaFit <- train(quality ~ ., data = df_pp,
                method = "lda",
                trControl = trainControl(method = "cv", number=10),
                metric="Kappa")
ldaFit

# PLSDA
install.packages("pls")
library(pls)
set.seed(1056)
# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
plsFit <- train(quality ~ ., data = df_pp,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:11),
                preProc = c("center", "scale", "BoxCox"),
                trControl = trainControl(method = "cv", number=10),
                metric="Kappa")
plsFit


# Penalized Model
install.packages("glmnet")
library(glmnet)

#glmnGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1), lambda = seq(.01, .2, length = 40))
glmnGrid <- expand.grid(alpha = c( .2), lambda = seq(.001, .002, length = 10))
set.seed(1056)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
glmnFit <- train(quality ~ ., data = df_pp,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 trControl = trainControl(method = "cv", number=10),
                 metric="Kappa")
glmnFit

# Nearest Centroid
install.packages("pamr")
library(pamr)
set.seed(1056)
nscGrid <- data.frame(.threshold = 0:25)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied.
nscTuned <- train(quality ~ ., data = df_pp,
                  method = "pam",
                  tuneGrid = nscGrid,
                  trControl = trainControl(method = "cv", number=10),
                  metric="Kappa")

nscTuned


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


# Logistic Regression
## multi class penalized linear model (multinom) - logistic regression 
library(nnet)
set.seed(1056)

# Convert quality to a factor for logistic regression
df_pp$quality <- as.factor(df_pp$quality)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
logisticReg <- train(quality ~ ., data = df_pp, 
                     method = "glm",
                     trControl = trainControl(method = "cv", number=10),
                     metric="Kappa")
logisticReg

# LDA
set.seed(1056)
# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
ldaFit <- train(quality ~ ., data = df_pp,
                method = "lda",
                trControl = trainControl(method = "cv", number=10),
                metric="Kappa")
ldaFit

# PLSDA
install.packages("pls")
library(pls)
set.seed(1056)
# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
plsFit <- train(quality ~ ., data = df_pp,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:11),
                preProc = c("center", "scale", "BoxCox"),
                trControl = trainControl(method = "cv", number=10),
                metric="Kappa")
plsFit


# Penalized Model
install.packages("glmnet")
library(glmnet)

#glmnGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1), lambda = seq(.01, .2, length = 40))
glmnGrid <- expand.grid(alpha = c( .2), lambda = seq(.001, .002, length = 10))
set.seed(1056)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied. 
glmnFit <- train(quality ~ ., data = df_pp,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 trControl = trainControl(method = "cv", number=10),
                 metric="Kappa")
glmnFit

# Nearest Centroid
install.packages("pamr")
library(pamr)
set.seed(1056)
nscGrid <- data.frame(.threshold = 0:25)

# No preprocessing needed within train, df_pp has already had center, scale, box cox 
# transformations applied.
nscTuned <- train(quality ~ ., data = df_pp,
                  method = "pam",
                  tuneGrid = nscGrid,
                  trControl = trainControl(method = "cv", number=10),
                  metric="Kappa")

nscTuned
