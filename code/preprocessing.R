library(AppliedPredictiveModeling)
library(readr)
library(caret)
wine_tibble <- read_csv("../data/winequality-red.csv")
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

write.csv(df_preproc, 
          "../data/winequality_center_scale_boxcox.csv",
          row.names = FALSE)


