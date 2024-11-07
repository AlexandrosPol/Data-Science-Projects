# Feature Selection and linear regression
# Author: Alexandros Polyzoidis

# Overview:
# This script demonstrates feature selection techniques and linear regression modeling 
# to predict the age (rings) of abalones based on physical measurements.

# Install and load necessary packages
if (!require("AppliedPredictiveModeling")) install.packages("AppliedPredictiveModeling")
if (!require("caret")) install.packages("caret")
if (!require("Metrics")) install.packages("Metrics")
library(AppliedPredictiveModeling)
library(caret)
library(Metrics)

# Data Preparation
# Load the abalone dataset from the AppliedPredictiveModeling package
data("abalone", package = "AppliedPredictiveModeling")

# Exclude non-predictive variables: 'Type' (categorical) and 'Rings' (target)
predictors <- abalone[ ,-c(1,9)]
head(predictors)

# Keep the 'Rings' variable as the response variable 'y'
y <- abalone$Rings
head(y)

# Feature Importance Analysis
# Rank the variables of the “df_data” data frame in relation to “y” (the age) and store the results
ranking <- filterVarImp(x = predictors, y = target)
ranking

# Sort the variables by decreasing importance
ranking$varNames <- rownames(ranking)
ranked_vars <- ranking[order(-ranking$Overall), ]
ranked_vars
ranked_vars$varNames

# Display the rounded overall importance scores
round(ranked_vars$Overall, 3)

# Answer (b)
# (1) Construct a data frame that contains the 2 top-ranked variables from the previous section
# Extract the top 2 ranked variables from the previous section
x_rank <- df_data[ ,ranked_vars$varNames[1:2]]
head(x_rank)

# (2) Normalize the variables of the data frame
x_rank <- as.data.frame(scale(x_rank))
head(x_rank)

# (3) Construct a linear regression model for predicting the age (“y”) from the variables in “x_rank”.
# Combine the response variable 'y' and the top-ranked variables 'x_rank' into a new data frame 'data2'
data2 <- as.data.frame(cbind(y, x_rank))

# Display the first few rows of the combined data frame
head(data2)

# Define a linear regression model ('model_rank') to predict 'y' from all variables in 'data2'
model_rank <- lm(y~., data = data2)
model_rank

# Display the linear regression model summary
summary(model_rank)

# (4) Calculate the mean squared error of the age (“y”) in relation to the predictions 
# made by the linear regression model, using the fitting data.
mse_value <- round(mse(y, predict(model_rank, data2)), 3)

# Display the calculated mean squared error
cat("Mean Squared Error:", mse_value)

# Answer (c)
# Scale the data of the "df_data" data frame using the "scale" function
scaled_df_data <- scale(df_data)
head(scaled_df_data)

# Calculate the principal components of the scaled data
results_PCA <- prcomp(scaled_df_data, center=TRUE, scale=TRUE)
results_PCA

# Display a summary of the principal component analysis
summary(results_PCA)

# Store the transformed data in a new data frame
pca_data_transformed <- as.data.frame(predict(results_PCA))
head(pca_data_transformed)

#(i) Report the proportion of “variance” of the principal components
Prop_of_Var <- round(summary(results_PCA)$importance["Proportion of Variance", ],3)
Prop_of_Var

# (ii) Report the percentage of the variance explained by the two first components.
Percentage <- sum(Prop_of_Var[1:2]) * 100
Percentage

# Answer (d)
# Extract the principal components from the PCA results
x_PCA <- results_PCA$x
head(x_PCA)

# (1) Save to a new data frame the 2 first Principal Components (PC1, PC2) calculated before.
x_PCA2 <- as.data.frame(x_PCA[ ,1:2])
head(x_PCA2)

# (2) Include in the same data frame the “Rings” variable that you will call “age”. 
x_PCA2$age <- y
head(x_PCA2)
class(x_PCA2$age)

# (3) Define a linear regression model for predicting the age from the 2 first principal components.
model_PCA <- lm(x_PCA2$age~.,data = x_PCA2)
model_PCA

# (4) Calculate the MSE of fit of the real “age” values in relation to the predicted values of “age”. 
mse <- round(mse(y, predict(model_PCA,x_PCA2)),3)
mse

# Comments on the results
# Before PCA, the MSE was 6.297, indicating the model's performance in predicting 'age'. 
# After applying PCA and using the first two principal components, the MSE increased to 6.645. 
# The application of PCA resulted in a simplified model with fewer predictors, 
# enhancing interpretability due to the use of principal components. 
# However, this simplification came at the cost of a slight decrease in predictive accuracy, 
# as indicated by the higher MSE.
