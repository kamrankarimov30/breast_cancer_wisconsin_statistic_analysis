library(tidyverse)
library(mice)

# Loading dataset
data <- read.csv("wisconsin.csv")

# Exploratory Data Analysis
# Displaying first few rows of dataset
head(data)

# summary and structure of data
summary(data)
str(data)

# Checking for missing values
sum(is.na(data))

# Imputing Missing Values using MICE
# 'mice' function is applied to columns 2 to 10, excluding the ID column
dataset_impute <- mice(data[,2:10], print = FALSE)

# Combining the imputed dataset with the target class
# Target class is in the 10th column
BreastCancer <- cbind(data[,10, drop = FALSE], mice::complete(dataset_impute, 1))

# structure of 'Class'
table(BreastCancer$Class)


# Encoding Categorical Data
# Converting 'Class' to a binary factor variable: benign as 0 and malignant as 1
BreastCancer$Class <- ifelse(BreastCancer$Class == 'benign', 0, 1)
BreastCancer$Class <- factor(BreastCancer$Class, levels = c(0, 1))


# Renaming last column to 'Class_Label'
colnames(BreastCancer)[ncol(BreastCancer)] <- 'Class_Label'

# Displaying first few rows of processed dataset
head(BreastCancer)

# Checking for missing values
sum(is.na(BreastCancer))


#########################################
# Splitting test and train data

# Loading caTools library for dataset splitting
library(caTools)

# Seed for reproducibility
set.seed(150)

# Splitting dataset into training and test sets
split <- sample.split(BreastCancer$Class, SplitRatio = 0.7)
training_set <- subset(BreastCancer, split == TRUE)
test_set <- subset(BreastCancer, split == FALSE)

# Checking the dimensions of training and test datasets
dim(training_set)
dim(test_set)

# Preparing 'topredict_set' by removing the target class and class label
# This set will be used for predictions
topredict_set <- test_set[,2:9]
dim(topredict_set)


########################
# Classifiers

# Naive Bayes

library(e1071) # for naiveBayes
library(caret) # for confusionMatrix

# Implementing Naive Bayes
model_naive <- naiveBayes(Class ~ ., data = training_set)

preds_naive <- predict(model_naive, newdata = topredict_set)

# Creating a confusion matrix
conf_matrix_naive <- table(preds_naive, test_set$Class)

# Printing confusion matrix
conf_matrix_naive

# Calculating and printing accuracy of the model using confusion matrix
confusionMatrix(conf_matrix_naive)


#################
# Logistic Regression

# Excluding 'Class_Label' from training_set
training_set <- training_set[, -which(names(training_set) == 'Class_Label')]

# Implementing Logistic Regression
model_logistic <- glm(Class ~ ., data = training_set, family = binomial)

# Checking for convergence
if (!model_logistic$converged) {
  warning("glm algorithm did not converge!")
}

# Making sure that same columns are in topredict_set as in training_set
topredict_set <- topredict_set[, names(topredict_set) %in% names(training_set)]

# Predicting target class for the validation set


# code assigns predictions (1 or 0) based on probability estimates from a Logistic Regression model.
# If the estimated probability of the positive class is > 0.5, it's classified as 1; otherwise, as 0.
# Parameters:
# - model_logistic: Trained Logistic Regression model.
# - newdata: Dataset for predictions (topredict_set).
# - type = "response": Specifies probability estimates.
# Output:
# - preds_logistic: Predictions using a 0.5 decision threshold.
preds_logistic <- ifelse(predict(model_logistic, newdata = topredict_set, type = "response") > 0.5, 1, 0)

# Creating confusion matrix
conf_matrix_logistic <- table(Predicted = preds_logistic, Actual = test_set$Class)

print(conf_matrix_logistic)

# Calculating and printing accuracy of the model using confusion matrix
confusionMatrix(conf_matrix_logistic)


################
# KNN
library(class)

# Set seed for reproducibility
set.seed(150)

# Preparing training and test sets by removing the 'Class_Label'
training_set_knn <- training_set[, !names(training_set) %in% "Class_Label"]
topredict_features <- topredict_set[, !names(topredict_set) %in% c("Class", "Class_Label")]

# Prepare training features without the 'Class' column
training_features <- training_set_knn[, !names(training_set_knn) %in% "Class"]
training_labels <- training_set_knn$Class

# Ensuring that number of rows in training features is equal to the number of class labels
if (nrow(training_features) != length(training_labels)) {
  stop("Number of rows in training features does not match the number of training class labels.")
}

# Example of determining best_k using cross-validation
k_values <- 1:20 # A range of k values to try

# Create a vector to store accuracy for each k
accuracy_values <- numeric(length(k_values))

# Loop over possible values of k
for (k in k_values) {
  preds <- knn(train = training_features, test = topredict_features, cl = training_labels, k = k)
  conf_matrix <- table(Predicted = preds, Actual = test_set$Class)
  accuracy_values[k] <- sum(diag(conf_matrix)) / sum(conf_matrix)
}

# Determining best k  
best_k <- which.max(accuracy_values)

# Printing best k value
print(paste("The best k value is:", best_k))

# knn using best k value
preds_knn <- knn(train = training_features,
                 test = topredict_features,
                 cl = training_labels,
                 k = best_k)

# Creating confusion matrix
conf_matrix_knn <- table(Predicted = preds_knn, Actual = test_set$Class)

# Printing confusion matrix
print(conf_matrix_knn)

# Calculating and printing accuracy of the model using confusion matrix
conf_matrix_knn_factor <- confusionMatrix(as.factor(preds_knn), as.factor(test_set$Class))

# Printing confusion matrix with detailed statistics
print(conf_matrix_knn_factor)


#################
# Random Forest Classifier

# Loading library
library(randomForest)

# Ensuring 'Class' is present in training_set
if (!"Class" %in% colnames(training_set)) {
  stop("'Class' column is missing in training_set")
}

# Creating modified training set by excluding 'Class'
training_set_rf <- training_set[, !names(training_set) %in% "Class_Label"]
topredict_features_rf <- topredict_set[, !names(topredict_set) %in% c("Class", "Class_Label")]

# Prepare training features without the 'Class' column
training_features_rf <- training_set_rf[, !names(training_set_rf) %in% "Class"]
training_labels_rf <- training_set_rf$Class

# Ensuring that number of rows in training features is equal to the number of class labels
if (nrow(training_features_rf) != length(training_labels_rf)) {
  stop("Number of rows in training features does not match the number of training class labels")
}

# Implementing Random Forest with importance and limited trees (5) for efficiency
model_rf <- randomForest(Class ~ ., data = training_set_rf, importance = TRUE, ntree = 5)

# Predicting target class for the validation set
preds_rf <- predict(model_rf, topredict_features_rf)

# Creating confusion matrix
conf_matrix_rf <- confusionMatrix(as.factor(preds_rf), as.factor(test_set$Class))

# Printing confusion matrix with detailed statistics
print(conf_matrix_rf)


########################
# Pairwise Scatter Plot
#------
# selecting relevant features for the plot
library(reshape2)
selected_features <- c("Cell.size", "Cell.shape", "Marg.adhesion", "Epith.c.size", "Bare.nuclei", "Bl.cromatin", "Normal.nucleoli", "Mitoses", "Class")

# Creating subset of dataset with those selected features
subset_data <- data[, selected_features]

subset_data <- na.omit(subset_data)
cor_df <- round(cor(subset_data[,-9]), 2)

melted_cor <- melt(cor_df)
ggplot(data = melted_cor, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value), size = 5) +
  scale_fill_gradient2(low = "blue", high = "red",
                       limit = c(-1,1), name="Correlation") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.background = element_blank())




##################
# ROC Results

library(pROC)

# probabilities for Naive Bayes
preds_nb <- predict(model_naive, newdata = topredict_set, type = "raw")

# Creating ROC curve
roc_nb <- roc(test_set$Class, preds_nb[, 2])

####

# probabilities for Logistic Regression
preds_logistic <- predict(model_logistic, newdata = topredict_set, type = "response")

# Creating ROC curve
roc_logistic <- roc(test_set$Class, preds_logistic)

######

# Creating ROC curve
roc_knn <- roc(test_set$Class, preds_knn)


##################
# probabilities for Random Forest
preds_rf <- as.numeric(predict(model_rf, newdata = topredict_set, type = "response"))

# Creating ROC curve
roc_rf <- roc(test_set$Class, preds_rf)


################
#Combinin ROC Curves

# Combine ROC curves in one plot
plot(roc_nb, col = "blue", lwd = 2, main = "ROC Curves")
lines(roc_logistic, col = "green", lwd = 2)
lines(roc_knn, col = "orange", lwd = 2)
lines(roc_rf, col = "red", lwd = 2)
auc_nb <- auc(roc_nb)
auc_logistic <- auc(roc_logistic)
auc_knn <- auc(roc_knn)
auc_rf <- auc(roc_rf)
legend_text <- c(paste('Naive Bayes AUC =', round(auc_nb, 3)),
                 paste('Logistic Regression AUC =', round(auc_logistic, 3)),
                 paste('KNN AUC =', round(auc_knn, 3)),
                 paste('Random Forest AUC =', round(auc_rf, 3)))
legend("bottomright", legend = legend_text, col = c("blue", "green", "orange", "red"), lwd = 2)
