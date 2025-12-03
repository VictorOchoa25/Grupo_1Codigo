# 1. SETUP & LIBRARIES
# ------------------------------------------------------------------------------
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(rpart)) install.packages("rpart")
if (!require(rpart.plot)) install.packages("rpart.plot")
if (!require(randomForest)) install.packages("randomForest")
if (!require(pROC)) install.packages("pROC") # Required for ROC Curves

library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(gridExtra)

# Create Results Folder
dir.create("Results", showWarnings = FALSE)

# Start Logging
sink("Results/Full_Modeling_Log.txt", split = TRUE)
print(paste("Analysis Started:", Sys.time()))

# 2. LOAD & PREPARE DATA
# ------------------------------------------------------------------------------
# Load Data
df_raw <- read.csv("Data/studentmat.csv", sep = ";", stringsAsFactors = FALSE)

# Clean Data
df_raw$G3 <- as.numeric(df_raw$G3)
df_raw$age <- as.numeric(df_raw$age)
df_raw$absences <- as.numeric(df_raw$absences)
df_raw$Medu <- as.numeric(df_raw$Medu)
df_raw$goout <- as.numeric(df_raw$goout)
df_raw$failures <- as.numeric(df_raw$failures)
df_raw$Mjob <- as.factor(df_raw$Mjob)

# FILTER: Keep only the "Top 6" + Target
selected_vars <- c("failures", "absences", "Medu", "age", "Mjob", "goout", "G3")
df_model <- df_raw[, selected_vars]

print("Data Loaded. Using variables: failures, absences, Medu, age, Mjob, goout")

# ==============================================================================
# PHASE A: REGRESSION (Predicting Exact Grade 0-20)
# ==============================================================================
print("=========================================")
print("PHASE A: REGRESSION ANALYSIS (Target: G3)")
print("=========================================")

# A1. Split Data (Regression)
set.seed(123)
trainIndex <- createDataPartition(df_model$G3, p = 0.75, list = FALSE)
train_reg <- df_model[trainIndex, ]
valid_reg <- df_model[-trainIndex, ]

# A2. Train Models (Regression)
ctrl_reg <- trainControl(method = "cv", number = 10)

# Linear Regression
print(">> Training Linear Regression...")
model_lm <- train(G3 ~ ., data = train_reg, method = "lm", trControl = ctrl_reg)

# Random Forest (Regression)
print(">> Training Random Forest (Reg)...")
model_rf_reg <- train(G3 ~ ., data = train_reg, method = "rf", trControl = ctrl_reg)

# Decision Tree (Regression)
print(">> Training Decision Tree (Reg)...")
model_tree_reg <- train(G3 ~ ., data = train_reg, method = "rpart", trControl = ctrl_reg)

# A3. Evaluate (RMSE, R2, MAE)
preds_lm <- predict(model_lm, valid_reg)
preds_rf <- predict(model_rf_reg, valid_reg)
preds_tree <- predict(model_tree_reg, valid_reg)

calc_metrics <- function(actual, predicted, model_name) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  r2 <- cor(actual, predicted)^2
  return(c(Model = model_name, RMSE = round(rmse, 2), MAE = round(mae, 2), R2 = round(r2, 2)))
}

results_reg <- rbind(
  calc_metrics(valid_reg$G3, preds_lm, "Linear Regression"),
  calc_metrics(valid_reg$G3, preds_rf, "Random Forest"),
  calc_metrics(valid_reg$G3, preds_tree, "Decision Tree")
)

print("--- REGRESSION RESULTS (RMSE = Lower is Better) ---")
print(as.data.frame(results_reg))
write.table(results_reg, "Results/REG_Metrics.txt", sep="\t", quote=FALSE, row.names=FALSE)

# A4. Plot Regression Predictions
png("Results/REG_Actual_vs_Predicted.png", width=800, height=400)
par(mfrow=c(1,2))
plot(valid_reg$G3, preds_lm, main="Linear Reg: Actual vs Pred", xlab="Actual", ylab="Predicted", pch=19, col="blue")
abline(0,1, col="red", lwd=2)
plot(valid_reg$G3, preds_rf, main="Random Forest: Actual vs Pred", xlab="Actual", ylab="Predicted", pch=19, col="green")
abline(0,1, col="red", lwd=2)
dev.off()