# ==============================================================================
# MASTER SCRIPT: STUDENT PERFORMANCE (FULL PIPELINE)
# ==============================================================================
# PHASES:
# 1. Data Prep (Cleaning & Target Definition)
# 2. EDA (Exploratory Analysis)
# 3. Feature Selection (Chi-Square, Boruta, RFE)
# 4. Modeling (Logistic, RF, SVM, GBM) with Up-Sampling
# ==============================================================================

# 1. SETUP & LIBRARIES
# ------------------------------------------------------------------------------
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(randomForest)) install.packages("randomForest")
if (!require(Boruta)) install.packages("Boruta")
if (!require(kernlab)) install.packages("kernlab")
if (!require(gbm)) install.packages("gbm")
if (!require(gridExtra)) install.packages("gridExtra")

library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(Boruta)
library(kernlab)
library(gbm)
library(gridExtra)

# Create Output Directory
if (!dir.exists("Results_Classification")) {
  dir.create("Results_Classification")
}

# Start Main Log
sink("Results_Classification/Full_Pipeline_Log.txt", split = TRUE)
print("--- STARTING FULL PIPELINE ANALYSIS ---")

# 2. DATA LOADING & PREPARATION
# ------------------------------------------------------------------------------
# Adjust path if necessary (e.g., "Data/studentmat.csv")
filename <- "Data/studentmat.csv" 
if (!file.exists(filename)) {
  # Fallback to root directory if Data folder doesn't exist
  filename <- "studentmat.csv"
}
if (!file.exists(filename)) stop("Error: studentmat.csv not found.")

df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)

# Clean numeric variables
df_raw$age <- as.numeric(df_raw$age)
df_raw$absences <- as.numeric(df_raw$absences)
df_raw$G3 <- as.numeric(df_raw$G3)
df_raw$Medu <- as.numeric(df_raw$Medu)
df_raw$failures <- as.numeric(df_raw$failures)
df_raw$goout <- as.numeric(df_raw$goout)

# --- DEFINE TARGET (PASS/FAIL) ---
df_class <- df_raw %>%
  mutate(Status = ifelse(G3 >= 10, "Pass", "Fail")) %>%
  mutate(Status = as.factor(Status)) %>%
  select(-G1, -G2, -G3) # Remove grades to avoid leakage

# Convert characters to factors
df_class <- df_class %>%
  mutate(across(where(is.character), as.factor))

print("Target Variable 'Status' Created. Distribution:")
print(table(df_class$Status))

# 3. DATA PARTITIONING
# ------------------------------------------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(df_class$Status, p = 0.75, list = FALSE)
train_set <- df_class[trainIndex, ]
valid_set <- df_class[-trainIndex, ]

print(paste("Train Set:", nrow(train_set), "rows | Valid Set:", nrow(valid_set), "rows"))

# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("--- PHASE 4: RUNNING EDA ---")

# Save plots function
save_plot <- function(p, name) {
  png(paste0("Results_Classification/EDA_", name, ".png"), width = 600, height = 400)
  print(p)
  dev.off()
}

# Generate Plots Loop
vars <- names(train_set)
for (v in vars) {
  if (v != "Status") {
    if (is.numeric(train_set[[v]])) {
      # Boxplot for numerics
      p <- ggplot(train_set, aes(x = Status, y = .data[[v]], fill = Status)) +
        geom_boxplot(alpha = 0.6) + labs(title = paste(v, "by Status")) + theme_minimal()
    } else {
      # Bar chart for categorical
      p <- train_set %>%
        group_by(.data[[v]]) %>%
        summarise(Fail_Rate = mean(Status == "Fail"), Count = n()) %>%
        ggplot(aes(x = .data[[v]], y = Fail_Rate, fill = .data[[v]])) +
        geom_bar(stat = "identity", alpha = 0.8) +
        labs(title = paste("Failure Rate by", v), y = "Fail Rate") + theme_minimal() + theme(legend.position="none")
    }
    save_plot(p, v)
  }
}
print("EDA Plots Saved.")

# ==============================================================================
# 5. FEATURE SELECTION ALGORITHMS
# ==============================================================================
print("--- PHASE 5: RUNNING FEATURE SELECTION ---")

# A. Chi-Squared (Categorical Relations)
print(">> Running Chi-Squared Tests...")
cat_vars <- names(train_set)[sapply(train_set, is.factor)]
chi_results <- data.frame(Variable = character(), P_Value = numeric())

for (v in cat_vars) {
  if (v != "Status") {
    test <- chisq.test(table(train_set[[v]], train_set$Status))
    chi_results <- rbind(chi_results, data.frame(Variable = v, P_Value = test$p.value))
  }
}
chi_results <- chi_results[order(chi_results$P_Value), ]
write.table(chi_results, "Results_Classification/Selection_ChiSquared.txt", sep="\t", row.names=FALSE, quote=FALSE)
print("Chi-Squared Results Saved.")

# B. Boruta (All-Relevant Selection)
print(">> Running Boruta Algorithm (This may take a moment)...")
set.seed(123)
boruta_cls <- Boruta(Status ~ ., data = train_set, doTrace = 0)
print(boruta_cls)

# Save Boruta Plot
png("Results_Classification/Selection_Boruta_Plot.png", width=800, height=600)
plot(boruta_cls, cex.axis=.7, las=2, xlab="", main="Boruta Importance")
dev.off()

# Save Boruta Table
boruta_stats <- attStats(boruta_cls)
boruta_stats$Variable <- rownames(boruta_stats)
write.table(boruta_stats, "Results_Classification/Selection_Boruta_Table.txt", sep="\t", row.names=FALSE, quote=FALSE)

# C. RFE (Recursive Feature Elimination)
print(">> Running RFE Algorithm...")
ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
set.seed(123)
rfe_cls <- rfe(train_set[, names(train_set) != "Status"], 
               train_set$Status, sizes = c(1:15), rfeControl = ctrl_rfe)
print(rfe_cls)

# Save RFE Plot
png("Results_Classification/Selection_RFE_Accuracy.png")
plot(rfe_cls, type=c("g", "o"))
dev.off()

# ==============================================================================
# 6. SELECTION DECISION (THE BRIDGE)
# ==============================================================================
print("--- PHASE 6: FINAL VARIABLE SELECTION ---")

# Based on the outputs generated in Phase 5 (Boruta & RFE), we define the 
# optimal subset of variables here for the modeling phase.
# -------------------------------------------------------------------------
# RATIONALE:
# 1. 'failures', 'absences', 'age': Consistently top-ranked in all methods.
# 2. 'higher', 'guardian', 'Medu': Identified by Boruta/Chi-Square as key social factors.
# 3. 'goout': Identified by Random Forest importance as a behavioral factor.
# -------------------------------------------------------------------------

selected_vars_cls <- c("failures", "absences", "higher", "age", "Medu", "goout", "guardian", "Status")

print("Variables selected for Modeling:")
print(selected_vars_cls)

# Filter Data
train_final <- train_set[, selected_vars_cls]
valid_final <- valid_set[, selected_vars_cls]

# ==============================================================================
# 7. MODEL TRAINING (4 MODELS + UP-SAMPLING)
# ==============================================================================
print("--- PHASE 7: TRAINING MODELS ---")

# Control: 10-Fold CV + Up-Sampling for Balance
ctrl_cls <- trainControl(method = "cv",
                         number = 10,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary,
                         sampling = "up") # Native Balancing

# A. Logistic Regression
print(">> Training Logistic Regression...")
set.seed(123)
model_glm <- train(Status ~ ., data = train_final, method = "glm", family = "binomial", 
                   metric = "ROC", trControl = ctrl_cls)

# B. Random Forest
print(">> Training Random Forest...")
set.seed(123)
model_rf <- train(Status ~ ., data = train_final, method = "rf", 
                  metric = "ROC", trControl = ctrl_cls)

# C. SVM Radial
print(">> Training SVM Radial...")
set.seed(123)
model_svm <- train(Status ~ ., data = train_final, method = "svmRadial", 
                   metric = "ROC", preProcess = c("center", "scale"), trControl = ctrl_cls)

# D. Gradient Boosting
print(">> Training Gradient Boosting...")
set.seed(123)
model_gbm <- train(Status ~ ., data = train_final, method = "gbm", 
                   metric = "ROC", trControl = ctrl_cls, verbose = FALSE)

# ==============================================================================
# 8. EVALUATION & RESULTS
# ==============================================================================
print("--- PHASE 8: EVALUATION ---")

evaluate_and_save <- function(model, test_data, model_name) {
  probs <- predict(model, test_data, type = "prob")
  classes <- predict(model, test_data)
  
  cm <- confusionMatrix(classes, test_data$Status, mode = "everything")
  roc_obj <- roc(test_data$Status, probs[, "Pass"], levels = c("Fail", "Pass"), direction = "<")
  auc_val <- auc(roc_obj)
  
  # Save to TXT
  filename <- paste0("Results_Classification/Result_", gsub(" ", "_", model_name), ".txt")
  sink(filename)
  print(paste("=== RESULTS FOR:", model_name, "==="))
  print(paste("AUC Score:", round(auc_val, 4)))
  print(cm$table)
  print(cm$byClass)
  sink()
  
  return(list(roc_obj = roc_obj, auc = auc_val))
}

# Run Eval
res_glm <- evaluate_and_save(model_glm, valid_final, "Logistic Regression")
res_rf  <- evaluate_and_save(model_rf, valid_final, "Random Forest")
res_svm <- evaluate_and_save(model_svm, valid_final, "SVM Radial")
res_gbm <- evaluate_and_save(model_gbm, valid_final, "Gradient Boosting")

# Final Leaderboard
summary_table <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM Radial", "Gradient Boosting"),
  AUC = c(res_glm$auc, res_rf$auc, res_svm$auc, res_gbm$auc)
)
summary_table <- summary_table[order(-summary_table$AUC),]
write.table(summary_table, "Results_Classification/Final_Model_Leaderboard.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# Final ROC Plot
png("Results_Classification/FINAL_ROC_Comparison.png", width = 800, height = 600)
plot(res_glm$roc_obj, col = "blue", lwd = 2, main = "ROC Comparison (4 Models)")
plot(res_rf$roc_obj, col = "red", lwd = 2, add = TRUE)
plot(res_svm$roc_obj, col = "purple", lwd = 2, add = TRUE)
plot(res_gbm$roc_obj, col = "orange", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("GLM (", round(res_glm$auc,2), ")"), 
                  paste("RF (", round(res_rf$auc,2), ")"),
                  paste("SVM (", round(res_svm$auc,2), ")"),
                  paste("GBM (", round(res_gbm$auc,2), ")")),
       col = c("blue", "red", "purple", "orange"), lwd = 2)
dev.off()

print("--- FULL PIPELINE COMPLETE. ALL RESULTS IN 'Results_Classification/' ---")
sink() # Close Main Log