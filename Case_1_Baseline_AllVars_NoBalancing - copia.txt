# ==============================================================================
# SCRIPT 1: ALL VARIABLES / NO BALANCING (7 MODELS) - ROBUST FIX
# ==============================================================================

# 1. SETUP
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(randomForest)) install.packages("randomForest")
if (!require(gbm)) install.packages("gbm")
if (!require(kernlab)) install.packages("kernlab") # For SVM
if (!require(e1071)) install.packages("e1071")
if (!require(MASS)) install.packages("MASS")      # For LDA
if (!require(klaR)) install.packages("klaR")      # For Naive Bayes
if (!require(naivebayes)) install.packages("naivebayes")  # REQUIRED for naive_bayes method

library(tidyverse); library(caret); library(pROC); library(randomForest)
library(gbm); library(kernlab); library(e1071); library(MASS); library(klaR); library(naivebayes)

# Output Folder
dir.create("Results_Case1_All_NoBal", showWarnings = FALSE)

# 2. DATA PREP
filename <- "Data/studentmat.csv"
if (!file.exists(filename)) filename <- "studentmat.csv"
if (!file.exists(filename)) stop("Error: studentmat.csv not found.")

df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)

# Robust Numeric Conversion
numeric_cols <- c("age", "absences", "G3", "Medu", "Fedu", "traveltime", 
                  "studytime", "failures", "famrel", "freetime", "goout", 
                  "Dalc", "Walc", "health", "G1", "G2")

for(col in numeric_cols) {
  if(col %in% names(df_raw)) {
    df_raw[[col]] <- as.numeric(df_raw[[col]])
  }
}

# Create Target & Remove Grades
df_class <- df_raw %>%
  mutate(Status = ifelse(G3 >= 10, "Pass", "Fail")) %>%
  mutate(Status = as.factor(Status)) %>%
  dplyr::select(-G1, -G2, -G3) %>% 
  mutate(across(where(is.character), as.factor))

# Check structure
print("Data structure after processing:")
print(str(df_class))

# 3. SPLIT
set.seed(123)
trainIndex <- createDataPartition(df_class$Status, p = 0.75, list = FALSE)
train_final <- df_class[trainIndex, ]
valid_final <- df_class[-trainIndex, ]

# 4. CONFIGURATION (NO BALANCING)
ctrl <- trainControl(method = "cv", number = 10, 
                     classProbs = TRUE, 
                     summaryFunction = twoClassSummary,
                     savePredictions = "final")

# 5. MODELING LOOP
print(">> CASE 1 STARTING: ALL VARS / NO BALANCE")

model_list <- c("glm", "knn", "svmRadial", "rpart", "rf", "gbm", "naive_bayes")
results_summary <- data.frame(Model=character(), AUC=numeric(), 
                              Accuracy=numeric(), Sensitivity=numeric(), 
                              Specificity=numeric(), stringsAsFactors = FALSE)

for (method in model_list) {
  print(paste("Training:", method))
  set.seed(123)
  
  fit <- tryCatch({
    if(method == "gbm") {
      train(Status ~ ., data = train_final, method = method, 
            metric="ROC", trControl = ctrl, 
            preProcess = c("center", "scale", "nzv"),
            verbose=FALSE,
            tuneLength = 3)
    } else if(method == "rf") {
      train(Status ~ ., data = train_final, method = method, 
            metric="ROC", trControl = ctrl, 
            preProcess = c("center", "scale", "nzv"),
            ntree = 100,
            tuneLength = 3)
    } else if(method == "naive_bayes") {
      # Naive Bayes doesn't need scaling
      train(Status ~ ., data = train_final, method = method, 
            metric="ROC", trControl = ctrl,
            preProcess = c("nzv"),
            tuneLength = 3)
    } else {
      train(Status ~ ., data = train_final, method = method, 
            metric="ROC", trControl = ctrl, 
            preProcess = c("center", "scale", "nzv"),
            tuneLength = 3)
    }
  }, error = function(e) { 
    print(paste("Error training", method, ":", e$message))
    return(NULL) 
  })
  
  if (!is.null(fit)) {
    # Evaluate
    probs <- predict(fit, valid_final, type = "prob")
    classes <- predict(fit, valid_final)
    cm <- confusionMatrix(classes, valid_final$Status, mode = "everything")
    
    # Handle Probabilities Column Name
    if("Pass" %in% colnames(probs)) {
      pass_col <- "Pass"
    } else if("pass" %in% colnames(probs)) {
      pass_col <- "pass"
    } else {
      pass_col <- colnames(probs)[2]
    }
    
    # Calculate ROC
    roc_obj <- tryCatch({
      roc(valid_final$Status, probs[, pass_col], 
          levels = c("Fail", "Pass"), 
          direction = "<")
    }, error = function(e) {
      print(paste("  ROC Error:", e$message))
      return(NULL)
    })
    
    if(!is.null(roc_obj)) {
      auc_val <- auc(roc_obj)
      
      # Save report
      sink(paste0("Results_Case1_All_NoBal/Res_", method, ".txt"))
      print(paste("MODEL:", method))
      print(paste("AUC:", round(auc_val, 4)))
      print(cm)
      sink()
      
      # Add to Summary
      results_summary <- rbind(results_summary, data.frame(
        Model = method,
        AUC = round(auc_val, 4),
        Accuracy = round(cm$overall['Accuracy'], 4),
        Sensitivity = round(cm$byClass['Sensitivity'], 4),
        Specificity = round(cm$byClass['Specificity'], 4),
        stringsAsFactors = FALSE
      ))
      
      # Save model
      saveRDS(fit, file = paste0("Results_Case1_All_NoBal/model_", method, ".rds"))
    }
  }
}

# 6. SAVE AND DISPLAY RESULTS
print("=== FINAL RESULTS ===")
print(results_summary[order(-results_summary$AUC), ])

write.table(results_summary, 
            "Results_Case1_All_NoBal/Summary_Leaderboard.txt", 
            sep = "\t", 
            row.names = FALSE)

# Save as CSV
write.csv(results_summary, 
          "Results_Case1_All_NoBal/Summary_Leaderboard.csv", 
          row.names = FALSE)

# 7. CREATE VISUALIZATION
if(nrow(results_summary) > 0) {
  library(ggplot2)
  
  p <- ggplot(results_summary, aes(x = reorder(Model, AUC), y = AUC, fill = AUC)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = round(AUC, 3)), hjust = -0.2, size = 3) +
    coord_flip() +
    scale_fill_gradient(low = "blue", high = "red") +
    labs(title = "Model Performance Comparison (All Variables, No Balancing)",
         x = "Model",
         y = "AUC") +
    theme_minimal() +
    ylim(0, 1)
  
  ggsave("Results_Case1_All_NoBal/Model_Comparison.png", 
         plot = p, width = 10, height = 6, dpi = 300)
  
  print("Model comparison plot saved.")
}

print("CASE 1 COMPLETED SUCCESSFULLY.")