# ==============================================================================
# SCRIPT 2: ALL VARIABLES / WITH BALANCING (Up-Sampling)
# ==============================================================================

# 1. SETUP
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(randomForest)) install.packages("randomForest")
if (!require(gbm)) install.packages("gbm")
if (!require(kernlab)) install.packages("kernlab")
if (!require(e1071)) install.packages("e1071")
if (!require(MASS)) install.packages("MASS")

library(tidyverse); library(caret); library(pROC); library(randomForest)
library(gbm); library(kernlab); library(e1071); library(MASS)

# Output Folder
dir.create("Results_Case2_All_Balanced", showWarnings = FALSE)

# 2. DATA PREP
filename <- "Data/studentmat.csv"
if (!file.exists(filename)) filename <- "studentmat.csv"
if (!file.exists(filename)) stop("Error: studentmat.csv not found.")

df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)

# Convert ALL numeric columns properly
numeric_cols <- c("age", "absences", "G3", "Medu", "Fedu", "traveltime", 
                  "studytime", "failures", "famrel", "freetime", "goout", 
                  "Dalc", "Walc", "health", "G1", "G2")

for(col in numeric_cols) {
  if(col %in% names(df_raw)) {
    df_raw[[col]] <- as.numeric(df_raw[[col]])
  }
}

# Create Target & Remove Grades - FIXED with dplyr::select
df_class <- df_raw %>%
  mutate(Status = ifelse(G3 >= 10, "Pass", "Fail")) %>%
  mutate(Status = as.factor(Status)) %>%
  dplyr::select(-G1, -G2, -G3) %>%  # Use dplyr::select to avoid conflict
  mutate(across(where(is.character), as.factor))

print("Data structure after processing:")
print(str(df_class))
print(paste("Rows:", nrow(df_class)))
print(paste("Columns:", ncol(df_class)))
print(table(df_class$Status))

# 3. SPLIT
set.seed(123)
trainIndex <- createDataPartition(df_class$Status, p = 0.75, list = FALSE)
train_final <- df_class[trainIndex, ]
valid_final <- df_class[-trainIndex, ]

print("Train set class distribution:")
print(table(train_final$Status))
print("Validation set class distribution:")
print(table(valid_final$Status))

# 4. CONFIGURATION (WITH BALANCING)
ctrl <- trainControl(method = "cv", number = 10, 
                     classProbs = TRUE, 
                     summaryFunction = twoClassSummary,
                     sampling = "up",  # <--- Up-Sampling Active
                     savePredictions = "final")

# 5. MODELING LOOP
print(">> CASE 2 STARTING: ALL VARS / WITH BALANCE")

# Remove lda if it causes issues, or handle it separately
model_list <- c("glm", "knn", "svmRadial", "rpart", "rf", "gbm")
results_summary <- data.frame(Model = character(), 
                              AUC = numeric(), 
                              Accuracy = numeric(),
                              Sensitivity = numeric(), 
                              Specificity = numeric(),
                              stringsAsFactors = FALSE)

for (method in model_list) {
  print(paste("Training:", method))
  set.seed(123)
  
  fit <- tryCatch({
    if(method == "gbm") {
      train(Status ~ ., data = train_final, method = method, 
            metric = "ROC", trControl = ctrl, 
            preProcess = c("center", "scale", "nzv"),
            verbose = FALSE,
            tuneLength = 3)
    } else if(method == "rf") {
      train(Status ~ ., data = train_final, method = method, 
            metric = "ROC", trControl = ctrl, 
            preProcess = c("center", "scale", "nzv"),
            ntree = 100,
            tuneLength = 3)
    } else {
      train(Status ~ ., data = train_final, method = method, 
            metric = "ROC", trControl = ctrl, 
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
      
      # Save detailed report
      sink(paste0("Results_Case2_All_Balanced/Res_", method, ".txt"))
      print(paste("MODEL:", method))
      print(paste("AUC:", round(auc_val, 4)))
      print("Confusion Matrix:")
      print(cm$table)
      print("Detailed Metrics:")
      print(cm$byClass)
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
      saveRDS(fit, file = paste0("Results_Case2_All_Balanced/model_", method, ".rds"))
    }
  }
}

# 6. SAVE AND DISPLAY RESULTS
print("=== FINAL RESULTS (WITH BALANCING) ===")
print(results_summary[order(-results_summary$AUC), ])

write.table(results_summary, 
            "Results_Case2_All_Balanced/Summary_Leaderboard.txt", 
            sep = "\t", 
            row.names = FALSE)

write.csv(results_summary, 
          "Results_Case2_All_Balanced/Summary_Leaderboard.csv", 
          row.names = FALSE)

# 7. CREATE VISUALIZATION
if(nrow(results_summary) > 0) {
  library(ggplot2)
  
  p <- ggplot(results_summary, aes(x = reorder(Model, AUC), y = AUC, fill = AUC)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = round(AUC, 3)), hjust = -0.2, size = 3) +
    coord_flip() +
    scale_fill_gradient(low = "blue", high = "red") +
    labs(title = "Model Performance Comparison (All Variables, With Up-Sampling)",
         subtitle = "Balancing applied to handle class imbalance",
         x = "Model",
         y = "AUC") +
    theme_minimal() +
    ylim(0, 1)
  
  ggsave("Results_Case2_All_Balanced/Model_Comparison.png", 
         plot = p, width = 10, height = 6, dpi = 300)
  
  print("Model comparison plot saved.")
}

# 8. COMPARE WITH PREVIOUS CASE (if available) - FIXED
if(file.exists("Results_Case1_All_NoBal/Summary_Leaderboard.csv")) {
  prev_results <- read.csv("Results_Case1_All_NoBal/Summary_Leaderboard.csv")
  
  # FIXED: Use dplyr::select() and dplyr::rename() explicitly
  prev_data <- prev_results %>% 
    dplyr::select(Model, AUC_NoBalance = AUC, Sens_NoBalance = Sensitivity) %>% 
    dplyr::rename(Model1 = Model)
  
  curr_data <- results_summary %>% 
    dplyr::select(Model, AUC_Balance = AUC, Sens_Balance = Sensitivity)
  
  comparison <- merge(
    prev_data,
    curr_data,
    by.x = "Model1", by.y = "Model", all = TRUE
  )
  
  print("=== COMPARISON: NO BALANCE vs WITH BALANCE ===")
  print(comparison)
  
  write.csv(comparison, 
            "Results_Case2_All_Balanced/Comparison_with_NoBalance.csv", 
            row.names = FALSE)
}

print("CASE 2 COMPLETED SUCCESSFULLY.")

