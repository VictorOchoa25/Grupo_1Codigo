# ==============================================================================
# INTEGRATED SCRIPT: STUDENT PERFORMANCE ANALYSIS
# ==============================================================================
# OUTPUT: All files and logs saved in 'Resultados_CódigoFinal'
# ==============================================================================

# 1. SETUP & LIBRARIES
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(randomForest)) install.packages("randomForest")
if (!require(gbm)) install.packages("gbm")
if (!require(kernlab)) install.packages("kernlab")
if (!require(e1071)) install.packages("e1071")
if (!require(Boruta)) install.packages("Boruta")
if (!require(naivebayes)) install.packages("naivebayes")

library(tidyverse)
library(caret); library(pROC); library(randomForest)
library(gbm); library(kernlab); library(e1071); library(Boruta); library(naivebayes)

# ==============================================================================
# 2. MASTER DIRECTORY & LOGGING SETUP
# ==============================================================================
# Create the main master folder
main_dir <- "Resultados_CódigoFinal"
dir.create(main_dir, showWarnings = FALSE)

# Define the log file path
log_file <- file.path(main_dir, "FULL_CONSOLE_LOG.txt")

# Start Global Sinking
# split = TRUE allows you to see output in RStudio AND save it to the txt file
console_log <- file(log_file, open = "wt")
sink(console_log, split = TRUE) 
sink(console_log, type = "message") # Capture warnings/messages too

print("========================================================")
print(paste("STARTING EXECUTION:", Sys.time()))
print(paste("ALL OUTPUTS WILL BE SAVED IN:", main_dir))
print("========================================================")

# Create Sub-directories inside the Master Folder
dir_selection <- file.path(main_dir, "02_Selection_Results")
dir_case1     <- file.path(main_dir, "Results_Case1_All_NoBal")
dir_case2     <- file.path(main_dir, "Results_Case2_All_Balanced")
dir_case3     <- file.path(main_dir, "Results_Case3_Selected_NoBal")
dir_case4     <- file.path(main_dir, "Results_Case4_Selected_Balanced")

dir.create(dir_selection, showWarnings = FALSE)
dir.create(dir_case1, showWarnings = FALSE)
dir.create(dir_case2, showWarnings = FALSE)
dir.create(dir_case3, showWarnings = FALSE)
dir.create(dir_case4, showWarnings = FALSE)

# ==============================================================================
# 3. DATA PREPARATION (GLOBAL)
# ==============================================================================
print("--- STARTING DATA PREPARATION ---")

filename <- "Data/studentmat.csv"
if (!file.exists(filename)) filename <- "studentmat.csv"
if (!file.exists(filename)) {
  # Stop sinking before erroring out so you can see the error
  sink(); sink(type="message")
  stop("Error: studentmat.csv not found.")
}

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

print("Data structure:")
print(str(df_class))

print("=== CLASS DISTRIBUTION (COUNTS & PROPORTIONS) ===")
# Absolute numbers
print("Counts:")
print(table(df_class$Status))

# Proportions
print("Proportions:")
print(prop.table(table(df_class$Status)))
print("================================================")

# ==============================================================================
# 4. GLOBAL SPLIT
# ==============================================================================
set.seed(123)
trainIndex <- createDataPartition(df_class$Status, p = 0.75, list = FALSE)

# Full Datasets (For Cases 1 & 2)
train_full <- df_class[trainIndex, ]
valid_full <- df_class[-trainIndex, ]

# ==============================================================================
# 5. FEATURE SELECTION PROCESS
# ==============================================================================
print("--- STARTING FEATURE SELECTION ---")
# We remove the internal 'sink' here so the output goes to the MAIN log file
# but we still save the tables.

# A. Chi-Squared
print(">> Running Chi-Squared Tests...")
cat_vars <- names(df_class)[sapply(df_class, is.factor)]
chi_df <- data.frame(Variable=character(), P_Value=numeric())

for (v in cat_vars) {
  if (v != "Status") {
    test <- chisq.test(table(df_class[[v]], df_class$Status))
    chi_df <- rbind(chi_df, data.frame(Variable=v, P_Value=test$p.value))
  }
}
chi_df <- chi_df[order(chi_df$P_Value),]
write.table(chi_df, file.path(dir_selection, "01_ChiSquared_Ranking.txt"), sep="\t", row.names=FALSE, quote=FALSE)
print("Chi-Squared Results (Top 5):")
print(head(chi_df))

# B. Boruta
print(">> Running Boruta Algorithm...")
set.seed(123)
boruta_out <- Boruta(Status ~ ., data = df_class, doTrace = 0) 

png(file.path(dir_selection, "02_Boruta_Plot.png"), width=800, height=600)
plot(boruta_out, cex.axis=.7, las=2, xlab="", main="Boruta Importance")
dev.off()

boruta_decisions <- attStats(boruta_out)
boruta_decisions$Variable <- rownames(boruta_decisions)
write.table(boruta_decisions, file.path(dir_selection, "02_Boruta_Decisions.txt"), sep="\t", row.names=FALSE, quote=FALSE)

# C. RFE (Recursive Feature Elimination)
print(">> Running RFE Algorithm...")
ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
set.seed(123)
rfe_out <- rfe(df_class[, names(df_class) != "Status"], 
               df_class$Status, sizes = c(1:15), rfeControl = ctrl_rfe)

png(file.path(dir_selection, "03_RFE_Accuracy.png"))
plot(rfe_out, type=c("g", "o"))
dev.off()
print(paste("RFE Optimal Variables:", paste(predictors(rfe_out), collapse=", ")))


# DEFINE SELECTED VARIABLES FOR CASES 3 & 4
sel_vars <- c("failures", "absences", "higher", "age", "Medu", "goout", "guardian", "Status")
print("Variables Selected for Cases 3 & 4:")
print(sel_vars)

# Create Subsets for Selected Vars
train_sel <- train_full[, sel_vars]
valid_sel <- valid_full[, sel_vars]

# ==============================================================================
# 6. HELPER FUNCTION: MODELING PIPELINE
# ==============================================================================
run_experiment_case <- function(case_name, output_dir, train_data, valid_data, ctrl_params, model_list) {
  
  print(paste0(">> STARTING: ", case_name))
  
  summary_df <- data.frame(Model=character(), AUC=numeric(), Accuracy=numeric(), 
                           Sensitivity=numeric(), Specificity=numeric(), stringsAsFactors=FALSE)
  
  for (method in model_list) {
    print(paste("   Training:", method))
    set.seed(123)
    
    fit <- tryCatch({
      if(method == "gbm") {
        train(Status ~ ., data = train_data, method = method, metric="ROC", trControl = ctrl_params, 
              preProcess = c("center", "scale", "nzv"), verbose=FALSE, tuneLength = 3)
      } else if(method == "rf") {
        train(Status ~ ., data = train_data, method = method, metric="ROC", trControl = ctrl_params, 
              preProcess = c("center", "scale", "nzv"), ntree = 100, tuneLength = 3)
      } else if(method == "naive_bayes") {
        train(Status ~ ., data = train_data, method = method, metric="ROC", trControl = ctrl_params,
              preProcess = c("nzv"), tuneLength = 3)
      } else {
        train(Status ~ ., data = train_data, method = method, metric="ROC", trControl = ctrl_params, 
              preProcess = c("center", "scale", "nzv"), tuneLength = 3)
      }
    }, error = function(e) { 
      print(paste("   Error training", method, ":", e$message))
      return(NULL) 
    })
    
    if (!is.null(fit)) {
      probs <- predict(fit, valid_data, type = "prob")
      classes <- predict(fit, valid_data)
      cm <- confusionMatrix(classes, valid_data$Status, mode = "everything")
      
      pass_col <- if("Pass" %in% colnames(probs)) "Pass" else if("pass" %in% colnames(probs)) "pass" else colnames(probs)[2]
      
      roc_obj <- tryCatch({
        roc(valid_data$Status, probs[, pass_col], levels = c("Fail", "Pass"), direction = "<", quiet=TRUE)
      }, error = function(e) return(NULL))
      
      if(!is.null(roc_obj)) {
        auc_val <- auc(roc_obj)
        
        # NOTE: Sinking here momentarily redirects output to the individual file
        # It will temporarily stop writing to the main log, write to the file, then return.
        indiv_file <- file.path(output_dir, paste0("Res_", method, ".txt"))
        sink(indiv_file)
        print(paste("MODEL:", method))
        print(paste("AUC:", round(auc_val, 4)))
        print(cm)
        sink() # Close individual file
        # Output returns to Master Log here (because of the outer sink)
        
        summary_df <- rbind(summary_df, data.frame(
          Model = method,
          AUC = round(as.numeric(auc_val), 4),
          Accuracy = round(cm$overall['Accuracy'], 4),
          Sensitivity = round(cm$byClass['Sensitivity'], 4),
          Specificity = round(cm$byClass['Specificity'], 4),
          stringsAsFactors = FALSE
        ))
        
        saveRDS(fit, file = file.path(output_dir, paste0("model_", method, ".rds")))
      }
    }
  }
  
  # Save Summary as TXT
  summary_df <- summary_df[order(-summary_df$AUC), ]
  write.table(summary_df, file.path(output_dir, "Summary_Leaderboard.txt"), 
              sep = "\t", row.names = FALSE, quote = FALSE)
  
  if(nrow(summary_df) > 0) {
    p <- ggplot(summary_df, aes(x = reorder(Model, AUC), y = AUC, fill = AUC)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = round(AUC, 3)), hjust = -0.2, size = 3) +
      coord_flip() +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(title = paste("Model Performance:", case_name), x = "Model", y = "AUC") +
      theme_minimal() + ylim(0, 1)
    ggsave(file.path(output_dir, "Model_Comparison.png"), plot = p, width = 10, height = 6)
  }
  
  return(summary_df)
}

# ==============================================================================
# 7. EXECUTION
# ==============================================================================

# Controls
ctrl_nobal <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                           summaryFunction = twoClassSummary, savePredictions = "final")

ctrl_bal   <- trainControl(method = "cv", number = 10, classProbs = TRUE, 
                           summaryFunction = twoClassSummary, savePredictions = "final",
                           sampling = "up") 

models_case1 <- c("glm", "knn", "svmRadial", "rpart", "rf", "gbm", "naive_bayes")
models_std   <- c("glm", "knn", "svmRadial", "rpart", "rf", "gbm")

# --- CASE 1: All Vars / No Balance ---
res_c1 <- run_experiment_case("Case 1 (All/NoBal)", dir_case1, 
                              train_full, valid_full, ctrl_nobal, models_case1)

# --- CASE 2: All Vars / Balanced ---
res_c2 <- run_experiment_case("Case 2 (All/Balanced)", dir_case2, 
                              train_full, valid_full, ctrl_bal, models_std)

# --- CASE 3: Selected Vars / No Balance ---
res_c3 <- run_experiment_case("Case 3 (Sel/NoBal)", dir_case3, 
                              train_sel, valid_sel, ctrl_nobal, models_std)

# --- CASE 4: Selected Vars / Balanced ---
res_c4 <- run_experiment_case("Case 4 (Sel/Balanced)", dir_case4, 
                              train_sel, valid_sel, ctrl_bal, models_std)

# ==============================================================================
# 8. FINAL COMPARISON
# ==============================================================================
print("--- GENERATING FINAL COMPARISON ---")

res_c1$Case <- "Case 1: All/NoBal"
res_c2$Case <- "Case 2: All/Balanced"
res_c3$Case <- "Case 3: Sel/NoBal"
res_c4$Case <- "Case 4: Sel/Balanced"

all_results <- bind_rows(res_c1, res_c2, res_c3, res_c4) %>%
  dplyr::select(Case, Model, AUC, Accuracy, Sensitivity, Specificity) %>%
  arrange(Case, desc(AUC))

print("=== OVERALL RESULTS ===")
print(all_results)

# Save Final Comparison as TXT
write.table(all_results, file.path(main_dir, "Comparison_All_4_Cases.txt"), 
            sep = "\t", row.names = FALSE, quote = FALSE)

if(nrow(all_results) > 0) {
  p_final <- ggplot(all_results, aes(x = Model, y = AUC, fill = Case)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    geom_text(aes(label = round(AUC, 3)), position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
    labs(title = "Performance Summary Across All 4 Experimental Cases", x = "Model", y = "AUC") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_brewer(palette = "Set2") +
    ylim(0, 1.05)
  
  ggsave(file.path(main_dir, "Final_Comparison_Plot.png"), plot = p_final, width = 12, height = 8)
}

print("=== PROCESS COMPLETE: Check folder 'Resultados_CódigoFinal' ===")

# ==============================================================================
# 9. CLOSE LOGGING
# ==============================================================================
# Close the message sink first, then the output sink
sink(type = "message") 
sink()
close(console_log)