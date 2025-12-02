# ==============================================================================
# STUDENT PERFORMANCE: COMPLETE EDA + FEATURE SELECTION + SAVING RESULTS (TXT)
# ==============================================================================

# 1. SETUP & LIBRARIES
# ------------------------------------------------------------------------------
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(gridExtra)) install.packages("gridExtra")
if (!require(randomForest)) install.packages("randomForest")
if (!require(Boruta)) install.packages("Boruta")

library(tidyverse)
library(caret)
library(gridExtra)
library(randomForest)
library(Boruta)

# --- CREATE RESULTS FOLDER ---
dir.create("Results", showWarnings = FALSE)

# --- START LOGGING TO FILE ---
sink("Results/analysis_log.txt", split = TRUE)

print("--- STARTING ANALYSIS ---")
print(paste("Date:", Sys.time()))

# 2. LOAD DATA
# ------------------------------------------------------------------------------
filename <- "Data/studentmat.csv" 
if (!file.exists(filename)) {
  sink()
  stop("Error: studentmat.csv file not found.")
}

df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)
print("Data Loaded Successfully.")

# 3. CLEANING & TRANSFORMATION
# ------------------------------------------------------------------------------
df_raw$G1 <- suppressWarnings(as.numeric(df_raw$G1))
df_raw$G2 <- suppressWarnings(as.numeric(df_raw$G2))
df_raw$G3 <- suppressWarnings(as.numeric(df_raw$G3))
df_raw$age <- as.numeric(df_raw$age)
df_raw$absences <- as.numeric(df_raw$absences)

vars_ordinales <- c("Medu", "Fedu", "traveltime", "studytime", "failures", 
                    "famrel", "freetime", "goout", "Dalc", "Walc", "health")
df_raw[vars_ordinales] <- lapply(df_raw[vars_ordinales], as.numeric)

df_clean <- df_raw %>%
  mutate(across(where(is.character), as.factor))

df_clean <- na.omit(df_clean)

# 4. PARTITION (75% TRAIN - 25% VALIDATION)
# ------------------------------------------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(df_clean$G3, p = 0.75, list = FALSE, times = 1)
train_set <- df_clean[ trainIndex,]
validation_set  <- df_clean[-trainIndex,]

print(paste("Train Set:", nrow(train_set), "rows | Validation Set:", nrow(validation_set), "rows"))

# ==============================================================================
# 5. NUMERIC EXPLORATION (DESCRIPTIVE STATS)
# ==============================================================================
print("--- 5. GENERATING DESCRIPTIVE STATISTICS ---")

stats_numericos <- train_set %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Valor") %>%
  group_by(Variable) %>%
  summarise(
    Mean = mean(Valor, na.rm = TRUE),
    Median = median(Valor, na.rm = TRUE),
    SD = sd(Valor, na.rm = TRUE),
    Min = min(Valor, na.rm = TRUE),
    Max = max(Valor, na.rm = TRUE)
  ) %>%
  mutate(across(where(is.numeric), round, 2))

print(as.data.frame(stats_numericos))

# --- SAVE AS TXT ---
write.table(stats_numericos, "Results/00_Descriptive_Stats.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)
print("Saved: Results/00_Descriptive_Stats.txt")

# ==============================================================================
# 6. VISUAL EXPLORATION (UNIVARIATE)
# ==============================================================================
print("--- 6. GENERATING DISTRIBUTION PLOTS ---")

plot_distribution_smart <- function(data, var_name) {
  col_data <- data[[var_name]]
  
  if(is.numeric(col_data)) {
    n_unique <- length(unique(col_data))
    
    p_box <- ggplot(data, aes(x = .data[[var_name]])) +
      geom_boxplot(fill = "#e74c3c", color = "#2c3e50", alpha = 0.6, outlier.colour = "red") +
      theme_void() + labs(title = paste("Dist:", var_name))
    
    if (n_unique < 15) {
      p_dist <- ggplot(data, aes(x = .data[[var_name]])) +
        geom_bar(fill = "#3498db", color = "white", alpha = 0.8) +
        theme_minimal() + scale_x_continuous(breaks = unique(sort(col_data)))
    } else {
      p_dist <- ggplot(data, aes(x = .data[[var_name]])) +
        geom_histogram(aes(y = ..density..), bins = 20, fill = "#3498db", color = "white", alpha = 0.7) +
        geom_density(color = "#2c3e50", size = 1) + theme_minimal()
    }
    grid.arrange(p_box, p_dist, ncol = 1, heights = c(1, 3))
    
  } else {
    p <- ggplot(data, aes(x = .data[[var_name]])) +
      geom_bar(fill = "#e67e22", color = "white", alpha = 0.8) +
      theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = paste("Freq:", var_name))
    print(p)
  }
}

all_vars <- names(train_set)
for (var in all_vars) {
  png(filename = paste0("Results/01_Dist_", var, ".png"), width = 600, height = 400)
  plot_distribution_smart(train_set, var)
  dev.off()
}
print("Saved univariate plots.")

# ==============================================================================
# 7. BIVARIATE EXPLORATION (RELATION VS G3)
# ==============================================================================
print("--- 7. GENERATING RELATIONAL PLOTS (VS G3) ---")

plot_vs_target_complete <- function(data, x_var_name, target_var = "G3") {
  col_data <- data[[x_var_name]]
  es_discreta <- is.numeric(col_data) && length(unique(col_data)) < 15
  
  if(is.numeric(col_data) && !es_discreta) {
    p <- ggplot(data, aes(x = .data[[x_var_name]], y = .data[[target_var]])) +
      geom_jitter(alpha = 0.5, width = 0.2, color = "#2980b9") +
      geom_smooth(method = "lm", se = FALSE, color = "#c0392b", linetype = "dashed") +
      theme_minimal() + labs(title = paste(x_var_name, "vs G3"), y = "G3")
  } else {
    p <- ggplot(data, aes(x = as.factor(.data[[x_var_name]]), y = .data[[target_var]], fill = as.factor(.data[[x_var_name]]))) +
      geom_jitter(color = "#34495e", width = 0.2, alpha = 0.5) +
      geom_boxplot(alpha = 0.6, outlier.shape = NA) +
      theme_minimal() + theme(legend.position = "none") +
      labs(title = paste(x_var_name, "vs G3"), x = x_var_name, y = "G3")
  }
  print(p)
}

for (var in all_vars) {
  if(var != "G3") {
    png(filename = paste0("Results/02_Rel_", var, "_vs_G3.png"), width = 600, height = 400)
    plot_vs_target_complete(train_set, var)
    dev.off()
  }
}
print("Saved bivariate plots.")

# ==============================================================================
# 8. FEATURE SELECTION
# ==============================================================================
print("--- 8. FEATURE SELECTION ANALYSIS ---")
train_fs <- train_set %>% select(-G1, -G2)

# --- METHOD 1: Pearson Correlation ---
print(">> [1/4] Calculating Correlations...")
cor_matrix <- cor(train_fs %>% select(where(is.numeric)))
cor_target <- sort(abs(cor_matrix[,"G3"]), decreasing = TRUE)
print(head(cor_target[-1], 10))

# --- SAVE AS TXT ---
cor_df <- data.frame(Variable=names(cor_target), Abs_Correlation=cor_target)
write.table(cor_df, "Results/03_Correlation_Ranking.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# --- METHOD 2: Random Forest Importance ---
print(">> [2/4] Random Forest Importance...")
set.seed(123)
rf_model <- randomForest(G3 ~ ., data = train_fs, importance = TRUE, ntree = 100)

png("Results/04_RF_Importance_Plot.png", width = 800, height = 600)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()

# --- METHOD 3: Boruta ---
print(">> [3/4] Running Boruta Algorithm...")
set.seed(123)
boruta_output <- Boruta(G3 ~ ., data = train_fs, doTrace = 0)
print(boruta_output)

png("Results/05_Boruta_Plot.png", width = 800, height = 600)
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Boruta Feature Importance")
dev.off()

# --- SAVE AS TXT ---
boruta_df <- attStats(boruta_output)
boruta_df$Variable <- rownames(boruta_df) # Make row names a real column
write.table(boruta_df, "Results/05_Boruta_Decisions.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# --- METHOD 4: RFE ---
print(">> [4/4] Running RFE...")
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

set.seed(123)
rfe_result <- rfe(train_fs[, -which(names(train_fs) == "G3")], 
                  train_fs$G3, 
                  sizes = c(1:15), 
                  rfeControl = ctrl)

print(rfe_result)

png("Results/06_RFE_Accuracy_Plot.png", width = 600, height = 400)
plot(rfe_result, type=c("g", "o"), main="RFE: Accuracy vs No. Variables")
dev.off()

print("--- ANALYSIS COMPLETE. ALL RESULTS SAVED IN 'Results/' FOLDER ---")
sink()