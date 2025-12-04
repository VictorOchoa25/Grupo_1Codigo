# ==============================================================================
# SCRIPT 1: ADVANCED EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
# Objective: Visuals AND Descriptive Metrics (Univariate & Bivariate)
# Output: Folder '01_EDA_Results' containing .png plots and .txt tables

# 1. SETUP & LIBRARIES
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(gridExtra)) install.packages("gridExtra")
if (!require(e1071)) install.packages("e1071") # For skewness

library(tidyverse)
library(gridExtra)
library(e1071)

# Create Output Folder
dir.create("01_EDA_Results", showWarnings = FALSE)

print("--- STARTING ADVANCED EDA ---")

# 2. DATA LOADING & PREPARATION
filename <- "Data/studentmat.csv" 
if (!file.exists(filename)) filename <- "studentmat.csv"
if (!file.exists(filename)) stop("Error: studentmat.csv not found.")

df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)

# Clean and Convert
df_raw$age <- as.numeric(df_raw$age)
df_raw$absences <- as.numeric(df_raw$absences)
df_raw$G3 <- as.numeric(df_raw$G3)
df_raw$Medu <- as.numeric(df_raw$Medu)
df_raw$Fedu <- as.numeric(df_raw$Fedu)
df_raw$traveltime <- as.numeric(df_raw$traveltime)
df_raw$studytime <- as.numeric(df_raw$studytime)
df_raw$failures <- as.numeric(df_raw$failures)
df_raw$famrel <- as.numeric(df_raw$famrel)
df_raw$freetime <- as.numeric(df_raw$freetime)
df_raw$goout <- as.numeric(df_raw$goout)
df_raw$Dalc <- as.numeric(df_raw$Dalc)
df_raw$Walc <- as.numeric(df_raw$Walc)
df_raw$health <- as.numeric(df_raw$health)

# Create Target 'Status' (Pass/Fail)
df_eda <- df_raw %>%
  mutate(Status = ifelse(G3 >= 10, "Pass", "Fail")) %>%
  mutate(Status = as.factor(Status)) %>%
  select(-G1, -G2, -G3) # Remove grades

print("Data Loaded.")

# ==============================================================================
# 3. UNIVARIATE ANALYSIS (PLOTS + METRICS)
# ==============================================================================
print("--- PROCESSING UNIVARIATE ANALYSIS ---")

# --- A. NUMERIC VARIABLES ---
numeric_vars <- names(df_eda)[sapply(df_eda, is.numeric)]

# 1. Calculate Statistics
stats_num <- df_eda %>%
  select(all_of(numeric_vars)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    Mean = mean(Value, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    Std_Dev = sd(Value, na.rm = TRUE),
    Min = min(Value, na.rm = TRUE),
    Max = max(Value, na.rm = TRUE),
    Skewness = skewness(Value, na.rm = TRUE)
  ) %>%
  mutate(across(where(is.numeric), round, 2))

# Save Table
write.table(stats_num, "01_EDA_Results/Univariate_Numeric_Stats.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# 2. Save Plots
for (v in numeric_vars) {
  p <- ggplot(df_eda, aes(x = .data[[v]])) +
    geom_histogram(fill = "steelblue", color = "white", bins = 15) +
    geom_vline(aes(xintercept = mean(.data[[v]])), color="red", linetype="dashed", size=1) +
    labs(title = paste("Distribution:", v), subtitle = "Red line = Mean") + theme_minimal()
  
  png(paste0("01_EDA_Results/Uni_Num_", v, ".png"), width = 600, height = 400)
  print(p)
  dev.off()
}

# --- B. CATEGORICAL VARIABLES ---
cat_vars <- names(df_eda)[sapply(df_eda, is.factor)]
cat_vars <- cat_vars[cat_vars != "Status"]

# 1. Calculate Statistics (Frequency Tables)
sink("01_EDA_Results/Univariate_Categorical_Stats.txt")
for (v in cat_vars) {
  print(paste("=== VARIABLE:", v, "==="))
  tbl <- table(df_eda[[v]])
  prop <- round(prop.table(tbl) * 100, 2)
  res <- cbind(Count = tbl, Percentage = prop)
  print(res)
  print("")
}
sink()

# 2. Save Plots
for (v in cat_vars) {
  p <- ggplot(df_eda, aes(x = .data[[v]])) +
    geom_bar(fill = "coral", color = "white", alpha=0.8) +
    geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
    labs(title = paste("Frequency:", v)) + theme_minimal()
  
  png(paste0("01_EDA_Results/Uni_Cat_", v, ".png"), width = 600, height = 400)
  print(p)
  dev.off()
}

# ==============================================================================
# 4. BIVARIATE ANALYSIS (RELATIONSHIPS WITH STATUS)
# ==============================================================================
print("--- PROCESSING BIVARIATE ANALYSIS ---")

# --- A. NUMERIC VS STATUS ---

# 1. Calculate Comparative Stats (Mean by Group)
stats_bi_num <- df_eda %>%
  select(Status, all_of(numeric_vars)) %>%
  pivot_longer(-Status, names_to = "Variable", values_to = "Value") %>%
  group_by(Variable, Status) %>%
  summarise(Mean = mean(Value, na.rm=TRUE), Median = median(Value, na.rm=TRUE)) %>%
  pivot_wider(names_from = Status, values_from = c(Mean, Median)) %>%
  mutate(Diff_Mean = abs(Mean_Pass - Mean_Fail)) %>%
  mutate(across(where(is.numeric), round, 2)) %>%
  arrange(desc(Diff_Mean)) # Sort by biggest difference

# Save Table
write.table(stats_bi_num, "01_EDA_Results/Bivariate_Numeric_Stats.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# 2. Save Plots
for (v in numeric_vars) {
  p <- ggplot(df_eda, aes(x = Status, y = .data[[v]], fill = Status)) +
    geom_boxplot(alpha = 0.7) +
    stat_summary(fun = mean, geom = "point", shape = 20, size = 3, color = "white") +
    labs(title = paste(v, "by Status"), subtitle = "White dot = Mean") + theme_minimal()
  
  png(paste0("01_EDA_Results/Bi_Num_", v, ".png"), width = 600, height = 400)
  print(p)
  dev.off()
}

# --- B. CATEGORICAL VS STATUS ---

# 1. Calculate Cross-Tabs & Chi-Square
sink("01_EDA_Results/Bivariate_Categorical_Stats.txt")
print("CROSSTABS AND FAILURE RATES")
print("---------------------------")

chi_summary <- data.frame(Variable=character(), P_Value=numeric())

for (v in cat_vars) {
  print(paste(">>> VARIABLE:", v))
  
  # Contingency Table
  tbl <- table(df_eda[[v]], df_eda$Status)
  print("Counts:")
  print(tbl)
  
  # Row Percentages (This shows Failure Rate per group)
  props <- round(prop.table(tbl, 1) * 100, 2)
  print("Row Percentages (Pass/Fail Rates):")
  print(props)
  
  # Chi-Squared Test
  test <- chisq.test(tbl)
  print(paste("Chi-Square P-Value:", round(test$p.value, 5)))
  print("--------------------------------------------------")
  
  # Save for summary
  chi_summary <- rbind(chi_summary, data.frame(Variable=v, P_Value=test$p.value))
}
sink()

# Save just the ranking of P-values to a clean table
chi_summary <- chi_summary[order(chi_summary$P_Value), ]
write.table(chi_summary, "01_EDA_Results/Bivariate_ChiSquare_Summary.txt", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# 2. Save Plots (Stacked Bar)
for (v in cat_vars) {
  p <- ggplot(df_eda, aes(x = .data[[v]], fill = Status)) +
    geom_bar(position = "fill") +
    geom_hline(yintercept = 0.5, linetype="dashed", color="gray") +
    labs(title = paste(v, "Impact on Pass/Fail"), y = "Proportion") +
    scale_fill_brewer(palette = "Set1") + theme_minimal()
  
  png(paste0("01_EDA_Results/Bi_Cat_", v, ".png"), width = 600, height = 400)
  print(p)
  dev.off()
}

print("--- EDA COMPLETE. CHECK '01_EDA_Results' FOLDER ---")