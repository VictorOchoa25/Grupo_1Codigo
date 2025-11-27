# ==============================================================================
# STUDENT PERFORMANCE DATA EXPLORATION (Complete Script)
# ==============================================================================

# 1. SETUP & LIBRARIES
# ------------------------------------------------------------------------------
# Check if tidyverse is installed; if not, install it.
if (!require(tidyverse)) install.packages("tidyverse")

library(tidyverse) # For data manipulation (dplyr) and plotting (ggplot2)

# 2. LOAD DATA
# ------------------------------------------------------------------------------
# We use sep = ";" because the file uses semicolons as delimiters.
# stringsAsFactors = FALSE ensures text is read as raw characters initially.
df <- read.csv("student-mat.csv", sep = ";", stringsAsFactors = FALSE)

print("--- Initial Data Preview ---")
glimpse(df)

# 3. DATA CLEANING & TYPE CONVERSION
# ------------------------------------------------------------------------------

# STEP A: Ensure Grades are Numeric
# Sometimes G1/G2 are quoted in the CSV (e.g., "5"). We force them to be numeric.
# suppressWarnings is used in case 'NA's are introduced by non-numeric values.
df$G1 <- suppressWarnings(as.numeric(df$G1))
df$G2 <- suppressWarnings(as.numeric(df$G2))
df$G3 <- suppressWarnings(as.numeric(df$G3))
df$age <- as.numeric(df$age)
df$absences <- as.numeric(df$absences)

# STEP B: Convert Categorical Strings to Factors
# Columns that are still 'character' (like school, sex, job) are converted to factors.
df <- df %>%
  mutate(across(where(is.character), as.factor))

print("--- Structure After Cleaning ---")
str(df)

print("--- Summary Statistics ---")
summary(df)

# 4. VISUALIZATION
# ------------------------------------------------------------------------------

# --- A. Numeric Variables (Histograms) ---
# We focus on the key numeric variables: Age, Absences, and Grades.
numeric_vars <- df %>% 
  select(age, absences, G1, G2, G3) %>% 
  pivot_longer(everything(), names_to = "Variable", values_to = "Value")

plot_numeric <- ggplot(numeric_vars, aes(x = Value)) +
  geom_histogram(bins = 15, fill = "#2c3e50", color = "white") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Numeric Variables",
       subtitle = "Histograms for Age, Absences, and Grades (G1, G2, G3)",
       x = "Value", y = "Frequency")

print(plot_numeric)

# --- B. Categorical Variables (Bar Charts) ---
# We select a subset of interesting categorical variables to plot.
categorical_vars <- df %>% 
  select(sex, school, address, Pstatus, Mjob, Fjob, internet, higher) %>% 
  pivot_longer(everything(), names_to = "Variable", values_to = "Category")

plot_categorical <- ggplot(categorical_vars, aes(x = Category)) +
  geom_bar(fill = "#e74c3c", color = "white") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # Tilt text for readability
  labs(title = "Distribution of Categorical Variables",
       subtitle = "Counts for Demographics and Background",
       x = "Category", y = "Count")

print(plot_categorical)

# 5. CORRELATION ANALYSIS
# ------------------------------------------------------------------------------
# Calculate correlation matrix for numeric columns only
# use = "complete.obs" ignores rows with missing values
numeric_data <- df %>% select(where(is.numeric))
cor_matrix <- cor(numeric_data, use = "complete.obs")

print("--- Correlation Matrix (Key Variables) ---")
print(round(cor_matrix, 2))

print("--- Correlation with Final Grade (G3) ---")
# Sort variables by how strongly they correlate with the final grade
print(sort(cor_matrix[,"G3"], decreasing = TRUE))