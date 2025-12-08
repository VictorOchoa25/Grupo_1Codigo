# ==============================================================================
# SCRIPT 2: FEATURE SELECTION PROCESS
# ==============================================================================
# Objective: Apply algorithms to select optimal variables for Classification
# Output: Folder '02_Selection_Results' (Rankings + Decision Tables)

# 1. SETUP
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(Boruta)) install.packages("Boruta")
if (!require(randomForest)) install.packages("randomForest")

library(tidyverse); library(caret); library(Boruta); library(randomForest)

dir.create("02_Selection_Results", showWarnings = FALSE)
sink("02_Selection_Results/00_Selection_Log.txt", split = TRUE)
print("--- STARTING FEATURE SELECTION ---")

# 2. DATA PREP
df_raw <- read.csv("Data/studentmat.csv", sep = ";", stringsAsFactors = FALSE)
df_raw$age <- as.numeric(df_raw$age); df_raw$absences <- as.numeric(df_raw$absences)
df_raw$G3 <- as.numeric(df_raw$G3)

# Target: Status
df_sel <- df_raw %>%
  mutate(Status = ifelse(G3 >= 10, "Pass", "Fail")) %>%
  mutate(Status = as.factor(Status)) %>%
  select(-G1, -G2, -G3) %>%
  mutate(across(where(is.character), as.factor))

set.seed(123)
# Note: Feature selection is usually done on Train set to avoid bias, 
# but here we use full data for exploration per your request structure.
# Ideally, split here if rigor is required.

# 3. METHOD A: CHI-SQUARED (Categorical vs Categorical)
# ------------------------------------------------------------------------------
print(">> Running Chi-Squared Tests...")
cat_vars <- names(df_sel)[sapply(df_sel, is.factor)]
chi_df <- data.frame(Variable=character(), P_Value=numeric())

for (v in cat_vars) {
  if (v != "Status") {
    test <- chisq.test(table(df_sel[[v]], df_sel$Status))
    chi_df <- rbind(chi_df, data.frame(Variable=v, P_Value=test$p.value))
  }
}
chi_df <- chi_df[order(chi_df$P_Value),]
write.table(chi_df, "02_Selection_Results/01_ChiSquared_Ranking.txt", sep="\t", row.names=FALSE, quote=FALSE)
print(head(chi_df))

# 4. METHOD B: BORUTA (All-Relevant)
# ------------------------------------------------------------------------------
print(">> Running Boruta Algorithm...")
set.seed(123)
boruta_out <- Boruta(Status ~ ., data = df_sel, doTrace = 2)

png("02_Selection_Results/02_Boruta_Plot.png", width=800, height=600)
plot(boruta_out, cex.axis=.7, las=2, xlab="", main="Boruta Importance")
dev.off()

boruta_decisions <- attStats(boruta_out)
boruta_decisions$Variable <- rownames(boruta_decisions)
write.table(boruta_decisions, "02_Selection_Results/02_Boruta_Decisions.txt", sep="\t", row.names=FALSE, quote=FALSE)

# 5. METHOD C: RFE (Recursive Feature Elimination)
# ------------------------------------------------------------------------------
print(">> Running RFE Algorithm (Random Forest Wrapper)...")
ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
set.seed(123)
rfe_out <- rfe(df_sel[, names(df_sel) != "Status"], 
               df_sel$Status, sizes = c(1:15), rfeControl = ctrl_rfe)

png("02_Selection_Results/03_RFE_Accuracy.png")
plot(rfe_out, type=c("g", "o"))
dev.off()

print(rfe_out)
print(paste("Optimal Variables:", paste(predictors(rfe_out), collapse=", ")))

print("--- SELECTION COMPLETE. RESULTS SAVED IN '02_Selection_Results' ---")
sink()