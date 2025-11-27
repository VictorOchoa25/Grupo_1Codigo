# ==============================================================================
# STUDENT PERFORMANCE: EDA CON PARTICIÓN 75/25
# ==============================================================================

# 1. CONFIGURACIÓN Y LIBRERÍAS
# ------------------------------------------------------------------------------
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret") 

library(tidyverse) 
library(caret) 

# 2. CARGAR DATOS (Con Verificación)
# ------------------------------------------------------------------------------
filename <- "student-mat.csv"

if (!file.exists(filename)) {
  stop(paste("ERROR: El archivo", filename, "no se encuentra.\n",
             "Tu directorio de trabajo actual es:", getwd()))
}

# Cargamos los datos completos
# stringsAsFactors = FALSE para manejar texto manualmente luego
df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)

print("--- Dimensión Original de los Datos ---")
print(dim(df_raw))

# 3. LIMPIEZA Y PREPROCESAMIENTO
# ------------------------------------------------------------------------------

# PASO A: Asegurar que las variables numéricas sean realmente numéricas
# Las notas (G1, G2, G3) a veces vienen como texto entre comillas en este dataset
df_raw$G1 <- suppressWarnings(as.numeric(df_raw$G1))
df_raw$G2 <- suppressWarnings(as.numeric(df_raw$G2))
df_raw$G3 <- suppressWarnings(as.numeric(df_raw$G3))
df_raw$age <- as.numeric(df_raw$age)
df_raw$absences <- as.numeric(df_raw$absences)

# PASO B: Convertir columnas de texto a factores (Categorías)
df_clean <- df_raw %>%
  mutate(across(where(is.character), as.factor))

# Eliminamos filas con NAs si algo falló en la conversión (opcional)
df_clean <- na.omit(df_clean)

# 4. PARTICIÓN DE DATOS (75% TRAIN - 25% VALIDATION)
# ------------------------------------------------------------------------------
set.seed(123) # Semilla para reproducibilidad

# Usamos createDataPartition de 'caret' para mantener la proporción de G3
trainIndex <- createDataPartition(df_clean$G3, p = 0.75, 
                                  list = FALSE, 
                                  times = 1)

train_set <- df_clean[ trainIndex,]
validation_set  <- df_clean[-trainIndex,]

print("--- RESULTADO DE LA PARTICIÓN (75/25) ---")
print(paste("Total Datos:", nrow(df_clean)))
print(paste("Set Entrenamiento (75%):", nrow(train_set), "filas"))
print(paste("Set Validación (25%):", nrow(validation_set), "filas"))

# ------------------------------------------------------------------------------
# A PARTIR DE AQUI, EL ANÁLISIS SE HACE SOLO CON 'train_set'
# ------------------------------------------------------------------------------

# 5. VISUALIZACIÓN (SOLO TRAIN)
# ------------------------------------------------------------------------------

# --- A. Variables Numéricas (Histogramas) ---
numeric_vars <- train_set %>% 
  dplyr::select(age, absences, G1, G2, G3) %>% 
  pivot_longer(everything(), names_to = "Variable", values_to = "Value")

plot_numeric <- ggplot(numeric_vars, aes(x = Value)) +
  geom_histogram(bins = 15, fill = "#2c3e50", color = "white") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribución de Variables Numéricas (Set Entrenamiento)",
       subtitle = "Basado en el 75% de los datos",
       x = "Valor", y = "Frecuencia")

print(plot_numeric)

# --- B. Variables Categóricas (Barras) ---
categorical_vars <- train_set %>% 
  dplyr::select(sex, school, address, Pstatus, Mjob, Fjob, internet, higher) %>% 
  pivot_longer(everything(), names_to = "Variable", values_to = "Category")

plot_categorical <- ggplot(categorical_vars, aes(x = Category)) +
  geom_bar(fill = "#e74c3c", color = "white") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribución de Variables Categóricas (Set Entrenamiento)",
       subtitle = "Conteos demográficos",
       x = "Categoría", y = "Cantidad")

print(plot_categorical)

# 6. ANÁLISIS DE CORRELACIÓN (SOLO TRAIN)
# ------------------------------------------------------------------------------
numeric_data <- train_set %>% dplyr::select(where(is.numeric))
cor_matrix <- cor(numeric_data, use = "complete.obs")

print("--- Matriz de Correlación (Top variables con G3 en Train) ---")
print(sort(cor_matrix[,"G3"], decreasing = TRUE))