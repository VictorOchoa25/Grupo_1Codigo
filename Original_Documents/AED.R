# ==============================================================================
# STUDENT PERFORMANCE: EDA COMPLETO (INTELIGENTE)
# ==============================================================================

# 1. LIBRERÍAS
# ------------------------------------------------------------------------------
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret") 
if (!require(gridExtra)) install.packages("gridExtra") 

library(tidyverse) 
library(caret) 
library(gridExtra)

# 2. CARGAR DATOS
# ------------------------------------------------------------------------------
filename <- "Data/studentmat.csv"
if (!file.exists(filename)) stop("No se encuentra el archivo studentmat.csv")

df_raw <- read.csv(filename, sep = ";", stringsAsFactors = FALSE)

# 3. LIMPIEZA Y TRANSFORMACIÓN
# ------------------------------------------------------------------------------

# A. Limpieza de Notas
df_raw$G1 <- suppressWarnings(as.numeric(df_raw$G1))
df_raw$G2 <- suppressWarnings(as.numeric(df_raw$G2))
df_raw$G3 <- suppressWarnings(as.numeric(df_raw$G3))
df_raw$age <- as.numeric(df_raw$age)
df_raw$absences <- as.numeric(df_raw$absences)

# B. Variables Ordinales (Mantener como numéricas para ver tendencias)
vars_ordinales <- c("Medu", "Fedu", "traveltime", "studytime", "failures", 
                    "famrel", "freetime", "goout", "Dalc", "Walc", "health")

df_raw[vars_ordinales] <- lapply(df_raw[vars_ordinales], as.numeric)

# C. Variables Categóricas (Convertir a factor)
df_clean <- df_raw %>%
  mutate(across(where(is.character), as.factor))

df_clean <- na.omit(df_clean)

# 4. PARTICIÓN (75% TRAIN - 25% VALIDATION)
# ------------------------------------------------------------------------------
set.seed(123)
trainIndex <- createDataPartition(df_clean$G3, p = 0.75, list = FALSE, times = 1)
train_set <- df_clean[ trainIndex,]
validation_set  <- df_clean[-trainIndex,]

print(paste("Train Set:", nrow(train_set), "filas | Validation Set:", nrow(validation_set), "filas"))

# ==============================================================================
# 5. EXPLORACIÓN NUMÉRICA (ESTADÍSTICOS DESCRIPTIVOS)
# ==============================================================================
print("--- 5. ESTADÍSTICOS DESCRIPTIVOS (Solo Train Set) ---")

stats_numericos <- train_set %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Valor") %>%
  group_by(Variable) %>%
  summarise(
    Media = mean(Valor, na.rm = TRUE),
    Mediana = median(Valor, na.rm = TRUE),
    Desv_Std = sd(Valor, na.rm = TRUE),
    Min = min(Valor, na.rm = TRUE),
    Max = max(Valor, na.rm = TRUE),
    Rango = Max - Min
  ) %>%
  mutate(across(where(is.numeric), round, 2))

print("Tabla de Resumen Numérico:")
print(as.data.frame(stats_numericos))

# ==============================================================================
# 6. EXPLORACIÓN VISUAL UNIVARIADA
# ==============================================================================
print("--- 6. GENERANDO GRÁFICOS DE DISTRIBUCIÓN INDIVIDUAL ---")

# Función mejorada: Decide el mejor gráfico según la cantidad de valores únicos
plot_distribution_smart <- function(data, var_name) {
  
  col_data <- data[[var_name]]
  
  if(is.numeric(col_data)) {
    
    # Contamos valores únicos para decidir
    n_unique <- length(unique(col_data))
    
    # 1. Boxplot (Siempre útil para numéricas)
    p_box <- ggplot(data, aes(x = .data[[var_name]])) +
      geom_boxplot(fill = "#e74c3c", color = "#2c3e50", alpha = 0.6, outlier.colour = "red") +
      theme_void() + 
      labs(title = paste("Boxplot + Distribución:", var_name))
    
    # 2. DECISIÓN: ¿Discreta o Continua?
    if (n_unique < 15) {
      # CASO DISCRETO (Edad, Medu, Freetime): Usar BARRAS (No Histograma con densidad)
      p_dist <- ggplot(data, aes(x = .data[[var_name]])) +
        geom_bar(fill = "#3498db", color = "white", alpha = 0.8) +
        theme_minimal() +
        scale_x_continuous(breaks = unique(sort(col_data))) + # Ejes enteros
        labs(x = "Valor (Discreto)", y = "Conteo")
      
    } else {
      # CASO CONTINUO (Notas, Ausencias): Usar HISTOGRAMA + DENSIDAD
      p_dist <- ggplot(data, aes(x = .data[[var_name]])) +
        geom_histogram(aes(y = ..density..), bins = 20, fill = "#3498db", color = "white", alpha = 0.7) +
        geom_density(color = "#2c3e50", size = 1) +
        theme_minimal() +
        labs(x = "Valor (Continuo)", y = "Densidad")
    }
    
    # Combinamos: Boxplot arriba, Distribución abajo
    grid.arrange(p_box, p_dist, ncol = 1, heights = c(1, 3))
    
  } else {
    # CASO CATEGÓRICO (Factor): Gráfico de Barras
    p <- ggplot(data, aes(x = .data[[var_name]])) +
      geom_bar(fill = "#e67e22", color = "white", alpha = 0.8) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = paste("Frecuencia:", var_name), x = var_name, y = "Conteo")
    
    print(p)
  }
}

# Obtenemos TODOS los nombres de variables
all_vars <- names(train_set)

# Iteramos e imprimimos los gráficos uno por uno
for (var in all_vars) {
  plot_distribution_smart(train_set, var)
  Sys.sleep(0.1) 
}

# ==============================================================================
# 7. EXPLORACIÓN VISUAL BIVARIADA (RELACIÓN VS TARGET G3)
# ==============================================================================
print("--- 7. GENERANDO GRÁFICOS DE RELACIÓN VS TARGET (G3) ---")

plot_vs_target_complete <- function(data, x_var_name, target_var = "G3") {
  
  col_data <- data[[x_var_name]]
  
  # Detectamos si es una variable "Discreta" (poca variedad, ej. 1 al 5)
  es_discreta <- is.numeric(col_data) && length(unique(col_data)) < 15
  
  if(is.numeric(col_data) && !es_discreta) {
    # CASO CONTINUO PURO (ej. Absences, G1, G2) -> SCATTER PLOT
    p <- ggplot(data, aes(x = .data[[x_var_name]], y = .data[[target_var]])) +
      geom_jitter(alpha = 0.5, width = 0.2, color = "#2980b9") +
      geom_smooth(method = "lm", se = FALSE, color = "#c0392b", linetype = "dashed") +
      theme_minimal() +
      labs(title = paste(x_var_name, "vs G3 (Scatter)"), x = x_var_name, y = "G3")
    
  } else {
    # CASO CATEGÓRICO O DISCRETO (ej. Medu, Health, Sex) -> BOX PLOT + JITTER (ESTILO MODERNO)
    
    # Truco visual: geom_jitter primero (puntos detrás), geom_boxplot después (caja semitransparente)
    # outlier.shape = NA es CLAVE: para no dibujar el outlier dos veces (como punto y en la caja)
    
    p <- ggplot(data, aes(x = as.factor(.data[[x_var_name]]), y = .data[[target_var]], fill = as.factor(.data[[x_var_name]]))) +
      geom_jitter(color = "#34495e", width = 0.2, alpha = 0.5) +  # Puntos dispersos
      geom_boxplot(alpha = 0.6, outlier.shape = NA) +            # Caja encima
      theme_minimal() +
      theme(legend.position = "none") + # Quitamos leyenda redundante
      labs(title = paste(x_var_name, "vs G3 (Boxplot + Jitter)"), 
           x = x_var_name, y = "G3")
  }
  print(p)
}

# Iteramos sobre TODAS las variables
for (var in all_vars) {
  if(var != "G3") { 
    plot_vs_target_complete(train_set, var)
    Sys.sleep(0.1)
  }
}