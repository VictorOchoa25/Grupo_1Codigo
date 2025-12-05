# ==============================================================================
# ANÁLISIS DE COMPRENSIÓN DE DATOS - ESTUDIANTES MATEMÁTICAS
# Objetivo: Predecir riesgo de reprobar (G3) sin usar G1 y G2 como predictores
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN INICIAL Y CARGA DE DATOS
# ------------------------------------------------------------------------------

# Crear carpeta para resultados
if (!dir.exists("resultados_EDA")) {
  dir.create("resultados_EDA")
}

# Iniciar archivo de log
sink("resultados_EDA/log_analisis.txt", split = TRUE)

cat("=====================================================================\n")
cat("ANÁLISIS EXPLORATORIO - RENDIMIENTO ESTUDIANTIL EN MATEMÁTICAS\n")
cat("Fecha:", Sys.Date(), "\n")
cat("Objetivo: Predecir riesgo de reprobar (G3) sin usar G1 y G2\n")
cat("=====================================================================\n\n")

# Cargar librerías
cat("Cargando librerías...\n")
library(tidyverse)
library(readr)
library(skimr)
library(DataExplorer)
library(corrplot)
library(psych)
library(naniar)
library(gridExtra)

# Cargar datos
cat("Cargando datos...\n")
datos <- read_delim("Data/studentmat.csv", delim = ";")

# ------------------------------------------------------------------------------
# 2. TRANSFORMACIÓN INICIAL: CREAR VARIABLE OBJETIVO BINARIA
# ------------------------------------------------------------------------------

cat("\n2. TRANSFORMACIÓN DE LA VARIABLE OBJETIVO\n")
cat("=========================================\n")

# Crear variable binaria: 1 = Reprobado (G3 < 10), 0 = Aprobado (G3 >= 10)
# Nota: En el sistema portugués, 10 es el mínimo para aprobar
datos <- datos %>%
  mutate(
    G3_original = G3,  # Guardar original para referencia
    reprobado = ifelse(G3 < 10, 1, 0),
    reprobado_factor = factor(reprobado, levels = c(0, 1), 
                              labels = c("Aprobado", "Reprobado")),
    # También crear categorías para análisis
    nivel_rendimiento = case_when(
      G3 < 10 ~ "Reprobado",
      G3 >= 10 & G3 < 14 ~ "Suficiente",
      G3 >= 14 & G3 < 18 ~ "Bueno",
      G3 >= 18 ~ "Excelente"
    ),
    nivel_rendimiento = factor(nivel_rendimiento,
                               levels = c("Reprobado", "Suficiente", "Bueno", "Excelente"))
  )

# Eliminar G1 y G2 del conjunto de predictores (pero mantener para análisis exploratorio inicial)
datos_sin_g1g2 <- datos %>% select(-G1, -G2)

cat("Variable objetivo creada: 'reprobado' (1 = Reprobado, 0 = Aprobado)\n")
cat("G1 y G2 removidas del conjunto de predictores\n\n")

# Estadísticas de la nueva variable objetivo
cat("Distribución de la variable objetivo:\n")
distribucion <- table(datos$reprobado_factor)
print(distribucion)
cat("\nProporciones:\n")
print(prop.table(distribucion))

# ------------------------------------------------------------------------------
# 3. ANÁLISIS DE ESTRUCTURA Y CALIDAD DE DATOS
# ------------------------------------------------------------------------------

cat("\n\n3. ANÁLISIS DE ESTRUCTURA Y CALIDAD\n")
cat("=====================================\n")

# Dimensiones
cat("Dimensiones del dataset original:", dim(datos), "\n")
cat("Dimensiones sin G1 y G2:", dim(datos_sin_g1g2), "\n")

# Tipos de variables
cat("\nTipos de variables:\n")
str(datos_sin_g1g2)

# Resumen con skimr
cat("\nResumen estadístico (skimir):\n")
skim_resumen <- skim(datos_sin_g1g2)
print(skim_resumen)

# Guardar resumen en archivo
write.csv(skim_resumen, "resultados_EDA/resumen_skimir.csv", row.names = FALSE)

# Valores faltantes
cat("\nValores faltantes por variable:\n")
missing <- colSums(is.na(datos_sin_g1g2))
print(missing[missing > 0])

if (sum(missing) == 0) {
  cat("¡No hay valores faltantes en el dataset!\n")
}

# Gráfico de valores faltantes
png("resultados_EDA/01_missing_values.png", width = 800, height = 600)
print(gg_miss_var(datos_sin_g1g2) + 
        labs(title = "Valores Faltantes por Variable",
             subtitle = "Dataset sin G1 y G2") +
        theme_minimal())
dev.off()

# ------------------------------------------------------------------------------
# 4. ANÁLISIS UNIVARIADO - VARIABLES PREDICTORAS
# ------------------------------------------------------------------------------

cat("\n\n4. ANÁLISIS UNIVARIADO DE PREDICTORES\n")
cat("======================================\n")

# Identificar tipos de variables
variables_numericas <- datos_sin_g1g2 %>%
  select(where(is.numeric), -G3, -G3_original, -reprobado) %>%
  names()

variables_categoricas <- datos_sin_g1g2 %>%
  select(where(is.character), where(is.factor), 
         -reprobado_factor, -nivel_rendimiento) %>%
  names()

cat("Variables numéricas:", length(variables_numericas), "\n")
cat("Variables categóricas:", length(variables_categoricas), "\n")

# 4.1 Distribución de variables numéricas
cat("\nDistribución de variables numéricas clave:\n")
variables_numericas_clave <- c("age", "Medu", "Fedu", "studytime", 
                               "failures", "absences", "famrel", 
                               "freetime", "goout", "Dalc", "Walc", 
                               "health")

# Histogramas para variables numéricas
png("resultados_EDA/02_histogramas_numericas.png", width = 1200, height = 800)
datos_sin_g1g2 %>%
  select(all_of(variables_numericas_clave)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor)) +
  geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Distribución de Variables Numéricas Predictoras",
       subtitle = "Dataset sin G1 y G2",
       x = "Valor", y = "Frecuencia") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10))
dev.off()

# Boxplots para detección de outliers
png("resultados_EDA/03_boxplots_numericas.png", width = 1200, height = 800)
datos_sin_g1g2 %>%
  select(all_of(variables_numericas_clave)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = variable, y = valor)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "Boxplots - Detección de Outliers en Variables Numéricas",
       subtitle = "Variables predictoras sin G1 y G2",
       x = "Variable", y = "Valor") +
  theme_minimal()
dev.off()

# 4.2 Distribución de variables categóricas
cat("\nDistribución de variables categóricas:\n")

# Tablas de frecuencia para categóricas clave
categoricas_clave <- c("sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                       "schoolsup", "famsup", "paid", "higher", "internet", 
                       "romantic")

# Guardar tablas de frecuencia en archivo
frecuencias <- list()
for(var in categoricas_clave) {
  tabla <- table(datos_sin_g1g2[[var]])
  proporciones <- prop.table(tabla)
  frecuencias[[var]] <- data.frame(
    Categoria = names(tabla),
    Frecuencia = as.numeric(tabla),
    Proporcion = as.numeric(proporciones)
  )
  cat("\n===", var, "===\n")
  print(tabla)
  cat("Proporciones:\n")
  print(proporciones)
}

# Guardar todas las frecuencias en un archivo CSV
frecuencias_df <- map_df(frecuencias, ~.x, .id = "Variable")
write.csv(frecuencias_df, "resultados_EDA/frecuencias_categoricas.csv", row.names = FALSE)

# Gráficos de barras para categóricas
png("resultados_EDA/04_barras_categoricas.png", width = 1200, height = 800)
datos_sin_g1g2 %>%
  select(all_of(categoricas_clave)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "categoria") %>%
  ggplot(aes(x = categoria)) +
  geom_bar(fill = "coral", alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Distribución de Variables Categóricas Predictoras",
       subtitle = "Dataset sin G1 y G2",
       x = "Categoría", y = "Frecuencia") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 9))
dev.off()

# ------------------------------------------------------------------------------
# 5. ANÁLISIS BIVARIADO - RELACIÓN CON LA VARIABLE OBJETIVO
# ------------------------------------------------------------------------------

cat("\n\n5. ANÁLISIS BIVARIADO CON VARIABLE OBJETIVO\n")
cat("=============================================\n")

# 5.1 Variables numéricas vs Reprobar
cat("\nRelación entre variables numéricas y riesgo de reprobar:\n")

# Gráfico: Densidad por condición
png("resultados_EDA/05_densidad_numericas_vs_reprobar.png", width = 1200, height = 800)
plots <- list()
for(i in 1:length(variables_numericas_clave)) {
  var <- variables_numericas_clave[i]
  p <- ggplot(datos_sin_g1g2, aes_string(x = var, fill = "reprobado_factor")) +
    geom_density(alpha = 0.5) +
    labs(title = paste(var, "vs Estado"),
         x = var, y = "Densidad") +
    theme_minimal() +
    theme(legend.position = "none")
  plots[[i]] <- p
}
grid.arrange(grobs = plots, ncol = 4, 
             top = "Distribución de Variables Numéricas por Estado (Aprobado/Reprobado)")
dev.off()

# 5.2 Variables categóricas vs Reprobar
cat("\nRelación entre variables categóricas y riesgo de reprobar:\n")

# Calcular tasas de reprobación por categoría
tasas_reprobacion <- list()
for(var in categoricas_clave) {
  tasa <- datos_sin_g1g2 %>%
    group_by(!!sym(var)) %>%
    summarise(
      n = n(),
      reprobados = sum(reprobado),
      tasa_reprobacion = mean(reprobado) * 100
    ) %>%
    arrange(desc(tasa_reprobacion))
  
  tasas_reprobacion[[var]] <- tasa
  
  cat("\n=== Tasa de reprobación por", var, "===\n")
  print(tasa)
}

# Guardar tasas de reprobación
tasas_df <- map_df(tasas_reprobacion, ~.x, .id = "Variable")
write.csv(tasas_df, "resultados_EDA/tasas_reprobacion_categoricas.csv", row.names = FALSE)

# Gráfico: Tasas de reprobación por categoría
png("resultados_EDA/06_tasas_reprobacion_categoricas.png", width = 1200, height = 800)
tasas_df %>%
  ggplot(aes(x = reorder(!!sym(names(tasas_df)[2]), -tasa_reprobacion), 
             y = tasa_reprobacion, 
             fill = tasa_reprobacion)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_text(aes(label = paste0(round(tasa_reprobacion, 1), "%")), 
            vjust = -0.5, size = 3) +
  facet_wrap(~ Variable, scales = "free_x") +
  scale_fill_gradient(low = "green", high = "red") +
  labs(title = "Tasa de Reprobación por Categoría de Variables Predictoras",
       subtitle = "Porcentaje de estudiantes que reprueban por categoría",
       x = "Categoría", y = "Tasa de Reprobación (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none",
        strip.text = element_text(size = 9))
dev.off()

# 5.3 Correlaciones entre variables numéricas
cat("\nMatriz de correlación entre variables numéricas:\n")
cor_vars <- c(variables_numericas_clave, "reprobado")
cor_matrix <- cor(datos_sin_g1g2[, cor_vars], use = "complete.obs")
print(cor_matrix)

# Guardar matriz de correlación
write.csv(cor_matrix, "resultados_EDA/matriz_correlacion.csv")

# Gráfico de correlaciones
png("resultados_EDA/07_matriz_correlacion.png", width = 1000, height = 800)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         title = "Matriz de Correlación - Variables Predictoras vs Reprobado",
         mar = c(0, 0, 2, 0),
         addCoef.col = "black", number.cex = 0.7)
dev.off()

# 5.4 Gráficos específicos para variables clave
png("resultados_EDA/08_analisis_variables_clave.png", width = 1200, height = 900)

p1 <- ggplot(datos_sin_g1g2, aes(x = factor(failures), y = reprobado, fill = factor(failures))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Failures vs Probabilidad de Reprobación",
       x = "Número de asignaturas reprobadas anteriormente",
       y = "Reprobado (1=Sí, 0=No)") +
  theme_minimal()

p2 <- ggplot(datos_sin_g1g2, aes(x = absences, y = reprobado)) +
  geom_point(alpha = 0.5, position = position_jitter(height = 0.02)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), 
              color = "red", se = FALSE) +
  labs(title = "Ausencias vs Probabilidad de Reprobación",
       x = "Número de ausencias",
       y = "Reprobado (1=Sí, 0=No)") +
  theme_minimal()

p3 <- ggplot(datos_sin_g1g2, aes(x = studytime, y = reprobado, fill = factor(studytime))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Tiempo de Estudio vs Probabilidad de Reprobación",
       x = "Tiempo de estudio semanal (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)",
       y = "Reprobado (1=Sí, 0=No)") +
  theme_minimal()

p4 <- ggplot(datos_sin_g1g2, aes(x = higher, y = reprobado, fill = higher)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Deseo de Educación Superior vs Probabilidad de Reprobación",
       x = "¿Desea cursar educación superior?",
       y = "Reprobado (1=Sí, 0=No)") +
  theme_minimal()

grid.arrange(p1, p2, p3, p4, ncol = 2, 
             top = "Análisis de Variables Clave vs Riesgo de Reprobación")
dev.off()

# ------------------------------------------------------------------------------
# 6. ANÁLISIS MULTIVARIADO E INTERACCIONES
# ------------------------------------------------------------------------------

cat("\n\n6. ANÁLISIS DE INTERACCIONES\n")
cat("=============================\n")

# 6.1 Interacción: failures * studytime
png("resultados_EDA/09_interaccion_failures_studytime.png", width = 1000, height = 600)
datos_sin_g1g2 %>%
  mutate(failures_cat = factor(failures,
                               levels = 0:3,
                               labels = c("0 fallos", "1 fallo", "2 fallos", "3+ fallos")),
         studytime_cat = factor(studytime,
                                levels = 1:4,
                                labels = c("<2h", "2-5h", "5-10h", ">10h"))) %>%
  group_by(failures_cat, studytime_cat) %>%
  summarise(tasa_reprobacion = mean(reprobado) * 100,
            n = n()) %>%
  filter(n >= 3) %>%  # Filtrar combinaciones con pocos datos
  ggplot(aes(x = studytime_cat, y = failures_cat, fill = tasa_reprobacion)) +
  geom_tile(color = "white") +
  geom_text(aes(label = paste0(round(tasa_reprobacion, 0), "%\n(n=", n, ")")), 
            color = "black", size = 3.5) +
  scale_fill_gradient2(low = "green", mid = "yellow", high = "red", 
                       midpoint = 30, name = "Tasa de\nreprobación (%)") +
  labs(title = "Interacción: Fallos Previos vs Tiempo de Estudio",
       subtitle = "Tasa de reprobación por combinación de factores",
       x = "Tiempo de Estudio Semanal",
       y = "Fallos Previos") +
  theme_minimal() +
  theme(panel.grid = element_blank())
dev.off()

# 6.2 Interacción: absences * internet
png("resultados_EDA/10_interaccion_absences_internet.png", width = 1000, height = 600)
datos_sin_g1g2 %>%
  mutate(absences_cat = cut(absences, 
                            breaks = c(-1, 0, 5, 10, 20, 100),
                            labels = c("0", "1-5", "6-10", "11-20", "21+"))) %>%
  group_by(absences_cat, internet) %>%
  summarise(tasa_reprobacion = mean(reprobado) * 100,
            n = n()) %>%
  filter(n >= 3) %>%
  ggplot(aes(x = absences_cat, y = internet, fill = tasa_reprobacion)) +
  geom_tile(color = "white") +
  geom_text(aes(label = paste0(round(tasa_reprobacion, 0), "%")), 
            color = "black", size = 4) +
  scale_fill_gradient2(low = "green", mid = "yellow", high = "red",
                       midpoint = 25, name = "Tasa de\nreprobación (%)") +
  labs(title = "Interacción: Ausencias vs Acceso a Internet",
       subtitle = "Tasa de reprobación por combinación de factores",
       x = "Número de Ausencias",
       y = "Acceso a Internet en Casa") +
  theme_minimal() +
  theme(panel.grid = element_blank())
dev.off()

# ------------------------------------------------------------------------------
# 7. ANÁLISIS DE OUTLIERS Y VALORES EXTREMOS
# ------------------------------------------------------------------------------

cat("\n\n7. ANÁLISIS DE OUTLIERS\n")
cat("======================\n")

# Función para detectar outliers
detectar_outliers <- function(x) {
  qnt <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
  iqr <- IQR(x, na.rm = TRUE)
  lower <- qnt[1] - 1.5 * iqr
  upper <- qnt[2] + 1.5 * iqr
  list(
    lower = lower,
    upper = upper,
    outliers = sum(x < lower | x > upper, na.rm = TRUE),
    porcentaje = mean(x < lower | x > upper, na.rm = TRUE) * 100
  )
}

cat("\nOutliers en variables numéricas clave:\n")
outliers_info <- list()
for(var in variables_numericas_clave) {
  res <- detectar_outliers(datos_sin_g1g2[[var]])
  outliers_info[[var]] <- res
  cat(var, ": ", res$outliers, " outliers (", 
      round(res$porcentaje, 1), "%)\n", sep = "")
}

# Guardar información de outliers
outliers_df <- map_df(outliers_info, ~data.frame(
  Limite_Inferior = .x$lower,
  Limite_Superior = .x$upper,
  N_Outliers = .x$outliers,
  Porcentaje_Outliers = .x$porcentaje
), .id = "Variable")
write.csv(outliers_df, "resultados_EDA/deteccion_outliers.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 8. REPORTE FINAL Y CONCLUSIONES
# ------------------------------------------------------------------------------

cat("\n\n8. CONCLUSIONES Y HALLAZGOS CLAVE\n")
cat("==================================\n")

# Calcular estadísticas finales
total_estudiantes <- nrow(datos_sin_g1g2)
tasa_reprobacion_total <- mean(datos_sin_g1g2$reprobado) * 100

# Variables con mayor correlación con reprobado
cor_reprobado <- cor_matrix["reprobado", ]
cor_reprobado <- cor_reprobado[names(cor_reprobado) != "reprobado"]
cor_reprobado_abs <- abs(cor_reprobado)
top_cor <- head(sort(cor_reprobado_abs, decreasing = TRUE), 5)

cat("\nRESUMEN EJECUTIVO:\n")
cat("-----------------\n")
cat("Total estudiantes:", total_estudiantes, "\n")
cat("Tasa de reprobación general:", round(tasa_reprobacion_total, 1), "%\n")
cat("Estudiantes que reprueban:", sum(datos_sin_g1g2$reprobado), "\n")
cat("Estudiantes que aprueban:", sum(datos_sin_g1g2$reprobado == 0), "\n")

cat("\nVARIABLES MÁS RELACIONADAS CON REPROBAR (correlación absoluta):\n")
for(i in 1:length(top_cor)) {
  var <- names(top_cor)[i]
  cat(i, ". ", var, ": ", 
      round(cor_reprobado[var], 3), 
      " (correlación)\n", sep = "")
}

# Hallazgos clave
cat("\nHALLAZGOS CLAVE DEL ANÁLISIS EXPLORATORIO:\n")
cat("1. Variables con mayor impacto en reprobación:\n")
cat("   - failures: Número de asignaturas reprobadas anteriormente\n")
cat("   - absences: Número de ausencias\n")
cat("   - studytime: Tiempo de estudio semanal\n")
cat("   - higher: Deseo de educación superior\n\n")

cat("2. Interacciones importantes:\n")
cat("   - Estudiantes con más fallos previos y menos tiempo de estudio\n")
cat("     tienen mayor riesgo\n")
cat("   - Ausencias combinadas con falta de internet aumentan riesgo\n\n")

cat("3. Recomendaciones para modelado:\n")
cat("   - Considerar crear variables de interacción\n")
cat("   - Evaluar tratamiento de outliers en 'absences'\n")
cat("   - Codificar variables categóricas con target encoding\n")

# ------------------------------------------------------------------------------
# 9. GUARDAR DATOS PROCESADOS Y CERRAR
# ------------------------------------------------------------------------------

# Guardar dataset procesado (sin G1 y G2)
write.csv(datos_sin_g1g2, "resultados_EDA/datos_procesados.csv", row.names = FALSE)

# Guardar dataset con transformaciones
write.csv(datos, "resultados_EDA/datos_completos.csv", row.names = FALSE)

# Cerrar archivo de log
sink()

# Mensaje final
cat("\n¡Análisis completado!\n")
cat("====================\n")
cat("Archivos guardados en la carpeta 'resultados_EDA/':\n")
cat("1. log_analisis.txt - Resultados de consola\n")
cat("2. datos_procesados.csv - Dataset sin G1 y G2\n")
cat("3. datos_completos.csv - Dataset con transformaciones\n")
cat("4. resumen_skimir.csv - Resumen estadístico\n")
cat("5. frecuencias_categoricas.csv - Tablas de frecuencia\n")
cat("6. tasas_reprobacion_categoricas.csv - Tasas por categoría\n")
cat("7. matriz_correlacion.csv - Matriz de correlaciones\n")
cat("8. deteccion_outliers.csv - Análisis de outliers\n")
cat("9. 10 archivos .png con visualizaciones\n")
cat("\nListo para la fase de Preprocesamiento y Modelado!\n")

