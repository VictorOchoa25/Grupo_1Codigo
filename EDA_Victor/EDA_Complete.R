# ==============================================================================
# ANÁLISIS EXPLORATORIO COMPLETO - FLUJO INTEGRADO
# ==============================================================================

# ------------------------------------------------------------------------------
# PARTE 1: CARGA Y PREPARACIÓN DE DATOS
# ------------------------------------------------------------------------------
library(tidyverse)
library(readr)
library(skimr)
library(patchwork)
library(ggridges)
library(kableExtra)

# 1.1 Cargar datos originales
cat("=== CARGANDO DATOS ORIGINALES ===\n")
datos <- read_delim("Data/studentmat.csv", 
                    delim = ";", escape_double = FALSE, trim_ws = TRUE)
cat("Dataset original cargado:", nrow(datos), "filas,", ncol(datos), "columnas\n\n")

# 1.2 Transformación inicial (Data Understanding)
cat("=== TRANSFORMACIÓN INICIAL ===\n")
datos_procesados <- datos %>%
  mutate(
    # Variable objetivo binaria
    reprobado = ifelse(G3 < 10, 1, 0),
    reprobado_factor = factor(reprobado, 
                              levels = c(0, 1), 
                              labels = c("Aprobado", "Reprobado")),
    
    # Categorías para mejor visualización
    studytime_cat = factor(studytime,
                           levels = 1:4,
                           labels = c("<2h", "2-5h", "5-10h", ">10h")),
    
    failures_cat = case_when(
      failures == 0 ~ "Ninguno",
      failures == 1 ~ "1 fallo",
      failures == 2 ~ "2 fallos",
      failures >= 3 ~ "3+ fallos"
    ) %>% factor(levels = c("Ninguno", "1 fallo", "2 fallos", "3+ fallos")),
    
    absences_cat = cut(absences,
                       breaks = c(-1, 0, 5, 10, 20, max(absences)),
                       labels = c("0", "1-5", "6-10", "11-20", "21+"))
  ) %>%
  # Eliminar G1 y G2 según tu requerimiento
  select(-G1, -G2)

cat("Transformaciones aplicadas:\n")
cat("- Variable objetivo: reprobado (1=Reprobado, 0=Aprobado)\n")
cat("- Variables G1 y G2 eliminadas\n")
cat("- Categorías creadas para visualización\n\n")

# 1.3 Verificación básica
cat("=== VERIFICACIÓN DE DATOS ===\n")
cat("Dimensiones finales:", dim(datos_procesados), "\n")
cat("Tasa de reprobación:", round(mean(datos_procesados$reprobado) * 100, 1), "%\n")
cat("Valores faltantes:", sum(is.na(datos_procesados)), "\n\n")

# ------------------------------------------------------------------------------
# PARTE 2: ANÁLISIS EXPLORATORIO VISUAL
# ------------------------------------------------------------------------------
cat("=== INICIANDO ANÁLISIS EXPLORATORIO VISUAL ===\n")

# 2.1 Configurar tema para gráficos profesionales
tema_presentacion <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5,
                                margin = margin(b = 15)),
      plot.subtitle = element_text(size = 12, color = "gray40", hjust = 0.5,
                                   margin = margin(b = 20)),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold"),
      legend.position = "right",
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 20, 20, 20)
    )
}

# Crear carpeta para resultados
if (!dir.exists("resultados_EDA")) {
  dir.create("resultados_EDA")
  cat("Carpeta 'resultados_EDA' creada\n")
}

# ------------------------------------------------------------------------------
# GRÁFICO 1: DISTRIBUCIÓN DE RESULTADOS
# ------------------------------------------------------------------------------
cat("Generando Gráfico 1: Distribución de resultados...\n")

p1_distribucion <- datos_procesados %>%
  group_by(reprobado_factor) %>%
  summarise(n = n()) %>%
  mutate(porcentaje = n/sum(n)*100,
         etiqueta = paste0(round(porcentaje, 1), "%")) %>%
  ggplot(aes(x = reorder(reprobado_factor, -porcentaje), 
             y = porcentaje, 
             fill = reorder(reprobado_factor, -porcentaje))) +
  geom_col(width = 0.6, alpha = 0.9) +
  geom_text(aes(label = etiqueta), vjust = -0.5, 
            size = 5, fontface = "bold", color = "#2E2828") +
  scale_fill_manual(values = c("#2E86AB", "#C73E1D")) +
  scale_y_continuous(limits = c(0, 80)) +
  labs(title = "DISTRIBUCIÓN DE RESULTADOS ACADÉMICOS",
       subtitle = "Porcentaje de estudiantes que aprueban vs reprueban Matemáticas",
       x = "Resultado", y = "Porcentaje (%)",
       caption = paste("Tasa de reprobación:", 
                       round(mean(datos_procesados$reprobado)*100, 1), "%")) +
  tema_presentacion() +
  theme(legend.position = "none")

ggsave("resultados_EDA/01_distribucion_resultados.png", p1_distribucion,
       width = 10, height = 6, dpi = 300)

# ------------------------------------------------------------------------------
# GRÁFICO 2: MATRIZ DE RIESGO (HEATMAP)
# ------------------------------------------------------------------------------
cat("Generando Gráfico 2: Matriz de riesgo...\n")

p2_heatmap_riesgo <- datos_procesados %>%
  group_by(failures_cat, studytime_cat) %>%
  summarise(
    tasa_reprobacion = mean(reprobado) * 100,
    n_estudiantes = n(),
    .groups = "drop"
  ) %>%
  filter(n_estudiantes >= 3) %>%
  ggplot(aes(x = studytime_cat, y = failures_cat, 
             fill = tasa_reprobacion)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = paste0(round(tasa_reprobacion, 0), "%\n(n=", n_estudiantes, ")")),
            color = "white", size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "#1A9641", mid = "#FFFFBF", high = "#D7191C",
                       midpoint = 30, 
                       name = "Tasa de\nReprobación (%)",
                       limits = c(0, 100)) +
  labs(title = "MATRIZ DE RIESGO ACADÉMICO",
       subtitle = "Interacción entre Fallos Previos y Tiempo de Estudio",
       x = "Tiempo de Estudio Semanal", y = "Fallos Previos") +
  tema_presentacion() +
  theme(panel.grid = element_blank())

ggsave("resultados_EDA/02_matriz_riesgo.png", p2_heatmap_riesgo,
       width = 12, height = 8, dpi = 300)

# ------------------------------------------------------------------------------
# GRÁFICO 3: DISTRIBUCIÓN POR VARIABLES CLAVE (RIDGELINE)
# ------------------------------------------------------------------------------
cat("Generando Gráfico 3: Distribución por variables clave...\n")

p3_ridgeline <- datos_procesados %>%
  select(reprobado_factor, failures, absences, studytime, Medu, Fedu) %>%
  pivot_longer(cols = -reprobado_factor, 
               names_to = "variable", 
               values_to = "valor") %>%
  mutate(variable = case_when(
    variable == "failures" ~ "Fallos Previos",
    variable == "absences" ~ "Ausencias",
    variable == "studytime" ~ "Tiempo de Estudio",
    variable == "Medu" ~ "Educación Madre",
    variable == "Fedu" ~ "Educación Padre"
  )) %>%
  ggplot(aes(x = valor, y = variable, fill = reprobado_factor)) +
  geom_density_ridges(alpha = 0.7, scale = 0.9) +
  scale_fill_manual(values = c("#2E86AB", "#C73E1D"),
                    name = "Resultado") +
  labs(title = "DISTRIBUCIÓN DE VARIABLES CLAVE POR RESULTADO",
       subtitle = "Comparación entre estudiantes que aprueban vs reprueban",
       x = "Valor", y = "Variable") +
  tema_presentacion() +
  theme(legend.position = "bottom")

ggsave("resultados_EDA/03_ridgeline_distribuciones.png", p3_ridgeline,
       width = 12, height = 8, dpi = 300)

# ------------------------------------------------------------------------------
# GRÁFICO 4: FACTORES SOCIODEMOGRÁFICOS (FACET WRAP)
# ------------------------------------------------------------------------------
cat("Generando Gráfico 4: Factores sociodemográficos...\n")

datos_categoricas <- datos_procesados %>%
  select(reprobado_factor, sex, address, famsize, higher, internet, romantic) %>%
  pivot_longer(cols = -reprobado_factor, 
               names_to = "categoria", 
               values_to = "nivel") %>%
  group_by(categoria, nivel, reprobado_factor) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(categoria, nivel) %>%
  mutate(porcentaje = n/sum(n)*100) %>%
  ungroup() %>%
  mutate(categoria = str_to_title(categoria))

p4_facet_categorico <- datos_categoricas %>%
  ggplot(aes(x = nivel, y = porcentaje, fill = reprobado_factor)) +
  geom_col(position = "dodge", alpha = 0.9) +
  geom_text(aes(label = paste0(round(porcentaje, 0), "%")),
            position = position_dodge(width = 0.9),
            vjust = -0.3, size = 3) +
  facet_wrap(~ categoria, scales = "free_x", ncol = 3) +
  scale_fill_manual(values = c("#2E86AB", "#C73E1D"),
                    name = "Resultado") +
  labs(title = "ANÁLISIS DE FACTORES SOCIODEMOGRÁFICOS",
       subtitle = "Tasa de reprobación por diferentes categorías",
       x = "", y = "Porcentaje (%)") +
  tema_presentacion() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
        legend.position = "bottom")

ggsave("resultados_EDA/04_facet_categorico.png", p4_facet_categorico,
       width = 14, height = 10, dpi = 300)

# ------------------------------------------------------------------------------
# GRÁFICO 5: IMPACTO DE AUSENCIAS (LÍNEA DE TIEMPO)
# ------------------------------------------------------------------------------
cat("Generando Gráfico 5: Impacto de ausencias...\n")

p5_timeline_ausencias <- datos_procesados %>%
  group_by(absences_cat) %>%
  summarise(
    tasa_reprobacion = mean(reprobado) * 100,
    n_estudiantes = n(),
    error_est = sd(reprobado) / sqrt(n()) * 100
  ) %>%
  ggplot(aes(x = absences_cat, y = tasa_reprobacion, group = 1)) +
  geom_ribbon(aes(ymin = tasa_reprobacion - error_est,
                  ymax = tasa_reprobacion + error_est),
              fill = "#C73E1D", alpha = 0.2) +
  geom_line(color = "#C73E1D", size = 1.5) +
  geom_point(aes(size = n_estudiantes), color = "#C73E1D") +
  geom_text(aes(label = paste0(round(tasa_reprobacion, 0), "%")),
            vjust = -1.5, size = 3.5, fontface = "bold") +
  scale_size_continuous(range = c(3, 8), name = "N° Estudiantes") +
  labs(title = "IMPACTO DE LAS AUSENCIAS EN EL RENDIMIENTO",
       subtitle = "Relación entre número de ausencias y tasa de reprobación",
       x = "Rango de Ausencias", y = "Tasa de Reprobación (%)") +
  tema_presentacion()

ggsave("resultados_EDA/05_timeline_ausencias.png", p5_timeline_ausencias,
       width = 12, height = 6, dpi = 300)

# ------------------------------------------------------------------------------
# GRÁFICO 6: COMPARATIVA NUMÉRICA (BOXPLOTS)
# ------------------------------------------------------------------------------
cat("Generando Gráfico 6: Comparativa numérica...\n")

p6_boxplots <- datos_procesados %>%
  select(reprobado_factor, age, failures, absences, studytime, famrel) %>%
  pivot_longer(cols = -reprobado_factor, 
               names_to = "variable", 
               values_to = "valor") %>%
  mutate(variable = case_when(
    variable == "age" ~ "Edad",
    variable == "failures" ~ "Fallos Previos",
    variable == "absences" ~ "Ausencias",
    variable == "studytime" ~ "Tiempo de Estudio",
    variable == "famrel" ~ "Relaciones Familiares"
  )) %>%
  ggplot(aes(x = reprobado_factor, y = valor, fill = reprobado_factor)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 21, outlier.size = 2) +
  scale_fill_manual(values = c("#2E86AB", "#C73E1D"), name = "Resultado") +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  labs(title = "COMPARACIÓN DE VARIABLES NUMÉRICAS POR RESULTADO",
       subtitle = "Distribución de características entre grupos",
       x = "", y = "Valor") +
  tema_presentacion() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave("resultados_EDA/06_boxplots_comparativos.png", p6_boxplots,
       width = 14, height = 8, dpi = 300)

# ------------------------------------------------------------------------------
# GRÁFICO 7: LAYOUT COMPLETO (COMBINACIÓN)
# ------------------------------------------------------------------------------
cat("Generando Gráfico 7: Layout completo...\n")

layout_completo <- (p1_distribucion | p2_heatmap_riesgo) /
  (p3_ridgeline | p6_boxplots) /
  p5_timeline_ausencias +
  plot_annotation(
    title = "ANÁLISIS EXPLORATORIO COMPLETO - FACTORES DE RIESGO EN MATEMÁTICAS",
    subtitle = "Identificación de patrones asociados al rendimiento académico",
    caption = paste("Dataset: Student Performance | Total estudiantes:", nrow(datos_procesados)),
    theme = theme(
      plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
      plot.background = element_rect(fill = "white", color = NA)
    )
  )

ggsave("resultados_EDA/07_layout_completo.png", layout_completo,
       width = 16, height = 18, dpi = 300)

# ------------------------------------------------------------------------------
# PARTE 3: TABLAS RESUMEN PARA QUARTO
# ------------------------------------------------------------------------------
cat("=== GENERANDO TABLAS RESUMEN ===\n")

# 3.1 Tabla de estadísticas descriptivas
tabla_estadisticas <- datos_procesados %>%
  summarise(
    Variable = c("Tasa Reprobación", "Edad Promedio", "Ausencias Medias", 
                 "Fallos Previos", "Tiempo Estudio", "Educación Madre",
                 "Educación Padre", "Relaciones Familiares"),
    `Total (n=395)` = c(
      paste0(round(mean(reprobado) * 100, 1), "%"),
      paste0(round(mean(age, na.rm = TRUE), 1), " años"),
      round(mean(absences, na.rm = TRUE), 1),
      round(mean(failures, na.rm = TRUE), 2),
      round(mean(studytime, na.rm = TRUE), 1),
      round(mean(Medu, na.rm = TRUE), 1),
      round(mean(Fedu, na.rm = TRUE), 1),
      round(mean(famrel, na.rm = TRUE), 1)
    ),
    `Aprobados (n=325)` = c(
      "0%",
      paste0(round(mean(age[reprobado == 0], na.rm = TRUE), 1), " años"),
      round(mean(absences[reprobado == 0], na.rm = TRUE), 1),
      round(mean(failures[reprobado == 0], na.rm = TRUE), 2),
      round(mean(studytime[reprobado == 0], na.rm = TRUE), 1),
      round(mean(Medu[reprobado == 0], na.rm = TRUE), 1),
      round(mean(Fedu[reprobado == 0], na.rm = TRUE), 1),
      round(mean(famrel[reprobado == 0], na.rm = TRUE), 1)
    ),
    `Reprobados (n=70)` = c(
      "100%",
      paste0(round(mean(age[reprobado == 1], na.rm = TRUE), 1), " años"),
      round(mean(absences[reprobado == 1], na.rm = TRUE), 1),
      round(mean(failures[reprobado == 1], na.rm = TRUE), 2),
      round(mean(studytime[reprobado == 1], na.rm = TRUE), 1),
      round(mean(Medu[reprobado == 1], na.rm = TRUE), 1),
      round(mean(Fedu[reprobado == 1], na.rm = TRUE), 1),
      round(mean(famrel[reprobado == 1], na.rm = TRUE), 1)
    ),
    `Diferencia` = c(
      "100%",
      paste0(round(mean(age[reprobado == 1], na.rm = TRUE) - 
                     mean(age[reprobado == 0], na.rm = TRUE), 1), " años"),
      round(mean(absences[reprobado == 1], na.rm = TRUE) - 
              mean(absences[reprobado == 0], na.rm = TRUE), 1),
      round(mean(failures[reprobado == 1], na.rm = TRUE) - 
              mean(failures[reprobado == 0], na.rm = TRUE), 2),
      round(mean(studytime[reprobado == 1], na.rm = TRUE) - 
              mean(studytime[reprobado == 0], na.rm = TRUE), 1),
      round(mean(Medu[reprobado == 1], na.rm = TRUE) - 
              mean(Medu[reprobado == 0], na.rm = TRUE), 1),
      round(mean(Fedu[reprobado == 1], na.rm = TRUE) - 
              mean(Fedu[reprobado == 0], na.rm = TRUE), 1),
      round(mean(famrel[reprobado == 1], na.rm = TRUE) - 
              mean(famrel[reprobado == 0], na.rm = TRUE), 1)
    )
  )

# Guardar tabla
write_csv(tabla_estadisticas, "resultados_EDA/tabla_estadisticas.csv")
cat("Tabla de estadísticas guardada en: resultados_EDA/tabla_estadisticas.csv\n")

# 3.2 Tabla de tasas por categoría
tabla_tasas_categoria <- datos_procesados %>%
  summarise(
    Categoría = c("Sexo (Mujer)", "Sexo (Hombre)", 
                  "Dirección (Urbano)", "Dirección (Rural)",
                  "Educación Superior (Sí)", "Educación Superior (No)",
                  "Internet (Sí)", "Internet (No)"),
    `N Estudiantes` = c(
      sum(sex == "F"), sum(sex == "M"),
      sum(address == "U"), sum(address == "R"),
      sum(higher == "yes"), sum(higher == "no"),
      sum(internet == "yes"), sum(internet == "no")
    ),
    `Tasa Reprobación (%)` = c(
      round(mean(reprobado[sex == "F"]) * 100, 1),
      round(mean(reprobado[sex == "M"]) * 100, 1),
      round(mean(reprobado[address == "U"]) * 100, 1),
      round(mean(reprobado[address == "R"]) * 100, 1),
      round(mean(reprobado[higher == "yes"]) * 100, 1),
      round(mean(reprobado[higher == "no"]) * 100, 1),
      round(mean(reprobado[internet == "yes"]) * 100, 1),
      round(mean(reprobado[internet == "no"]) * 100, 1)
    )
  )

write_csv(tabla_tasas_categoria, "resultados_EDA/tabla_tasas_categoria.csv")
cat("Tabla de tasas por categoría guardada\n")

# ------------------------------------------------------------------------------
# PARTE 4: RESUMEN FINAL
# ------------------------------------------------------------------------------
cat("\n=== ANÁLISIS COMPLETADO ===\n")
cat("================================\n")
cat("RESULTADOS GENERADOS:\n")
cat("1. Gráficos guardados en carpeta 'resultados_EDA/':\n")
cat("   - 01_distribucion_resultados.png\n")
cat("   - 02_matriz_riesgo.png\n")
cat("   - 03_ridgeline_distribuciones.png\n")
cat("   - 04_facet_categorico.png\n")
cat("   - 05_timeline_ausencias.png\n")
cat("   - 06_boxplots_comparativos.png\n")
cat("   - 07_layout_completo.png\n")
cat("\n2. Tablas de resumen:\n")
cat("   - tabla_estadisticas.csv\n")
cat("   - tabla_tasas_categoria.csv\n")
cat("\n3. Estadísticas clave:\n")
cat("   - Total estudiantes:", nrow(datos_procesados), "\n")
cat("   - Tasa de reprobación:", round(mean(datos_procesados$reprobado) * 100, 1), "%\n")
cat("   - Variables analizadas:", ncol(datos_procesados), "\n")
cat("\n¡Listo para incluir en documento Quarto!\n")
