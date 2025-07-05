# Datos
rendimiento <- c(15, 18, 14, 17, 16, 22, 24, 21, 23, 22, 19, 20, 18, 20, 19)
fertilizante <- factor(rep(c("A", "B", "C"), each = 5))

# Realizar ANOVA
modelo <- aov(rendimiento ~ fertilizante)
summary(modelo)
