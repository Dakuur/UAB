# Cargar el archivo CSV en un marco de datos
enquesta <- read.csv("C:/Users/david/OneDrive - UAB/Documentos/UAB/2o/prolly/entrega2/enquesta.csv")

# Ver las primeras filas del marco de datos
head(enquesta)

# Supongamos que tienes un marco de datos llamado 'enquesta' con las columnas mencionadas (EDAT, SEXE, TABAC, OCI, etc.)

# Calcular la probabilidad de fumar en toda la muestra
prob_fumar_total <- sum(enquesta$TABAC == 1) / nrow(enquesta)

# Calcular la probabilidad de fumar entre los que hacen deportes
prob_fumar_deportes <- sum(enquesta$TABAC == 1 & enquesta$OCI == 4) / sum(enquesta$OCI == 4)

# Calcular la probabilidad de fumar entre los que no hacen deportes
prob_fumar_no_deportes <- sum(enquesta$TABAC == 1 & enquesta$OCI != 4) / sum(enquesta$OCI != 4)

# Mostrar resultados
cat("Probabilidad de fumar en toda la muestra:", prob_fumar_total, "\n")
cat("Probabilidad de fumar entre los que hacen deportes:", prob_fumar_deportes, "\n")
cat("Probabilidad de fumar entre los que no hacen deportes:", prob_fumar_no_deportes, "\n")

# Calcular la probabilidad de ser alta (>1.60 m) entre las mujeres
prob_alta_dona <- sum(enquesta$ALTURA > 160 & enquesta$SEXE == 'd') / sum(enquesta$SEXE == 'd')

# Calcular la probabilidad de que la actividad principal sea televisión o computadora entre las mujeres
prob_tv_ordinador_dona <- sum(enquesta$OCI %in% c(1, 2) & enquesta$SEXE == 'd') / sum(enquesta$SEXE == 'd')

# Mostrar resultados
cat("Probabilidad de ser alta (>1.60 m) entre las mujeres:", prob_alta_dona, "\n")
cat("Probabilidad de que la actividad principal sea televisión o computadora entre las mujeres:", prob_tv_ordinador_dona, "\n")

# Probabilidad de que un hombre que practica deportes fume
prob_fuma_home_esport <- sum(enquesta$TABAC == 1 & enquesta$SEXE == 'h' & enquesta$OCI == 4) / sum(enquesta$SEXE == 'h' & enquesta$OCI == 4)

# Probabilidad de que una persona seleccionada al azar que fuma sea hombre y practique deportes
prob_home_esport_fuma <- sum(enquesta$TABAC == 1 & enquesta$SEXE == 'h' & enquesta$OCI == 4) / sum(enquesta$TABAC == 1)

# Mostrar resultados
cat("Probabilidad de que un hombre que practica deportes fume:", prob_fuma_home_esport, "\n")
cat("Probabilidad de que una persona seleccionada al azar que fuma sea hombre y practique deportes:", prob_home_esport_fuma, "\n")

# Probabilidad de que una persona haga deporte y no fume o dedique su tiempo libre a la computadora o música (o lectura)
prob_esport_no_fuma_ordinador_musica <- sum((enquesta$OCI %in% c(2, 3) | enquesta$TABAC == 0) & enquesta$OCI == 4) / nrow(enquesta)

# Probabilidad de que una persona pese más de 60 kg entre los que tienen al menos 20 años
prob_peso_mas_60_edad_20 <- sum(enquesta$PES > 60 & enquesta$EDAT >= 20) / sum(enquesta$EDAT >= 20)

# Probabilidad de que una persona pese menos de 70 kg entre los que tienen como máximo 50 años
prob_peso_menos_70_edad_50 <- sum(enquesta$PES < 70 & enquesta$EDAT <= 50) / sum(enquesta$EDAT <= 50)

# Mostrar resultados
cat("Probabilidad de que una persona haga deporte y no fume o dedique su tiempo libre a la computadora o música (o lectura):", prob_esport_no_fuma_ordinador_musica, "\n")
cat("Probabilidad de que una persona pese más de 60 kg entre los que tienen al menos 20 años:", prob_peso_mas_60_edad_20, "\n")
cat("Probabilidad de que una persona pese menos de 70 kg entre los que tienen como máximo 50 años:", prob_peso_menos_70_edad_50, "\n")
