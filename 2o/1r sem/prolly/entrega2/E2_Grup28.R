# Cargar l'arxiu CSV
enquesta <- read.csv("C:/Users/adria/OneDrive/Escritorio/enquesta.csv")

# Probabilitat de fumar
fumar_total <- sum(enquesta$TABAC == 1) / nrow(enquesta)

# Probabilitat de fumar entre els que fan esport
fumar_esport <- sum(enquesta$TABAC == 1 & enquesta$OCI == 4) / sum(enquesta$OCI == 4)

# Probabilitat de fumar entre els que no fan esport
fumar_no_esport <- sum(enquesta$TABAC == 1 & enquesta$OCI != 4) / sum(enquesta$OCI != 4)

cat("Probabilidad de fumar en toda la muestra:", fumar_total, "\n")
cat("Probabilidad de fumar entre los que hacen deportes:", fumar_esport, "\n")
cat("Probabilidad de fumar entre los que no hacen deportes:", fumar_no_esport, "\n")

# Calcular la probabilidad de ser alta (>1.60 m) entre las mujeres
dona_alta <- sum(enquesta$ALTURA > 160 & enquesta$SEXE == 'd') / sum(enquesta$SEXE == 'd')

# Calcular la probabilidad de que la actividad principal sea televisión o computadora entre las mujeres
tv_ordinador_dona <- sum((enquesta$OCI == 1 | enquesta$OCI == 2) & enquesta$SEXE == 'd') / sum(enquesta$SEXE == 'd')

cat("Probabilidad de ser alta (>1.60 m) entre las mujeres:", dona_alta, "\n")
cat("Probabilidad de que la actividad principal sea televisión o computadora entre las mujeres:", tv_ordinador_dona, "\n")

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

cat("Probabilidad de que una persona haga deporte y no fume o dedique su tiempo libre a la computadora o música (o lectura):", prob_esport_no_fuma_ordinador_musica, "\n")
cat("Probabilidad de que una persona pese más de 60 kg entre los que tienen al menos 20 años:", prob_peso_mas_60_edad_20, "\n")
cat("Probabilidad de que una persona pese menos de 70 kg entre los que tienen como máximo 50 años:", prob_peso_menos_70_edad_50, "\n")

#2

#Funció dau()
dau <- function(k){
  if (k %% 2 == 0){
    res <- (k-1)/(5*k)
  }
  else{
    res <- (1+2*k)/36
  }
  return(res)
}

  
calcular_probabilidades_dado <- function(){
  caras <- 1:6
  probabilidades <- c(dau(1),dau(2),dau(3),dau(4),dau(5),dau(6)) 
  return(data.frame(Cara = caras, Probabilidad = probabilidades))
}

probabilitats <- calcular_probabilidades_dado()

#Representació de funció massa probabilitat
plot (probabilitats$Cara , probabilitats$Probabilidad, type = "h " , lwd =2 , bty = "n" , las =1 , xlim = c (0 ,8) ,
      ylim = c (0 ,0.3) , xaxp = c (0 , 8 , 8) , col = " orange " ,
      xlab = " Valors " , ylab = " Probabilitats " )
grid ()


#Representació de la funció de distribució de X amb un dau normal

#X
acum <- cumsum ( probabilitats$Probabilidad )
s <- stepfun (probabilitats$Cara , c (0 , acum ))

#DAU NORMAL
y <- c (1:6)
y2 <- rep(1/6, 6)
acum2 <- cumsum ( y2 )
s2 <- stepfun (y , c (0 , acum2 ))

par(mfrow = c(1, 2))

plot(s2, col = "blue", main = "DAU NORMAL", xlab = "CARA", ylab = "Probabilitat acumulada", xaxp = c (1 , 6 , 5), xlim = c(1,6))
plot(s, col = "red", main = "X", xlab = "CARA", ylab = "Probabilitat acumulada", xaxp = c (1,6,5), xlim = c(1,6))

par(mfrow = c(1, 1))

#Càlcul de l'esperança de la variable aleatòria X

esperança <- sum(probabilitats$Cara*probabilitats$Probabilidad)
esperança

variança <- sum(((probabilitats$Cara-esperança)^2)*probabilitats$Probabilidad)
variança


# Mostra de n = 300

mostra <- sample(probabilitats$Cara, 300, replace = TRUE, prob = probabilitats$Probabilidad)

mean(mostra)
var(mostra) #LA COMPARACIÓ S'APROXIMA MOLT


#3 

#Calcular i representar la funció de densitat

funcio_densitat <- function(x) {
  if (x > 1 & x < 3) {
    return(((x - 1)^3) /4)
  } else {
    return(0)
  }
}

x_vals <- seq(1.01, 2.99, length.out = 100)
y_vals <- sapply(x_vals, funcio_densitat)

plot(x_vals, y_vals, type = "l", col = "blue", lwd = 2, xlab = "x", ylab = "f(x)", main = "Funció de densitat")

#Representar la funció de distribució d'X

acum <- cumsum ( y_vals )
s <- stepfun (x_vals , c (0 , acum ))
plot(s)










