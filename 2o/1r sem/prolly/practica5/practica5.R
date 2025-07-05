#Nom: David Morillo Massagué
#NIU: 1666540


set.seed(1666540)

#EXERCICI 1

print("-------- EXERCICI 1 --------")

N <- 300
n <- 20
p <- 0.4

z <- qnorm(0.995)

mostra <- rbinom(N, n, p)

M <- mean(mostra)
desv <- sqrt((p * (1 - p) * n) / N)

err <- z * desv / sqrt(N)
interval <- c(M - err, M + err)

mitja_teoria <- n * p
contie_mitja_teoria <- interval[1] <= mitja_teoria & interval[2] >= mitja_teoria

print("Interval de confiança del 99% per la mitjana:")
print(interval)

print("Mitja teòrica:")
print(mitja_teoria)

print("Mitjana teòrica dins de l'interval?")
print(contie_mitja_teoria)

#EXERCICI 2

print("-------- EXERCICI 2 --------")

N <- 1000
n <- 20
m <- 3

z <- qnorm(0.95)

te_n <- numeric(N)

for (i in 1:N) {
  conjunt <- rnorm(n, m, 1)
  
  sigma_conj = sd(conjunt)
  
  mitja <- mean(conjunt)
  err <- z * sigma_conj / sqrt(n)
  
  interval <- c(mitja - err, mitja + err)
  
  te_n[i] <- interval[1] <= m && interval[2] >= m
}

ratio <- sum(te_n) / N

print("Proporció d'intervals que contenen m:")
print(ratio)

#EXERCICI 3

print("-------- EXERCICI 3 --------")

argent <- c(5.2, 4.8, 5.3, 5.7, 5.0, 4.7, 4.3, 5.5, 5.4, 5.1, 4.9, 5.8)

print("Amb confiança 0.95:")
conf <- 0.95
resultat <- t.test(argent , conf.level = conf)$conf.int
print(resultat)

print("Amb confiança 0.99:")
conf <- 0.99
resultat <- t.test(argent , conf.level = conf)$conf.int
print(resultat)

#EXERCICI 4

print("-------- EXERCICI 4 --------")

total <- 200
conf <- 0.92
perc_malalts <- 0.08

resultat <- prop.test(total*perc_malalts, total, conf.level = conf)$conf.int
print("Interval amb confiança 0.92:")
print(resultat)

#EXERCICI 5

print("-------- EXERCICI 5 --------")

#install.packages('EnvStats')
library(EnvStats)

conf <- 0.93
notes <- c(7.9, 8.3, 4.8, 8.4, 7.9, 5.2, 5.6, 3.2, 9.1, 7.7, 6.5, 4.4)
resultat <- varTest(notes , conf.level = conf)$conf.int
print("Interval amb confiança 0.93:")
print(resultat)

#EXERCICI 6

print("-------- EXERCICI 6 --------")

abans <- c(93, 106, 87, 92, 102, 95, 88, 110)
despres <- c(92, 102, 89, 92, 101, 96, 88, 105)

conf <- 0.97
resultat <- t.test (abans, despres, paired = TRUE, conf.level = conf)$conf.int
print("Interval amb confiança 0.97:")
print(resultat)

imprecis <- resultat[1] <= 0 & resultat[2] >= 0

decreix <- (max(resultat) < 0 & !imprecis)

creix <- (min(resultat) > 0 & !imprecis)
  
if(decreix){
  print("La dieta fa que la pressió disminueixi")
}

if(creix){
  print("La dieta fa que la pressió creixi")
}

if(imprecis){
  print("Com que 0 està dins de l'interval, no podem extraure una conclusió de l'efecte de la dieta (no hii ha suficient diferencia significa entre els dos grups)")
}

#EXERCICI 7

print("-------- EXERCICI 7 --------")

l <- c(6.7, 1.9, 6.4, 4.8, 2.6, 4.9, 6.7, 3.6, 1.5, 1.2, 2.4, 2.4, 4.6, 4.9, 4.8)
s <- c(6.2, 3.7, 4.5, 6.2, 6.0, 5.3, 3.5, 3.6, 3.1, 0.3, 5.3, 4.5, 4.5, 3.6, 4.5)

conf <- 0.95
var_l <- 3.5
var_s <- 2.2

resultat <- t.test (abans, despres, paired = FALSE, conf.level = conf)$conf.int
print("Interval amb confiança 0.95:")
print(resultat)

#EXERCICI 8

print("-------- EXERCICI 8 --------")

l <- c(6.7, 1.9, 6.4, 4.8, 2.6, 4.9, 6.7, 3.6, 1.5, 1.2, 2.4, 2.4, 4.6, 4.9, 4.8)
s <- c(6.2, 3.7, 4.5, 6.2, 6.0, 5.3, 3.5, 3.6, 3.1, 0.3, 5.3, 4.5, 4.5, 3.6, 4.5)

conf <- 0.95

iguals <- t.test(l, s, var.equal=TRUE, conf.level = conf)$conf.int
print("Assumint variàncies iguals:")
print(iguals)

diferents <- t.test(l, s, var.equal=FALSE, conf.level = conf)$conf.int
print("Assumint variàncies diferents:")
print(diferents)

#EXERCICI 9

print("-------- EXERCICI 9 --------")

load("~/UAB/2o/prolly/practica5/malaria.RData")
head(malaria)

#-------- A --------

edat <- malaria$Edat

conf <- 0.95
resultat <- t.test(edat , conf.level = conf)$conf.int
print("Interval de confiança del 0.95 per l'edat:")
print(resultat)

#-------- B --------

edat_homes <- malaria$Edat[malaria$Sexe == "H"]
edat_dones <- malaria$Edat[malaria$Sexe == "D"]

n_homes <- length(edat_homes)
n_dones <- length(edat_dones)
total <- n_homes + n_dones

prop_dones <- n_dones / total
prop_homes <- n_homes / total

conf <- 0.92

interval_dones <- prop.test(n_dones, total, conf.level = conf)$conf.int
print("Interval de confiança del 0.92 per la proporció de dones:")
print(interval_dones)

interval_homes <- prop.test(n_homes, total, conf.level = conf)$conf.int
print("Interval de confiança del 0.92 per la proporció d'homes:")
print(interval_homes)

#-------- C --------

conf <- 0.93
edat_homes <- malaria$Edat[malaria$Sexe == "H"]
edat_dones <- malaria$Edat[malaria$Sexe == "D"]

res <- t.test(edat_homes, edat_dones, var.equal=TRUE, conf.level = conf)$conf.int
print("Interval de confiança del 0.93 per la diferencia de mitjanes:")
print(res)

#-------- D --------

edat <- malaria$Edat

conf <- 0.90

interval <- varTest (edat, conf.level = conf)$conf.int
print("Interval de confiança del 0.90 per la variància de la variable Edat:")
print(interval)

#EXERCICI 10

print("-------- EXERCICI 10 --------")

#-------- A --------

primer <- c(7.9, 5.4, 8.3, 6.2, 8.2, 8.3, 7.8, 4.9, 6.2, 8.9, 7.8, 9.7, 7.2)
segon <- c(8.2, 5.7, 6.0, 4.2, 7.5, 4.6, 6.2, 5.2, 5.3, 9.2, 6.5, 8.1, 4.5)

conf <- 0.90
resultat <- t.test(primer , conf.level = conf)$conf.int
print("Interval de confiança del 0.90 per el primer parcial:")
print(resultat)

resultat <- t.test(segon , conf.level = conf)$conf.int
print("Interval de confiança del 0.90 per el segon parcial:")
print(resultat)

conf <- 0.80
resultat <- t.test(primer , conf.level = conf)$conf.int
print("Interval de confiança del 0.90 per el primer parcial:")
print(resultat)

resultat <- t.test(segon , conf.level = conf)$conf.int
print("Interval de confiança del 0.90 per el segon parcial:")
print(resultat)

print("Com que en el primer cas (confiança 0.90) hi ha solapament entre els intervals, no es pot concloure que la nota mitjana del primer parcial és més alta que la del segon")

print("Com que en el segon cas (confiança 0.80) no hi ha solapament entre els intervals, podem concloure que en el primer parcial hi ha hagut millors notes que en el segon")

#-------- B --------

conf <- 0.92

resultat <- t.test(primer, segon, conf.level = conf)$conf.int
print("Interval de confiança del 92% per a la diferència de mitjanes:")
print(resultat)

print("Com que l'interval no conté 0, es pot determinar que el primer conjunt (primer parcial) té millor mitjana")

#-------- C --------

conf <- 0.95
notes_finals <- primer * 0.4 + segon * 0.6
interval <- t.test(notes_finals, conf.level = conf)$conf.int
print("Interval de confiança del 0.93 per les notes finalss:")
print(interval)

#-------- D --------

conf <- 0.93
var_primer <- varTest(primer, conf.level = conf)$conf.int
print("Interval de confiança 0.93 per la variança del primer parcial:")
print(var_primer)

var_segon <- varTest(segon, conf.level = conf)$conf.int
print("Interval de confiança 0.93 per la variança del segon parcial:")
print(var_segon)

print("Com que hi ha solapament entre els intervals, no es pot concloure quin conjunt de notes tenen més variabilitat")