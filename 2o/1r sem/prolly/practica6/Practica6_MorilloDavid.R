######### Practica 6 #########

############################## Excercici 1 #########################################
## Contrast mitjana - sigma desconeguda

load("dades1.RData")
x <- dades1$Nombres
n <- length(x)
sigma <- 0.9
xm <- mean(x)

# TEST D'HIPOTESIS:
# H0 : mu = 4
# H1 : mu < 4

mu <- 4
z <- (xm-mu)/sigma*sqrt(n)
pvalor <- pnorm(z)
cat("P-Valor:", pvalor, "\n")
# pvalor > alpha ---> Acceptem H0





############################## Excercici 2 #########################################
## Contrast mitjana - sigma desconeguda
load("dades2.RData")
x <- dades2$Nombres
alpha <- 0.1

## a)
# TEST D'HIPOTESIS:
# H0 : mu = 11
# H1 : mu > 11

t.test(dades2, mu = 11, alternative = "greater", conf.level = 1 - alpha)
# pvalor < alpha ---> Acceptem H1

## b)
# TEST D'HIPOTESIS:
# H0 : mu = 12
# H1 : mu =! 12

t.test(dades2, mu = 12, alternative = "two.sided", conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0





############################## Excercici 3 #########################################
## Contrast proporció

# TEST D'HIPOTESIS:
# H0 : p = 0.4
# H1 : mu > 0.4

N <- 135
n <- 300

alpha <- 0.05
prop.test(N, n, p = 0.4, alternative = "greater", correct = FALSE, conf.level = 1 - alpha)
# pvalor < alpha ---> Acceptem H1

alpha <- 0.01
prop.test(N, n, p = 0.4, alternative = "greater", correct = FALSE, conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0




############################## Excercici 4 #########################################
## Contrast sobre varianza
  
# TEST D'HIPOTESIS:
# H0 : sigma^2 = 1
# H1 : sigma^2 =! 1

x <- dades2$Nombres
alpha <- 0.1

library(EnvStats)

sigma2 <- 1.2
varTest(x, sigma.squared = sigma2 , conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0





############################## Excercici 5 #########################################
load("kidsfeet.RData")

## a)
# Contrast mitjana - sigma desconeguda

x <- kidsfeet$Longitud

# TEST D'HIPOTESIS:
# H0 : ml = 8.8
# H1 : ml > 8.8

alpha <- 0.05
t.test(x, mu = 8.8, alternative = "greater", conf.level = 1-alpha)
# pvalor < alpha ---> Acceptem H1

alpha <- 0.01
t.test(x, mu = 8.8, alternative = "greater", conf.level = 1-alpha)
# pvalor < alpha ---> Acceptem H1





## b)
## Contrast proporció

x <- kidsfeet$Peu
esq <- sum(x == "L")
n <- length(x)

# TEST D'HIPOTESIS:
# H0 : p = 0.3
# H1 : p < 0.3

prop_esq <- esq / n
# Test d'hp unilateral esquerra
alpha <- 0.05
prop.test(esq, n, p = 0.3, alternative = "less", conf.level = 1 - alpha, correct = FALSE)
# pvalor > alpha ---> Acceptem H0

alpha <- 0.10
prop.test(esq, n, p = 0.3, alternative = "less", conf.level = 1 - alpha, correct = FALSE)
# pvalor > alpha ---> Acceptem H0





## c)

x <- kidsfeet$Amplada

# TEST D'HIPOTESIS:
# H0 : sigma = 0.56
# H1 : sigma < 0.56

library(EnvStats)

sigma_p <- 0.56
alpha <- 0.05
varTest(x, sigma = sigma_p, conf.level = 1 - alpha)
# pvalor < alpha ---> Acceptem H1

alpha <- 0.03
varTest(x, sigma = sigma_p, conf.level = 1 - alpha)
# pvalor < alpha ---> Acceptem H1





## d)
alpha = 0.05

# TEST D'HIPOTESIS:
# H0: alphaA2 = alphaB2
# H1: alphaA2 != alphaB2

A = kidsfeet$Longitud
B = kidsfeet$Amplada

var.test(A, B, conf.level = 1 - alpha)
# p-value = 4.929e-08
# pvalor < alpha ---> Acceptem H1

# Diferencia significativa





## e)
alpha = 0.03

esq <- kidsfeet$Longitud[kidsfeet$Peu == "L"]
dre <- kidsfeet$Longitud[kidsfeet$Peu == "R"]

# TEST D'HIPOTESIS:
# H0: mA = mB
# H1: mA != mB

t.test(esq, dre, conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0
# Sense iferencia significativa






## f)
t.test(esq, dre, conf.level = 1 - alpha, var.equal = TRUE)
# pvalor > alpha ---> Acceptem H0
# Sense diferencia significativa





## g)
alpha =  0.03

esq <- kidsfeet$Longitud[kidsfeet$Peu == "L"]
dre <- kidsfeet$Longitud[kidsfeet$Peu == "R"]

# TEST D'HIPOTESIS:
# H0: sigma2A == sigma2B
# H1: sigma2A != sigma2B

var.test(esq, dre, conf.level = 1 - alpha)

# pvalor > alpha ---> Acceptem H0
# Sense diferencia significativa

#################### Excercici 6 ######################

v_A <- 615
n <- 1000
p_nula <- 0.60

# TEST D'HIPOTESIS:
# H0: p = 0.6
# H1: p > 0.6

prop.test(v_A, n, p = p_nula, alternative = "greater")
# pvalor > alpha ---> Acceptem H0
# No podem afirmar que es major

###################### Excercici 7 ########################

load("Anime.RData")

# a)

x <- anime$FONT
n <- length(x)
manga <- sum(x == "Manga")

p_manga <- manga / n
p_other <- 1- p_manga

# TEST D'HIPOTESIS:
# H0 : p_manga = p_other
# H1 : p_manga < p_other

p_nula <- 0.5
prop.test(manga, n, p = p_nula, alternative = "less")
# pvalor > alpha ---> Acceptem H0





# b)

# TEST D'HIPOTESIS:
# H0 : v_manga = v_manga
# H1 : v_manga > v_other

val_manga <- anime$PUNTS[anime$FONT == "Manga" & anime$ANY >= 2000]
val_other <- anime$PUNTS[anime$FONT != "Manga" & anime$ANY >= 2000]

t.test(val_manga, val_other, alternative = "greater", var.equal = FALSE)
# pvalor < alpha ---> Acceptem H1
# Sense diferencia significativa


