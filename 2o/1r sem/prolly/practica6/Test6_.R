# David Morillo Massagué

###################################
######### Test Practica 6 #########
###################################

library(EnvStats)

# Assigneu com a valor a la variable m el vostre NIU:
m <- 1666540

## Dades
set.seed(m)
n <- 120
dades <- round(rnorm(n, mean = 6, sd = 1), 1)





## Exercici 1 ################################
## Contrast mitjana - sigma desconeguda
x <- dades
alpha <- 0.01

## a)
# TEST D'HIPOTESIS:
# H0 : mu = 5.9
# H1 : mu != 5.9

t.test(x, mu = 5.9, alternative = "two.sided", conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0
# mitjana igual a 5.9







## Exercici 2
alpha <- 0.01
# TEST D'HIPOTESIS:
# H0 : sigma^2 = 1.3
# H1 : sigma^2 < 1.3

x <- dades

library(EnvStats)

sigma2 <- 1.3
varTest(x, sigma.squared = sigma2, alternative = "less", conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0
# sigma2 no menor a 1.3









## Exercici 3
## Contrast proporció

# TEST D'HIPOTESIS:
# H0 : p = 0.2
# H1 : p > 0.2
x <- dades
n <- len(dades)

major7 <- sum(x > 7)

proporcio = major7/n

alpha <- 0.01
prop.test(major7, n, p = 0.2, alternative = "greater", correct = FALSE, conf.level = 1 - alpha)
# pvalor > alpha ---> Acceptem H0
# proporció no major al 20%




