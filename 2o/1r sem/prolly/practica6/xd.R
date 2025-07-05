######### Practica 6 #########

## Contrast mitjana - sigma coneguda
# Exercici 1

load("dades1.RData")
x <- dades1$Nombres
#x <- c(1,2,3,2,1,3,4,2,3,1,5,3,4)
n <- length(x)
sigma <- 0.9
xm <- mean(x)

# TEST D'HIPOTESIS:
# H0 : mu = 4
# H1 : mu < 4

mu <- 4
z <- (xm-mu)/sigma*sqrt(n)
pvalor <- pnorm(z)
# pvalor > alpha ---> Acceptem H0





## Contrast mitjana - sigma desconeguda
# Exercici 2
load("dades2.RData")
x <- dades2$Nombres
alpha <- 0.1

## a)
# TEST D'HIPOTESIS:
# H0 : mu = 11
# H1 : mu > 11
# Test d'hp unilateral dret

t.test(dades2, mu = 11, alternative = "greater", conf.level = 1-
         alpha)
# pvalor < alpha ---> Acceptem H1

## b)
# TEST D'HIPOTESIS:
# H0 : mu = 12
# H1 : mu =! 12
# Test d'hp bilateral

t.test(dades2, mu = 12, alternative = "two.sided", conf.level = 1-
         alpha)
# pvalor > alpha ---> Acceptem H0





## Contrast proporció
# Exercici 3

# TEST D'HIPOTESIS:
# H0 : p = 0.4
# H1 : mu > 0.4

# Test d'hp unilateral dret

N <- 135
n <- 300
prop.test(N, n, p = 0.4, alternative = "greater", correct = FALSE)
# pvalor < alpha=0.05 ---> Acceptem H1
# pvalor > alpha=0.01 ---> Acceptem H0 (Canvia la resposta)





## Contrast sobre varianza

# Exercici 4

# TEST D'HIPOTESIS:
# H0 : sigma^2 = 1
# H1 : sigma^2 =! 1
# Test d'hp bilateral

library(EnvStats)

sigma2 <- 1
varTest(x, sigma.squared = sigma2)
# pvalor > alpha ---> Acceptem H0