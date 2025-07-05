taula <- data.frame(valoracio=c(1,2,3,4,5,6,7,8,9,10), frequencia=c(2,8,5,13,20,37,50,68,42,19))
a <- rep(taula$valoracio, taula$frequencia)
a

barplot(table(a))

mean(a)
median(a)

summary(a)

#FA, FAC, FR, FRA
cbind(table(a), cumsum(table(a)), table(a)/length(a), cumsum(table(a))/length(a))

varp <- function(a){
  n <- length(a)
  varx <- var(a)*(n-1)/n
  return(varx)
}

varp(a) # variança (no corregida)
var(a) # variança corregida
sqrt(varp(a)) # no corregida
sqrt(var(a)) # corregida

data <- read.table("UAB/2o/prolly/bonambient.csv", header = TRUE)
d = (data$valoracions)
d = table(data)
d

cbind(table(d), cumsum(table(d)), table(d)/length(d), cumsum(table(d))/length(d))
c(d)

hist(d)
barplot(d)

mean(d)
median(d)

summary(d)
