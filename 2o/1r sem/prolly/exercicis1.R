e <- exp(1)
A <- log(5)+3**5-sqrt(3*pi)*sin((2*pi)/3)-nthroot(e,3)
A
B <- (sqrt(3) + 5*pi)/(7-nthroot(2,5))
B
C = 2**(-7/13)*(11/9)**(-8/7)
C
s <- 763:825
D <- sum(1/s)
D
s <- 4:9
E <- sum(((3^s)*fact(s))/(s^s))
E

A <- matrix(c(7,8,10,5), 2,2)
A
B <- matrix(c(1,4,5,0,0,11), 2,3)
B
C <- matrix(c(7,0,3,12,9,3), 3,2)
C
D <- matrix(c(13,7,2,3), 2,2)
D

A+D

A%*%B
C%*%A
A%*%D

f <- function(n){
  s <- 1:n
  res <- sum(1/e**s)
  return(res)
}


g <- function(j,n) {
  s = j:n
  res = sum(choose(n,j)*1/fact(j))
  return(res)
}

f(10)
f(100)

g(4,10)
g(6,23)
