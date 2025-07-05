x <- seq(-0.5,0.5,by=0.005)
y <- cos(2/x)
y2 <- sin(2/x)
y3 <- y2+y
plot(x,y3,type="l")

A = 1:12
A
dim(A) <- c(3 ,4)
A

B = 13:24
B
dim(B) <- c(4 ,3)
B

A%*%B

X <- 1:9
dim(X) <- c(3,3)

X <- matrix(c(1,4,3,4,-5,6,7,8,9), 3, 3)

solve(X)
det(X)
dim(X)
range(X)
pi

