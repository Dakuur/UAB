iris
x <- LakeHuron

head(x)
table(x)
table (x) / length (x)
a <- cumsum(table(x)/length (x))

hist(x)

barplot(table(x))
pie(table(x))
boxplot(table(x))
summary(x)
mean(x, trim = 0.2)

cbind(table(x), table(x)/length(x), cumsum(table(x)), cumsum(table(x))/length(x))
summary(x)
summary(LakeHuron)