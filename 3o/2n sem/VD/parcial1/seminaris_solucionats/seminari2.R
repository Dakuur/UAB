?mtcars

library(tidyverse)

str(mtcars)

ggplot(mtcars, aes(x=factor(cyl), fill = factor(cyl))) +  geom_bar(width = 0.5) +
  xlab("Nombre de cilindres") + ylab("Cantitat de cotxes en el dataset") +
  scale_fill_discrete("Cylinder number", labels = c("Four","Six", "Eight")) +
  #guides(fill="none") +
  theme_minimal()

mtcars2 <- mtcars %>% mutate(am_str = factor(ifelse(am == 0, "Automatic", "Manual")))

ggplot(mtcars2, aes(cyl, fill = am_str)) + geom_bar(position = "fill") +
  xlab("Nomber of cylinders") + ylab("%") +
  scale_fill_discrete("Transmission type") +
  labs(title = "MTcars")

titanic <- read.csv("C:/Users/david/Downloads/titanic.csv")
ggplot(titanic, aes(x=factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "fill") +
  xlab("Passanger class") + ylab("%Survive rate")+
  labs(title = "Titanic victims")+
  scale_fill_discrete("", labels = c("Died","Survived"))


ggplot(mtcars, aes(hp)) + geom_histogram(binwidth = 10)

ggplot(mtcars, aes(hp)) + geom_freqpoly(binwidth = 5)



ggplot(mtcars2, aes(factor(cyl), mpg, fill = am_str)) +
  geom_boxplot(color="red", alpha=0.6, position = "dodge") +
  scale_fill_discrete("Transmission type")



ggplot(mtcars2, aes(x = qsec)) +
  geom_histogram(binwidth = 2,fill = "blue", alpha = 0.5) +
  geom_freqpoly(binwidth = 2, color ="red") +
  labs(title = "Histograma i FreqPoly de QSEC", x = "Temps en quarts de milla (qsec)",y = "Densitat")















