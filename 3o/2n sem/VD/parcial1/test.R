# Cargar ggplot2
library(ggplot2)

diamonds

ggplot(data = diamonds, aes(x=carat, y=price, color=factor(cut))) +
  geom_point(alpha=0.25) +
  labs(x="Carats",y="Price")

mpg

ggplot(data = mpg, aes(x = factor(drv), y = cty, fill = drv)) +
  geom_boxplot() +
  labs(x = "Tracción", y = "Consumo en ciudad", title = "Consumo en ciudad según el tipo de tracción") +
  theme_minimal()

mtcars

ggplot(data=mtcars, aes(x=factor(cyl), fill=factor(cyl))) +
  geom_bar(width = .5) +
  labs(x="Número de cilindros", title="Cantidad de vehículos según número de cilindros") +
  theme_minimal()

?mpg

library(ggplot2)

ggplot(data = mpg, aes(x = factor(displ), fill = factor(drv))) +
  geom_bar() +
  theme_minimal() +
  labs(x = "Desplazamiento del motor (displ)", y = "Frecuencia", fill = "Tracción", title = "Distribución del desplazamiento por tipo de tracción")

ggplot(mpg, aes(x = cut(displ, breaks = 5), fill = factor(drv))) +
  geom_bar() +
  theme_minimal() +
  labs(x = "Rangos de desplazamiento", y = "Frecuencia", fill = "Tracción", title = "Distribución por rango de desplazamiento")

str(mpg)

?mtcars

ggplot(data=mtcars, aes(x=cut(wt, breaks=5), fill=factor(am))) +
  geom_bar() +
  labs(x="Peso del vehículo", title="Distribución de tipo según peso") +
  theme_minimal()


mtcars2 <- within(mtcars, {am <- factor(am, labels = c("automatic", "manual"))})

ggplot(mtcars2, aes(cyl, fill=am))+
  geom_bar(position="dodge")+xlab("Number of Cylinders")+ylab("Count")


?mtcars

ggplot(mtcars, aes(x=hp)) +
  geom_histogram()

cars





