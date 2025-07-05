library(tidyverse)


data <- mtcars %>%
  group_by(cyl) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

# Crear el pie chart
ggplot(data, aes(x = "", y = percentage, fill = factor(cyl))) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  labs(fill = "Cilindros", title = "Distribución de Cilindros en mtcars") +
  theme_void()

?starwars

ggplot(starwars,aes(x=height, fill=factor(gender))) +
  geom_histogram(binwidth = 10) +
  labs(title="Star Wars Characters", x="Height", y="Count") +
  scale_fill_discrete("Gender") +
  theme_minimal() + coord_flip()

ggplot(starwars, aes(x=factor(gender), y=height)) + geom_violin()

ggplot(starwars, mapping = aes(x = fct_infreq(eye_color))) +
  geom_bar() +
  coord_flip()


ggplot(starwars,aes(x=height)) +geom_density()

ggplot(starwars,aes(x=height))+geom_histogram(binwidth=10, aes(y=..density..))+
  geom_density(lwd = 2, colour = 'blue', fill = 'blue', alpha= 0.25)

?geom_density
