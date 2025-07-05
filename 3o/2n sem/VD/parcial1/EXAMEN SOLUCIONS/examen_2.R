titanic <- read.csv("C:/Users/david/Downloads/titanic.csv")

titanic <- titanic%>%mutate(Survived_str = ifelse(Survived == 1, "Yes", "No"))

ggplot(titanic, aes(x=factor(Pclass), fill = factor(Survived_str))) +
  geom_bar() + labs(title="Passangers per class", x="Class", y="Count", )

ggplot(titanic, aes(x=factor(Pclass), fill = factor(Survived_str))) +
  geom_bar(position = "fill") +
  labs(title="Safety per class", x="Class", y="Survive rate", fill="Survived")

?labs








ss <- read.csv("C:/Users/david/Downloads/simpsons_episodes.csv")
