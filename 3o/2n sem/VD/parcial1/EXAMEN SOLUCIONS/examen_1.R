install.packages("dplyr", lib="C:/Users/david/R/library")
install.packages("ggplot2")
install.packages("tidyverse")

df <- read.csv("C:/Users/david/Downloads/titanic.csv")
head(df, 10)

library(ggplot2)
library(dplyr)


mean(df$Survived == 1)

df <- df %>% mutate(age_int = round(Age))

df <- df %>%
  mutate(Survived = ifelse(Survived == 1, "Yes", "No"))

ggplot(df, aes(x=age_int, fill = factor(Survived))) + geom_bar()
    
ggplot(df, aes(x = age_int, fill = factor(Survived))) + 
    geom_bar(position = "dodge") +
    #scale_fill_manual(values = c("Yes" = "green", "No" = "red"))
    labs(fill = "Survived") + 
    xlab("Edat") +
    ylab("Quantitat de persones") +
    theme_minimal()

    
ggplot(df, aes(x = age_int, fill = factor(Sex))) + 
  geom_bar(position = "dodge") +
  #scale_fill_manual(values = c("male" = "light_blue", "female" = "red"))
  labs(fill = "Sexe") + 
  xlab("Edat") +
  ylab("Quantitat de persones") +
  theme_minimal()

ggplot(df, aes(x=factor(Pclass), fill = factor(Survived))) + geom_bar()
ggplot(df, aes(x=factor(Pclass), fill = factor(Survived))) + geom_bar(position = "dodge")
ggplot(df, aes(x=factor(Pclass), fill = factor(Survived))) + geom_bar(position = "fill")

ggplot(df, aes(x = factor(Pclass), fill = factor(Survived))) + 
  geom_bar(position = "fill") +  # "fill" normaliza las barras a proporciones
  scale_fill_manual(values = c("Yes" = "blue", "No" = "red")) +  # Colores personalizados
  labs(x = "Clase (Pclass)", 
       y = "Proporción de pasajeros", 
       fill = "Sobrevivió") +
  theme_minimal() +
  ggtitle("Proporción de pasajeros sobrevivientes por clase (Pclass)") +
  theme(plot.title = element_text(hjust = 0.5))






install.packages("IRkernel")
IRkernel::installspec(user = TRUE)






acc <- read.csv("C:/Users/david/Downloads/2017_accidents_vehicles_gu_bcn_.csv")


hour <- acc$Hora_dia

# Histogram of Hour
ggplot(acc, aes(x=as.numeric(hour))) +
  geom_histogram(binwidth=1, color="darkblue", fill="lightblue")

?geom_histogram












ss <- read.csv("C:/Users/david/Downloads/simpsons_episodes.csv")

ss <- ss %>% drop_na(imdb_rating)
  
ggplot(ss, aes(x=as.Date(original_air_date),
           y=imdb_rating)) +geom_point() + geom_smooth(method = "lm") +labs(title=paste("Els
Simpsons"),x="Any d’emissió original", y="Puntuació")

ss3 <- ss%>%filter(season<4)

ggplot(ss3, aes(x=as.Date(original_air_date),
               y=imdb_rating)) +geom_point(color=ss3$season) +
               geom_smooth(method = "loess") +labs(title=paste("Els
Simpsons"),x="Any d’emissió original", y="Puntuació")

ggplot(ss3, aes(x=as.Date(original_air_date), y=imdb_rating,)) +geom_point() +
  geom_smooth(level=0.85) +
  xlab('Any d’emissió original') + ylab('Puntuació') +
  facet_grid(season~ .) + ggtitle('Els Simpsons')+
  scale_color_discrete(name = "Temporada", labels=c("S1", "S2", "S3"))

ss10 <- ss%>%filter(season<11)

ggplot(ss10, aes(x=factor(season), y=imdb_votes)) + geom_boxplot() +
  xlab("season") + ylab("IMDB votes")

ggplot(ss10, aes(x=factor(season), y=imdb_votes)) + geom_violin(fill = "blue") +
  xlab("season") + ylab("IMDB votes")



bart <- filter(ss, grepl("Bart", title))
print("Numero d'episodis amb Bart:")
print(nrow(bart))

ggplot(bart, aes(views)) + geom_density()

?filter

?geom_violin
