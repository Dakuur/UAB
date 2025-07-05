library(tidyverse)
library(dplyr)

select(starwars, name, gender)

# Humans only
filter(starwars, species=="Human")
starwars %>% filter(species=="Human")

# Not humans only
filter(starwars, species!="Human")
starwars %>% filter(!species=="Human")

fem <- starwars%>%filter(gender=="feminine")
fem <- select(fem, name, gender)

arrange(fem, name)
arrange(fem, desc(name))
