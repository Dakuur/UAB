library(tidyverse)

str(mtcars)

mtcars <- mtcars%>%mutate(am_str = ifelse(am == 0, "Automatic", "Manual"))

#ggplot(mtcars, aes(x=factor(cyl), y=mpg, color=factor(am_str))) +
ggplot(mtcars, aes(x=factor(cyl), y=mpg, fill = factor(cyl))) +
  geom_violin() +
  #geom_point(size = mtcars$hp/50) + 
  #geom_point() + 
  theme_minimal()



ggplot(mtcars, aes(x=factor(cyl), y=mpg, color=factor(am_str))) +
  geom_point() +
  scale_color_discrete("Transmission type", labels = c("Automatic","Manual")) +
  labs(title = "MTcars") +
  scale_x_discrete("Cylinders")+scale_y_continuous ("Miles per gallon(US)") +
  theme_minimal()

mtcars2 <- within(mtcars, {
  vs <- factor(vs, labels = c("V", "S"))
  cyl <- factor(cyl)
})


ggplot(mtcars, aes(mpg, disp, color=factor(cyl)))+geom_point()+
  scale_x_continuous("Miles per gallon(US)")+
  scale_y_continuous("Displacement")+scale_color_discrete("Cylinders")
# Eixos --> Variables contínues
# Color --> Variables discretes



ggplot(mtcars, aes(mpg, disp, color=factor(cyl)))+geom_point()+
  scale_x_continuous("Miles per gallon(US)")+
  scale_y_continuous("Displacement")+scale_color_discrete("Cylinders") +
  geom_smooth(level=0.95)

ggplot(mtcars, aes(mpg, disp))+geom_point()+
  scale_x_continuous("Miles per gallon(US)")+
  scale_y_continuous("Displacement")+scale_color_discrete("Cylinders") +
  geom_smooth(level=0.95)


ggplot(mtcars, aes(mpg, disp, shape = factor(am_str)))+geom_point()+
  scale_x_continuous("Miles per gallon(US)")+
  scale_y_continuous("Displacement")+scale_color_discrete("Cylinders")

?geom_smooth






anscombedata=with(anscombe,data.frame(xVal=c(x1,x2,x3,x4),
                                      yVal=c(y1,y2,y3,y4), anscombegroup=gl(4,nrow(anscombe))))
ggplot(anscombedata,aes(x=xVal,y=yVal,group=anscombegroup))+
  geom_point()+facet_wrap(~anscombegroup)
