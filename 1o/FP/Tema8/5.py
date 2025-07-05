class Point():
    def __init__(self,x_ini,y_ini):
        self.x = x_ini
        self.y = y_ini

    def __str__(self):
        return "("+str(self.x)+", "+str(self.y)+")"

p1 = Point(5,5)
p2 = Point(10,10)

print(p1)