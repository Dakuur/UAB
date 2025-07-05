from ClassePoint import Point

class Rectangle:
    def __init__(self, origin, width, height):
        self.origin = origin
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height
    def perimeter(self):
        return 2*(self.width+self.height)
    def is_square(self):
        if self.width == self.height:
            return True
        else:
            return False
    def zoom(self, multiplicador):
        return Rectangle(self.origin, self.width*multiplicador, self.height*multiplicador)
    def move(self, x, y):
        return Rectangle(self.origin + Point(x,y), self.width, self.height)
    def get_vertex(self):
        llista = []
        llista.append(self.origin)
        llista.append(Point(self.origin.x + self.width, self.origin.y))
        llista.append(Point(self.origin.x + self.width, self.origin.y + self.height))
        llista.append(Point(self.origin.x, self.origin.y + self.height))
        return llista
    def contains(self, punt):
        if (self.origin.x <= punt.x < self.origin.x + self.width) and (self.origin.y <= punt.y < self.origin.y + self.height):
            return True
        else:
            return False
    def overlap(self, rectangle):
        vertexes_1 = self.get_vertex()
        vertexes_2 = rectangle.get_vertex()
        for v in vertexes_1:
            if rectangle.contains(v):
                return True
        for v in vertexes_2:
            if self.contains(v):
                return True
        return False