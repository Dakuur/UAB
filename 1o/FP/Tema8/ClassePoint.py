from math import sqrt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_origen(self):
        return sqrt(self.x**2 + self.y**2)

    def distance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def midpoint(self, other):
        x = (self.x + other.x) / 2
        y = (self.y + other.y) / 2
        return Point(x, y)

    def slope(self, other):
        return (other.y - self.y) / (other.x - self.x)

    def __str__(self):
        return f"({self.x},{self.y})"