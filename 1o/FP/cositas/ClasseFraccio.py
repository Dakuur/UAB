class Fraccio:
  def __init__(self, numerator, denominator):
    self.numerator = numerator
    self.denominator = denominator

  def __add__(self, other):
    numerator = self.numerator * other.denominator + self.denominator * other.numerator
    denominator = self.denominator * other.denominator
    return Fraccio(numerator, denominator)

  def __sub__(self, other):
    numerator = self.numerator * other.denominator - self.denominator * other.numerator
    denominator = self.denominator * other.denominator
    return Fraccio(numerator, denominator)

  def __str__(self):
    return f"{self.numerator}/{self.denominator}"

  def frac2float(self):
    return self.numerator / self.denominator

  def inv(self):
    return Fraccio(self.denominator, self.numerator)
