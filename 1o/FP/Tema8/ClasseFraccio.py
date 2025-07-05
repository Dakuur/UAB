class Fraccio:
  def __init__(self, numerador, denominador):
    self.numerador = numerador
    self.denominador = denominador

  def __add__(self, altrefraccio):
    numerador = self.numerador * altrefraccio.denominador + self.denominador * altrefraccio.numerador
    denominador = self.denominador * altrefraccio.denominador
    return Fraccio(numerador, denominador)

  def __sub__(self, altrefraccio):
    numerador = self.numerador * altrefraccio.denominador - self.denominador * altrefraccio.numerador
    denominador = self.denominador * altrefraccio.denominador
    return Fraccio(numerador, denominador)

  def __str__(self):
    return f"{self.numerador}/{self.denominador}"

  def frac2float(self):
    return self.numerador / self.denominador

  def inv(self):
    return Fraccio(self.denominador, self.numerador)