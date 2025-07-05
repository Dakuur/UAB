from dataclasses import dataclass

"""@dataclass
class Persona:
    def __init__(self, nombre, edad):
        self._nombre = nombre
        self._edad = edad
    
    # Getter para el atributo nombre
    @property
    def nombre(self):
        return self._nombre
    
    # Setter para el atributo nombre
    @nombre.setter
    def nombre(self, nuevo_nombre):
        self._nombre = nuevo_nombre
    
    # Getter para el atributo edad
    @property
    def edad(self):
        return self._edad
    
    # Setter para el atributo edad
    @edad.setter
    def edad(self, nueva_edad):
        self._edad = nueva_edad
        
# Crear una instancia de Persona
persona1 = Persona("Juan", 25)

# Obtener el valor del atributo nombre usando el getter
print(persona1.nombre) # Salida: "Juan"

# Modificar el valor del atributo nombre usando el setter
persona1.nombre = "Pedro"

# Obtener el valor actualizado del atributo nombre
print(persona1.nombre) # Salida: "Pedro"""

@dataclass
class Dog:
    _age: int = 0
    _color: str = "hola"

    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, new_value):
        self._age = new_value

    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, new_value):
        self._color = new_value

doggo = Dog( "brown")

print(doggo)

print(doggo.age)
print(doggo.color)

doggo.color = "pink"
doggo.age = 23190864

print(doggo)

print(doggo.age)
print(doggo.color)