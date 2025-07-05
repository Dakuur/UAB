from funcions import area_quadrat, area_rectangle, area_triangle
from menu import menu_seleccio

menu_seleccio()

menu=int(input(menu_seleccio()))

while menu !=4:
    if menu == 1:
        costat = float(input("Introdueix la longitud del costat: "))
        valor,area = area_quadrat(costat)
        if valor == 1:
            print("Error: Dimensions incorrectes")
        if valor == 0:
            print("Àrea:",area)
        menu=int(input(menu_seleccio()))
    if menu == 2:
        base = float(input("Introdueix la longitud de la base: "))
        altura = float(input("Introdueix la longitud de l'altura: "))
        valor,area = area_rectangle(base,altura)
        if valor == 1:
            print("Error: Dimensions incorrectes")
        if valor == 0:
            print("Àrea:",area)
        menu=int(input(menu_seleccio()))
    if menu == 3:
        base = float(input("Introdueix la longitud de la base: "))
        altura = float(input("Introdueix la longitud de l'altura: "))
        valor,area = area_triangle(base,altura)
        if valor == 1:
            print("Error: Dimensions incorrectes")
        if valor == 0:
            print("Àrea:",area)
        menu=int(input(menu_seleccio()))
    if menu < 1 or menu > 4:
        print("Error: Opció incorrecta")
        menu=int(input(menu_seleccio()))
        
    
if menu < 1 or menu > 4:
    print("Error: Opció incorrecta")