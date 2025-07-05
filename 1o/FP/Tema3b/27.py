num1 = int(input())

num2 = int(input())

if num2 > num1:
    num1 = num2

num3 = int(input())

if num3 > num1:
    num1 = num3

num4 = int(input())

if num4 > num1:
    num1 = num4

print("El número més gran de la sèrie és" , str(num1) + ". Comparacions: 3")