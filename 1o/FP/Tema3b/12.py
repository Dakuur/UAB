num1 = int(input())
num2 = int(input())

if num1 > num2:
    tmp2 = num2
    num2 = num1
    num1 = tmp2

print("El valor de num1 és " + str(num1) + " i el valor de num2 és " + str(num2))