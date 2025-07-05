from time import sleep

def Alerta(n):
    while n != 0:
        print("Alerta: Queden", n, "segons")
        sleep(1)
        n = n -1
    print("Alerta: S'ha acabat el temps")

segons = int(input())

Alerta(segons)