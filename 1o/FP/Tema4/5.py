num = int(input())

llista = []

missatge = "No hi ha nombre parell."

while num != 0:
    if num%2 == 0:
        missatge = "Hi ha nombre parell."
    llista.append(num)
    num = int(input())

print(missatge)