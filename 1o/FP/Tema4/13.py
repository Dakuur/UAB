dies = ("dilluns", "dimarts", "dimecres", "dijous", "divendres", "dissabte", "diumenge")

num = int(input())

while num != 0:
    if num in range(1, len(dies)+1):
        print(dies[num-1])
        num = int(input())
    else: 
        print("Error: Dia incorrecte")
        num = int(input())