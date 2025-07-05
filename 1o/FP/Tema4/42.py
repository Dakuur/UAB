adivinar = "python"
print("La paraula oculta té", len(adivinar),"lletres:")

guions = ""
for i in range(0, len(adivinar)):
    guions = guions + "-"
print(guions)

intents = 10
acabat = False

while intents > 0 and acabat == False:
    guess = str(input("Introdueix una lletra: "))
    for i in range(0, len(adivinar)): #recorrer tota la paraula
        if guess == adivinar[i]:
            abans = guions[:i]
            despres = guions[i+1:]
            guions = str(abans) + str(guess) + str(despres)
    
    intents = intents - 1

    if "-" not in guions:
        acabat = True
        print(guions, "Et queden",intents,"intents")
        print("Enhorabona, l'has encertat!!!")
    else:
        print(guions, "Et queden",intents,"intents")

if intents == 0:
    print("Ho sento, has perdut, la paraula era:", adivinar)