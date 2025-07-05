arxiu = open("arxiu.txt", "w")

arxiu.write("Hola\nBenvinguts al curs\nIntroducció al Python\n")

arxiu.close

arxiu = open("arxiu.txt", "a")

arxiu.write("Espero que us agradi\n")

arxiu.close

arxiu = open("arxiu.txt", "r")

print(arxiu.read())