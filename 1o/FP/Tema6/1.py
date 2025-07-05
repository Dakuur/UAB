uno = open("arxiu.txt","w")

uno.write("Hola\nBenvinguts al curs\nEspero que us agradi")

uno.close

uno = open("arxiu.txt","r")
print(uno.read())
uno.close

uno = open("arxiu.txt","r")
for i in range(0,3):
    print(i+1,uno.readline())
uno.close

uno = open("arxiu.txt","r")
for i in range(0,3):
    print(uno.readline())
uno.close