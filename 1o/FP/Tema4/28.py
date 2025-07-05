entrada = str(input())

a=e=i=o=u = 0


for x in range(0, len(entrada)):
    if entrada[x] == "a" or entrada[x] == "A":
        a=a+1
    elif entrada[x] == "e" or entrada[x] == "E":
        e=e+1
    elif entrada[x] == "i" or entrada[x] == "I":
        i=i+1
    elif entrada[x] == "o" or entrada[x] == "O":
        o=o+1
    elif entrada[x] == "u" or entrada[x] == "U":
        u=u+1

print("A: "+str(a)+" - E: "+str(e)+" - I: "+str(i)+" - O: "+str(o)+" - U: "+str(u)+"")