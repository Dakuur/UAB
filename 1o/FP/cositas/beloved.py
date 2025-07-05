#BELOVED
#0123456

import random

lista0 = ["b", "w", "g", "p", "m"]
lista1 = ["e", "a", "i", "o", "u"]
lista2 = ["l", "w", "b", "m"]
lista3 = ["o", "a", "i", "e", "u"]
lista4 = ["v", "w", "b", "l", "m"]
lista5 = ["e", "a", "i", "o", "u"]
lista6 = ["d", "t", "p", "p", "f"]

beloved = open("beloved.txt","w")

for i in range(0, len(lista0)):
    for j in range(0, len(lista1)):
        for k in range(0, len(lista2)):
            for l in range(0, len(lista3)):
                for m in range(0, len(lista4)):
                    for n in range(0, len(lista5)):
                        for o in range(0, len(lista6)):
                            beloved.write(lista0[i]+lista1[j]+lista2[k]+lista3[l]+lista4[m]+lista5[n]+lista6[o]+"\n")

beloved.close