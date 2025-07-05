def Lyrics2list(lletra):
    llist=[]
    for i in lletra:
        if (i>="A" and i<="Z"):
            llist.append(chr(ord(i)+32))
        elif (i>="a" and i<="z" or i==" "):
            llist.append(i)
    filtre="".join(llist)
    return(filtre.split(" "))

def Lyrics2frequencies(llista):
    llista = list(map(lambda x: x, llista))
    dicc = dict()
    for i in llista:
        if i not in dicc:
            dicc[i] = 1
        else:
            dicc[i] += 1
    return dicc

def Most_common_words(frequencies):
    maxim = frequencies[max(frequencies, key = lambda x: frequencies[x])]
    mesrepetides = list(filter(lambda x: frequencies[x] == maxim, frequencies))
    return mesrepetides, maxim