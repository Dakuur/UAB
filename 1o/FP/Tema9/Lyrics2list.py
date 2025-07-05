def Lyrics2list(lletra):
    llist=[]
    for i in lletra:
        if (i>="A" and i<="Z"):
            llist.append(chr(ord(i)+32))
        elif (i>="a" and i<="z" or i==" "):
            llist.append(i)
    filtre="".join(llist)
    return(filtre.split(" "))