vocal =str.upper(input())
trans = {"A" : ".-", "E" : ".", "I" : "..", "O" : "---", "U" : "..-"}
if vocal in trans:
    morse =  trans[vocal]
    print("Vocal:", str(vocal), "-", "Codi Morse:", str(morse))
else:
    print("Error: El caràcter introduït no és una vocal")