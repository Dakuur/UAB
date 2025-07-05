N = int(input())
L = int(input())

def modul(N,L):
    num = 1
    llista = ""
    while num*N < L:
        num = num * N
        llista = llista + " " + str(num)
    return llista
    
llista = modul(N, L)

print(llista[1:])