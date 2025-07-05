N = int(input())

num = 0
exponent = 0

llista = str("")

while num < 2**N:
    num = 2**exponent
    llista = llista + " " + str(num)
    exponent = exponent + 1

print(llista[1:])