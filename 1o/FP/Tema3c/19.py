n = int(input())
divisor = n-1

i = True

while i == True:

    if n%divisor == 0:
        print("NO PRIMER")
        i = False
    else:
        if divisor == 1:
            i = False
            print("PRIMER")
    divisor = divisor - 1
    




''''
for divisor in range(1, n):

    if n%divisor == 0:
        print("NO PRIMER")
        divisor = 1

    divisor = divisor - 1

if divisor == 1:
    print("PRIMER")
'''