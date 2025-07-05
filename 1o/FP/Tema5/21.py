'''
def divisio(divident, divisor):
    quocient = 0
    while divident >= divisor:
        divident = divident - divisor
        quocient = quocient + 1
    residu = divident
    return quocient, residu

divident = int(input())
divisor = int(input())

quocient, residu = divisio(divident, divisor)

print("Quocient:",quocient ,"i Residu:", residu)

'''

def divisio(dividend, divisor):
    if divisor == 0:
        return 1

    sign = -1 if ((dividend < 0) != (divisor < 0)) else 1
    dividend = abs(dividend)
    divisor = abs(divisor)
    quotient = 0
    while dividend >= divisor:
        quotient += 1
        dividend -= divisor

    return sign * quotient, dividend

divident = int(input())
divisor = int(input())

if divisor == 0:
    print("Error: Divisió per zero")
else:    
    quocient, residu = divisio(divident, divisor)

    print("Quocient:",quocient ,"i Residu:", residu)