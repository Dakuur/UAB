n = 5
positius = 0
negatius = 0

while n > 0:
    num = int(input())
    if num > 0:
        positius = positius + 1
    elif num < 0:
        negatius = negatius + 1
    n = n - 1

print("Positius:", str(positius), "-", "Negatius: ", negatius)