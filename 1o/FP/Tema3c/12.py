num = int(input())
times = 0

if num < 0:
    print("No hi ha nombres a la seqüència.")
else:
    max = num
    min = num
    while num >= 0 and times < 9:
        if num > max:
            max = num
        if num < min:
            min = num
        times = times + 1
        num = int(input())
        
    print("Màxim:", max ,"-", "Mínim:", min)