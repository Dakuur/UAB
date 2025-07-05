max = int(input())

primers = []
no_primers = []

for num in range(2, max+1):
    if num not in no_primers:
        primers.append(num)
    resultat = num
    i = 2
    while resultat <= max:
        no_primers.append(resultat)
        resultat = num*i
        i = i+1
print(primers)
#print(no_primers)