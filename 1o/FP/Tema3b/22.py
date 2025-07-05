num = int(input())
preu = float(input())

noiva = num*preu
iva = noiva*1.07

if iva < 500:
    total = iva
elif iva < 1000:
    total = iva*0.95
else:
    total = iva*0.9

print("L'import final de la compra són", str(total), "euros.")