llista1=[]
llista2=[]

for i in range(0,6):
    llista1.append(float(input()))
for i in range(0,6):
    llista2.append(float(input()))

iguals = True

for i in range(0,6):
    if llista1[i] != llista2[i]:
        iguals = False
        break

if iguals == True:
    print("IGUALS")
else:
    print("DIFERENTS")