any = int(input())

edat = 2022 - any

if edat < 18:
    print("Tens", edat ,"anys i ets menor d'edat.", end = '')
    if edat <= 12:
        print(" Encara no has acabat primària.")
    else:
        print(" Has acabat primària")
else:
    print("Tens", edat ,"anys i ets major d'edat.", end = '')
    if edat < 67:
        print(" Estàs en edat de treballar.")
    else:
        print(" Estàs en edat de jubilació.")