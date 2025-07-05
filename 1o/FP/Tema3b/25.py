teoria = float(input())
seminaris = float(input())
practiques = float(input())

nota = 0.4*teoria + 0.3*seminaris + 0.3*practiques

if nota < 5:
    acta = "SUSPES"
elif nota < 7:
    acta = "APROVAT"
elif nota < 9:
    acta = "NOTABLE"
elif nota < 10:
    acta = "EXCEL.LENT"
else:
    acta = "MATRICULA D'HONOR"

print("La nota final és", nota , "-", acta)