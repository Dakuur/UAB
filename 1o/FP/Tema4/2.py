mesos = ("gener", "febrer", "març", "abril", "maig", "juny", "juliol", "agost", "setembre", "octubre", "novembre", "desembre")
#DD/MM/AAAA

data = str(input())

dia = int(data[0:2])
num_mes = int(data[3:5]) - 1

if num_mes not in range(0,13):
    print("Error: Mes incorrecte")
else:
    mes = mesos[num_mes]
    any = data[6:11]
    print(dia, "de", mes, "de", any)