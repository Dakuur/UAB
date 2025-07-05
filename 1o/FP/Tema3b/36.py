data = str(input()) #DD/MM/AAAA

dia = (data[0:2])
mes = (data[3:5])
any = (data[6:10])

if int(mes) in range(1,13):

    diesfebrer = 28

    if int(any)%4 == 0:
        diesfebrer = 29
    if int(any)%100 == 0:
        diesfebrer = 28
    if int(any)%400 == 0:
        diesfebrer = 29

    diapermes = {
        "01" : 31,
        "02" : int(diesfebrer),
        "03" : 31,
        "04" : 30,
        "05" : 31,
        "06" : 30,
        "07" : 31,
        "08" : 31,
        "09" : 30,
        "10" : 31,
        "11" : 30,
        "12" : 31,
    }

    if int(dia) in range(1, diapermes[mes]+1):
        print("Data correcta")
    else:
        print("Error: Dia incorrecte")

#    print(dia,mes,any)

else:
    print("Error: Mes incorrecte")