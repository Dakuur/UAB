from time import sleep

hora = str(input())

hh = int(hora[0:2])
mm = int(hora[3:5])
ss = int(hora[6:8])

segons = 5

while segons > 0:
    ss = int(ss) + 1

    if ss == 60:
        ss = 0
        mm = int(mm) + 1
    if mm == 60:
        mm = 0
        hh = int(hh) + 1
    if hh == 24:
        hh = 0

    if int(hh) < 10:
        hh = str("0"+str(int(hh)))
    if int(mm) < 10:
        mm = str("0"+str(int(mm)))
    if int(ss) < 10:
        ss = str("0"+str(int(ss)))


    print(str(hh)+":"+str(mm)+":"+str(ss))

    sleep(1)
    segons = segons - 1