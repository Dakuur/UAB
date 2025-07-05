def duesXifres(n):
    if n < 10:
        n = str("0"+str(n))
    return(n)

def incrementSegon(hh,mm,ss):
    ss = ss + 1
    if ss == 60:
        ss = 0
        mm = mm + 1
    if mm == 60:
        mm = 0
        hh = hh + 1
    if hh == 24:
        hh = 0
    return hh,mm,ss

hora = str(input()) #format hh:mm:ss

hh = int(hora[0:2])
mm = int(hora[3:5])
ss = int(hora[6:8])

hh,mm,ss=incrementSegon(hh,mm,ss)

hh=duesXifres(hh)
mm=duesXifres(mm)
ss=duesXifres(ss)

print(str(hh)+":"+str(mm)+":"+str(ss))