hora = str(input()) #format hh:mm:ss

hh = int(hora[0:2])
mm = int(hora[3:5])
ss = int(hora[6:8])

ss = ss + 1

if ss == 60:
    ss = 0
    mm = mm + 1
if mm == 60:
    mm = 0
    hh = hh + 1
if hh == 24:
    hh = 0

if hh < 10:
    hh = str("0"+str(hh))
if mm < 10:
    mm = str("0"+str(mm))
if ss < 10:
    ss = str("0"+str(ss))


print(str(hh)+":"+str(mm)+":"+str(ss))