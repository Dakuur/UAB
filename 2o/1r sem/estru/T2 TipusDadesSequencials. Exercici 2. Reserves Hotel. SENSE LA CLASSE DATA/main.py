import ReservesHotel
import datetime


def mostraBool(logic):
	if (logic):
		print("TRUE")
	else:
		print("FALSE")

def testNReservesDia(reserves):
    reduccio = 0.0
    dataTest =[ datetime.datetime(2018,1,1),datetime.datetime( 2018,1,2 ),datetime.datetime( 2018,1,3 ), datetime.datetime( 2018,1,4 ), datetime.datetime( 2018,1,5 ),datetime.datetime( 2018,1,6 ),datetime.datetime( 2017,12,31 )]
    nHabitacionsTest=[2,3,8,8,2,0,0]

    print("Comment :=>> Iniciant test metode NRESERVESDIA" )
    print("Comment :=>> =================================" )

    i=0
    for p in dataTest:
        print("Comment :=>> ------------------------------------------" )
        print("Comment :=>> TEST ", i+1 )
        print("Comment :=>> Data: ", dataTest[i].day , "/", dataTest[i].month , "/", dataTest[i].year )
        print("Comment :=>> ---" )
        print("Comment :=>> Valor retorn esperat: " , nHabitacionsTest[i] )
        print("Comment :=>> ---" )
        nHabitacions = reserves.nReservesDia(dataTest[i])
        print("Comment :=>> Valor retorn obtingut: " , nHabitacions )
        if (nHabitacions == nHabitacionsTest[i]):
            print("Comment :=>> CORRECTE" )
        else:
            print("Comment :=>> ERROR" )
            reduccio += 1.0
        i+=1
	
    if (reduccio > 3.0):
        reduccio = 3.0
    return reduccio



def testAfegeixReserva(reserves):
	reduccio = 0.0
	NPROVES = 5
	nomTest = [ "client_a", "client_b", "client_c", "client_d", "client_e" ]
	dataTest=[datetime.datetime( 2018,1,1 ),datetime.datetime( 2018,1,2 ),datetime.datetime( 2018,1,3 ),datetime.datetime(2018,1,4 ),datetime.datetime( 2018,1,5 )]
	nDiesTest=[ 2, 3, 2, 4, 2 ]
	nHabitacionsTest = [ 2, 3, 1, 3, 2 ]
	validTest = [ True, False, True, False, True ]

	print("Comment :=>> Iniciant test metode AFEGEIXRESERVA" )
	print("Comment :=>> ===================================" )

	for p in range (0,NPROVES):
		print("Comment :=>> ------------------------------------------" )
		print("Comment :=>> TEST " , p + 1 )
		print("Comment :=>> Nom Client: " , nomTest[p] )
		print("Comment :=>> Data Entrada: ", dataTest[p].day , "/" , dataTest[p].month,"/",dataTest[p].year)
		print("Comment :=>> Num. dies: " , nDiesTest[p] )
		print("Comment :=>> Num. habitacions: " , nHabitacionsTest[p] )
		print("Comment :=>> ---" )
		print("Comment :=>> Valor retorn esperat: ", validTest[p])
		print("Comment :=>> ---" )
		valid = reserves.afegeixReserva(nomTest[p], dataTest[p], nDiesTest[p], nHabitacionsTest[p]);
		print("Comment :=>> Valor retorn obtingut: ", mostraBool(valid))
		if (valid == validTest[p]):
			print("Comment :=>> CORRECTE" )
		else:
			print("Comment :=>> ERROR" )
			reduccio += 1.0
		
	if (reduccio > 3.0):
		reduccio = 3.0
	return reduccio


def testConsultaReserva(reserves):
    reduccio = 0.0
    NPROVES = 7

    nomTest = [ "CLIENT_2", "client_a", "CLIENT_7", "client_b", "CLIENT_8", "CLIENT_9", "CLIENT_2"]
    dataEntradaTest =[datetime.datetime( 2018,1,2 ),datetime.datetime( 2018,1,1 ),datetime.datetime( 2018,12,1 ),datetime.datetime( 2018,1,2 ),	datetime.datetime(  2018,6,1 ),datetime.datetime( 2018,6,1 ),datetime.datetime( 2018,6,1 )]
    dataSortidaTest = [datetime.datetime( 2018,1,3 ),datetime.datetime(  2018,1,3 ),datetime.datetime( 2018,12,3 ),datetime.datetime( 2018,1,1 ),datetime.datetime( 2018,6,3 ),datetime.datetime( 2018,1,1 ),datetime.datetime( 2018,1,1 )]
    	
    nHabitacionsTest=[ 1, 2, 6, 0, 6, 0, 0 ]
    preuTest = [100.0, 400.0, 1200.0, 0, 1200.0, 0, 0]
    validTest = [ True, True, True, False, True, False, False ]

    print("Comment :=>> Iniciant test metode CONSULTARESERVA" )
    print("Comment :=>> (Assumeix funcionament correcte de llegir reserves i afegir reserves)" )
    print("Comment :=>> =====================================================================" )

    for p in range(0, NPROVES):
        print("Comment :=>> ------------------------------------------" )
        print("Comment :=>> TEST " , p + 1 )
        print("Comment :=>> Nom Client: " , nomTest[p] )
        print("Comment :=>> Data Entrada: ", dataEntradaTest[p])
        print("Comment :=>> ---" )
        print("Comment :=>> Valor retorn esperat: ",validTest[p])
        if (validTest[p]):
            print("Comment :=>> Data sortida esperada: " , dataSortidaTest[p] )
            print("Comment :=>> Num. habitacions esperat: " , nHabitacionsTest[p] )
            print("Comment :=>> Preu esperat: ", preuTest[p] )
        print("Comment :=>> ---" )
        valid,dataSortida, nHabitacions, preu = reserves.consultaReserva(nomTest[p], dataEntradaTest[p])
        print("Comment :=>> Valor retorn obtingut: ", valid)
        if (valid):
            print("Comment :=>> Data sortida obtinguda: " , dataSortida )
            print("Comment :=>> Num. habitacions obtingut: ", nHabitacions )
            print("Comment :=>> Preu obtingut: " , preu )
        if (valid == validTest[p]):
            if (valid):
                if ((dataSortida == dataSortidaTest[p]) and (nHabitacions == nHabitacionsTest[p]) and (abs(preu - preuTest[p]) < 0.1)):
                    print("Comment :=>> CORRECTE" )
                else:
                    print("Comment :=>> ERROR" )
                    reduccio += 1.0
        else:
            print("Comment :=>> ERROR" )
            reduccio += 1.0		
    if (reduccio > 4.0):
        reduccio = 4.0
    return reduccio

grade = 10.0

reserves=ReservesHotel.ReservesHotel("hotel_1", 100.0, 10)

print("Comment :=>> Cridant al constructor de ReservesHotel per inicialitzar dades de l'hotel....." )
print("Comment :=>> ==============================================================================" )
print("Comment :=>> Nom de l'hotel: hotel_1" )
print("Comment :=>> Preu per dia: 100.0" )
print("Comment :=>> Num. habitacions: 10" )

print("Comment :=>> Llegint reserves del fitxer 'reserves1.txt'....." )
print("Comment :=>> ================================================" )


reserves.llegeixReserves("reserves1.txt")


grade -= testNReservesDia(reserves)


grade -= testAfegeixReserva(reserves)

print("Comment :=>> Llegint reserves del fitxer 'reserves2.txt'....." )
print("Comment :=>> ================================================" )

reserves.llegeixReserves("reserves2.txt")

grade -= testConsultaReserva(reserves)

if (grade < 0):
    grade = 0.0
print("Comment :=>> ------------------------------------------" )
if (grade == 10.0):
    print("Comment :=>> Final del test sense errors" )
print("Grade :=>> " , grade )
