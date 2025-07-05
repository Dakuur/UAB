-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
----------                                                                       ----------
----------      AQUEST DOCUMENT HA ESTAT CREAT AMB LA FINALITAT D'APRENDRE!      ----------
----------         COPIAR I ENGANXAR LES CONSULTES ET POT FER SUSPENDRE          ----------
----------   AFEGEIX LES NOVES CONSULTES AL FINAL DEL DOCUMENT SEGUINT L'ESTIL   ----------
----------                        I SOBRETOT, COMPARTEIX-LO!                     ----------
----------                         _______ _____  _   _                          ----------
----------                        |__   __|  __ \| \ | |                         ----------
----------                           | |  | |__) |  \| |                         ----------
----------                           | |  |  ___/| . ` |                         ----------
----------                           | |  | |    | |\  |                         ----------
----------                           |_|  |_|    |_| \_|                         ----------
----------                                                                       ----------
--------------------------------------------------------------------------12/01/2019-------                 
-------------------------------------------------------------------------------------------

/*1. Numero de passatgers (com a N_Passatgers) amb cognom ìBlancoî que tÈ el vol amb codi IBE2119 i data 30/10/13.*/
SELECT COUNT(P.NOM) AS N_PASSATGERS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER 
  AND B.DATA = TO_DATE('30/10/2013', 'DD/MM/YYYY') 
  AND B.CODI_VOL = 'IBE2119' 
  AND P.NOM LIKE '%Blanco%'
ORDER BY 1;

/*2. Numero de butaques reservades (com a N_Butaques_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.*/
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM BITLLET B
WHERE B.DATA = TO_DATE('27/08/2013', 'DD/MM/YYYY') 
  AND B.CODI_VOL = 'VLG2117'
ORDER BY 1;

/*3. Pes total facturat (com a Pes_Total) pel vol de KLM KLM4304 de l'9 d'Octubre de 2013*/
SELECT SUM(PES) AS PES_TOTAL
FROM MALETA
WHERE DATA = TO_DATE('09/10/2013', 'DD/MM/YYYY') 
  AND CODI_VOL = 'KLM4304'
ORDER BY 1;
      
/*4. El pes de les maletes (com a Pes_Total) dels passatgers italians.*/
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P 
WHERE P.PAIS = 'Italia' 
  AND M.NIF_PASSATGER = P.NIF
ORDER BY 1;

/*5. Numero m‡xim de bitllets (com a max bitllets) que ha comprat en una sola reserva en Narciso Blanco?*/
SELECT MAX(COUNT(*)) AS MAX_BITLLETS
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF AND
      P.NOM = 'Narciso Blanco'
GROUP BY B.LOCALITZADOR
ORDER BY 1;

/*6. Nom de la companyia i nombre de vols que tÈ (as N_VOLS).*/    
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1,2;

/*7. Nom de les Companyies amb mËs de 10 vols.*/
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(*) > 10
ORDER BY 1;

/*8. Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar tambÈ el nombre de vegades que han volat com a N_VEGADES.*/
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) >= 5
ORDER BY 2,1;

/*9. NIF i nom dels passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total*/
SELECT P.NIF, P.NOM, M.CODI_VOL, 
       TO_CHAR(M.DATA, 'DD/MM/YYYY') AS DATA, 
       SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM, M.CODI_VOL, M.DATA
HAVING SUM(PES) > 15
ORDER BY 1,2,3,4,5;

/*10. Nacionalitats representades per mÈs de dos passatgers amb bitllets.*/
SELECT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(*) > 2
ORDER BY 1;

/*1 (Reserves Vol) Nombre de persones mexicanes.*/
SELECT COUNT(NOM)
FROM PERSONA
WHERE PAIS = 'Mexic'
ORDER BY 1;

/*2 (Reserves vol) Nombre de maletes que ha facturat Chema Pamundi per cada vol que ha fet. Volem recuperar codi del vol, data i nombre de maletes.*/
SELECT B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, COUNT(M.CODI_MALETA) AS N_MALETES
FROM BITLLET B, PERSONA P, MALETA M
WHERE B.NIF_PASSATGER = P.NIF
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND P.NOM = 'Chema Pamundi'
GROUP BY B.CODI_VOL, B.DATA, M.CODI_MALETA
ORDER BY 1, 2, 3;

/*3 (Reserves Vol) Nom dels passatgers que han facturat mÈs d'una maleta vermella.*/
SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(P.NOM) > 1
ORDER BY 1;

/*4 (Reserves vol) N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.*/
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
  AND V.TERMINAL = PE.TERMINAL
  AND V.AREA = PE.AREA
  AND V.PORTA = PE.PORTA
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;

/*5 (Reserves Vol) Passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total*/
SELECT DISTINCT P.NIF, P.NOM, B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
GROUP BY P.NIF, P.NOM, B.CODI_VOL, B.DATA
HAVING SUM(M.PES) > 15
ORDER BY 1, 2, 3, 4, 5;

--1.- El nom i NIF de totes les persones.
SELECT Nom,NIF 
FROM PERSONA 
ORDER BY 1,2;

--2.- Nom dels persones vegetarians.
SELECT Nom 
FROM PERSONA 
WHERE OBSERVACIONS='Vegetaria/na' 
ORDER BY 1,2;

--3.- Codi de vol i data dels bitllets del passatger Alan Brito
SELECT B.CODI_VOL, B.DATA 
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_Passatger AND 
      P.Nom = 'Alan Brito'
ORDER BY 1,2;

--4.- Nom dels passatgers que tenen alguna maleta que pesa mÈs de 10 kilos.
--Especifica tambÈ en quin vol
SELECT P.NOM,M.CODI_VOL
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
      M.PES>10
ORDER BY 1,2;

--5.- El codi de la maleta, el pes i les mides de les maletes que ha facturat el passatger
--Jose Luis Lamata Feliz en els seus vols.
SELECT M.CODI_MALETA,M.PES,M.MIDES
FROM  PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
      P.NOM='Jose Luis Lamata Feliz'
ORDER BY 1,2,3;

--6.- Nom de les passatgers que han nascut entre el 05/06/1964 i el 03/09/1985.
SELECT Nom
FROM PERSONA
WHERE DATA_NAIXEMENT BETWEEN TO_DATE('05/06/1964','DD/MM/YYYY') 
                         AND TO_DATE('03/09/1985','DD/MM/YYYY')
ORDER BY 1;

--7.- Per al vol amb data 02/06/14 i codi RAM964 digueu el NIF dels passatgers que
--volen i el dels que han comprat el bitllet.
SELECT B.NIF_PASSATGER, R.NIF_CLIENT
FROM RESERVA R, BITLLET B
WHERE B.DATA= TO_DATE('02/06/14','dd/mm/YY') AND
      B.CODI_VOL = 'RAM964' AND
      R.LOCALITZADOR=B.LOCALITZADOR
ORDER BY 1,2;

--8.- Nom i edat dels passatgers del vol AEA2159 del 25/07/2013.
SELECT P.NOM,ROUND((CURRENT_DATE - P.DATA_NAIXEMENT)/365)
FROM PERSONA P, BITLLET B
WHERE B.CODI_VOL = 'AEA2159' AND
      B.DATA = TO_DATE('25/07/2013','DD/MM/YYYY') AND
      P.NIF=B.NIF_PASSATGER
ORDER BY 1,2;

--9.- Tots els models diferents d'aviÛ que existeixen
SELECT DISTINCT TIPUS_AVIO
FROM VOL
ORDER BY 1;

--10.- Les companyies amb vols al mes d'Agost de 2013.
SELECT DISTINCT V.Companyia
FROM VOL V
WHERE V.DATA BETWEEN TO_DATE('01/08/2013','DD/MM/YYYY') 
                 AND TO_DATE('31/08/2013','DD/MM/YYYY')
ORDER BY 1;


--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'‡rees
SELECT A.NOM, P.TERMINAL, COUNT (DISTINCT P.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT P
WHERE A.CODI_AEROPORT=P.CODI_AEROPORT
GROUP BY A.NOM, P.TERMINAL
ORDER BY 1,2,3

--Quantes maletes ha facturat en Aitor Tilla al vol amb destinaciÛ Pekin de la data 1/10/2013?
SELECT COUNT(DISTINCT M.CODI_MALETA)
FROM MALETA M, PERSONA P, VOL V
WHERE P.NIF=M.NIF_PASSATGER AND
M.CODI_VOL=V.CODI_VOL AND
M.DATA=V.DATA AND
V.DESTINACIO='Pekin' AND
V.DATA=TO_DATE('01/10/2013', 'DD/MM/YYYY') 

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT AVG(COUNT(DISTINCT PORTA))
FROM PORTA_EMBARCAMENT
GROUP BY CODI_AEROPORT

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P 
WHERE P.PAIS = 'Espanya' 
  AND M.NIF_PASSATGER = P.NIF
ORDER BY 1

/*1 (Reserves Vol) Nombre de persones mexicanes.*/
SELECT COUNT(NOM)
FROM PERSONA
WHERE PAIS = 'Mexic'
ORDER BY 1;

/*2 (Reserves vol) Nombre de maletes que ha facturat Chema Pamundi per cada vol que ha fet. Volem recuperar codi del vol, data i nombre de maletes.*/
SELECT B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, COUNT(M.CODI_MALETA) AS N_MALETES
FROM BITLLET B, PERSONA P, MALETA M
WHERE B.NIF_PASSATGER = P.NIF
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND P.NOM = 'Chema Pamundi'
GROUP BY B.CODI_VOL, B.DATA, M.CODI_MALETA
ORDER BY 1, 2, 3;

/*3 (Reserves Vol) Nom dels passatgers que han facturat mÈs d'una maleta vermella.*/
SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(P.NOM) > 1
ORDER BY 1;

/*4 (Reserves vol) N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.*/
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
  AND V.TERMINAL = PE.TERMINAL
  AND V.AREA = PE.AREA
  AND V.PORTA = PE.PORTA
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;

/*5 (Reserves Vol) Passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total*/
SELECT DISTINCT P.NIF, P.NOM, B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
GROUP BY P.NIF, P.NOM, B.CODI_VOL, B.DATA
HAVING SUM(M.PES) > 15
ORDER BY 1, 2, 3, 4, 5;

/*6 (Espectacles) Capacitat total del teatre Romea. */
SELECT ZR.CAPACITAT
FROM ZONES_RECINTE ZR, RECINTES R
WHERE ZR.CODI_RECINTE = R.CODI
AND R.NOM = 'Romea'
ORDER BY 1;

/*7 (Espectacles) Preu m‡xim i mÌnim de les entrades per líespectacle Jazz a la tardor.*/
SELECT MAX(PREU), MIN(PREU)
FROM ESPECTACLES E, PREUS_ESPECTACLES PE
WHERE PE.CODI_ESPECTACLE = E.CODI
  AND E.NOM = 'Jazz a la tardor'
ORDER BY 1,2;

/*8 (Espectacles) Data, hora i nombre díentrades venudes de cadascuna de les representacions de El M‡gic díOz.*/
SELECT TO_CHAR(E.DATA, 'DD/MM/YYYY') AS DATA, 
TO_CHAR(E.HORA, 'HH24:MI:SS') AS HORA,
COUNT(E.CODI_ESPECTACLE) AS N_ENTRADES
FROM ESPECTACLES ESP, ENTRADES E, REPRESENTACIONS R
WHERE E.CODI_ESPECTACLE = R.CODI_ESPECTACLE
  AND E.DATA = R.DATA
  AND E.HORA = E.HORA
  AND R.CODI_ESPECTACLE = ESP.CODI
  AND ESP.NOM = 'El M‡gic d''Oz'
GROUP BY E.DATA, E.HORA
ORDER BY 1, 2, 3;

/*9 (Espectacles) Nom dels espectacles on el preu de l'entrada mÈs econòmica sigui superior a 18Ä.*/
SELECT E.NOM
FROM ESPECTACLES E, PREUS_ESPECTACLES PE
WHERE PE.CODI_ESPECTACLE = E.CODI
GROUP BY E.NOM
HAVING MIN(PE.PREU) > 18
ORDER BY 1;

/*10 (Espectacles) DNI, nom i cognoms dels espectadors que sëhan gastat mÈs de 500Ä en espectacles.*/
SELECT ES.DNI, ES.NOM, ES.COGNOMS
FROM ESPECTADORS ES, ENTRADES E, REPRESENTACIONS R, ESPECTACLES ESP, PREUS_ESPECTACLES PE
WHERE ES.DNI = E.DNI_CLIENT
  AND E.CODI_ESPECTACLE = R.CODI_ESPECTACLE
  AND E.DATA = R.DATA
  AND E.HORA = E.HORA
  AND R.CODI_ESPECTACLE = ESP.CODI
  AND PE.CODI_ESPECTACLE = ESP.CODI
GROUP BY ES.DNI, ES.NOM, ES.COGNOMS
HAVING SUM(PE.PREU) > 500
ORDER BY 1, 2, 3

/*1. Numero de passatgers (com a N_Passatgers) amb cognom ìBlancoî que tÈ el vol amb codi IBE2119 i data 30/10/13.*/
SELECT COUNT(P.NOM) AS N_PASSATGERS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER 
  AND B.DATA = TO_DATE('30/10/2013', 'DD/MM/YYYY') 
  AND B.CODI_VOL = 'IBE2119' 
  AND P.NOM LIKE '%Blanco%'
ORDER BY 1;

/*2. Numero de butaques reservades (com a N_Butaques_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.*/
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM BITLLET B
WHERE B.DATA = TO_DATE('27/08/2013', 'DD/MM/YYYY') 
  AND B.CODI_VOL = 'VLG2117'
ORDER BY 1;

/*3. Pes total facturat (com a Pes_Total) pel vol de KLM KLM4304 de l'9 d'Octubre de 2013*/
SELECT SUM(PES) AS PES_TOTAL
FROM MALETA
WHERE DATA = TO_DATE('09/10/2013', 'DD/MM/YYYY') 
  AND CODI_VOL = 'KLM4304'
ORDER BY 1;
      
/*4. El pes de les maletes (com a Pes_Total) dels passatgers italians.*/
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P 
WHERE P.PAIS = 'Italia' 
  AND M.NIF_PASSATGER = P.NIF
ORDER BY 1;

/*5. Numero m‡xim de bitllets (com a max bitllets) que ha comprat en una sola reserva en Narciso Blanco?*/
SELECT MAX(COUNT(*)) AS MAX_BITLLETS
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF AND
      P.NOM = 'Narciso Blanco'
GROUP BY B.LOCALITZADOR
ORDER BY 1;

/*6. Nom de la companyia i nombre de vols que tÈ (as N_VOLS).*/    
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1,2;

/*7. Nom de les Companyies amb mËs de 10 vols.*/
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(*) > 10
ORDER BY 1;

/*8. Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar tambÈ el nombre de vegades que han volat com a N_VEGADES.*/
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) >= 5
ORDER BY 2,1;

/*9. NIF i nom dels passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total*/
SELECT P.NIF, P.NOM, M.CODI_VOL, 
       TO_CHAR(M.DATA, 'DD/MM/YYYY') AS DATA, 
       SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM, M.CODI_VOL, M.DATA
HAVING SUM(PES) > 15
ORDER BY 1,2,3,4,5;

/*10. Nacionalitats representades per mÈs de dos passatgers amb bitllets.*/
SELECT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(*) > 2
ORDER BY 1;

/*1 (Reserves Vol) Nombre de persones mexicanes.*/
SELECT COUNT(NOM)
FROM PERSONA
WHERE PAIS = 'Mexic'
ORDER BY 1;

/*2 (Reserves vol) Nombre de maletes que ha facturat Chema Pamundi per cada vol que ha fet. Volem recuperar codi del vol, data i nombre de maletes.*/
SELECT B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, COUNT(M.CODI_MALETA) AS N_MALETES
FROM BITLLET B, PERSONA P, MALETA M
WHERE B.NIF_PASSATGER = P.NIF
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND P.NOM = 'Chema Pamundi'
GROUP BY B.CODI_VOL, B.DATA, M.CODI_MALETA
ORDER BY 1, 2, 3;

/*3 (Reserves Vol) Nom dels passatgers que han facturat mÈs d'una maleta vermella.*/
SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(P.NOM) > 1
ORDER BY 1;

/*4 (Reserves vol) N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.*/
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
  AND V.TERMINAL = PE.TERMINAL
  AND V.AREA = PE.AREA
  AND V.PORTA = PE.PORTA
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;

/*5 (Reserves Vol) Passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total*/
SELECT DISTINCT P.NIF, P.NOM, B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
GROUP BY P.NIF, P.NOM, B.CODI_VOL, B.DATA
HAVING SUM(M.PES) > 15
ORDER BY 1, 2, 3, 4, 5;

--1.- El nom i NIF de totes les persones.
SELECT Nom,NIF 
FROM PERSONA 
ORDER BY 1,2;

--2.- Nom dels persones vegetarians.
SELECT Nom 
FROM PERSONA 
WHERE OBSERVACIONS='Vegetaria/na' 
ORDER BY 1,2;

--3.- Codi de vol i data dels bitllets del passatger Alan Brito
SELECT B.CODI_VOL, B.DATA 
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_Passatger AND 
      P.Nom = 'Alan Brito'
ORDER BY 1,2;

--4.- Nom dels passatgers que tenen alguna maleta que pesa mÈs de 10 kilos.
--Especifica tambÈ en quin vol
SELECT P.NOM,M.CODI_VOL
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
      M.PES>10
ORDER BY 1,2;

--5.- El codi de la maleta, el pes i les mides de les maletes que ha facturat el passatger
--Jose Luis Lamata Feliz en els seus vols.
SELECT M.CODI_MALETA,M.PES,M.MIDES
FROM  PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
      P.NOM='Jose Luis Lamata Feliz'
ORDER BY 1,2,3;

--6.- Nom de les passatgers que han nascut entre el 05/06/1964 i el 03/09/1985.
SELECT Nom
FROM PERSONA
WHERE DATA_NAIXEMENT BETWEEN TO_DATE('05/06/1964','DD/MM/YYYY') 
                         AND TO_DATE('03/09/1985','DD/MM/YYYY')
ORDER BY 1;

--7.- Per al vol amb data 02/06/14 i codi RAM964 digueu el NIF dels passatgers que
--volen i el dels que han comprat el bitllet.
SELECT B.NIF_PASSATGER, R.NIF_CLIENT
FROM RESERVA R, BITLLET B
WHERE B.DATA= TO_DATE('02/06/14','dd/mm/YY') AND
      B.CODI_VOL = 'RAM964' AND
      R.LOCALITZADOR=B.LOCALITZADOR
ORDER BY 1,2;

--8.- Nom i edat dels passatgers del vol AEA2159 del 25/07/2013.
SELECT P.NOM,ROUND((CURRENT_DATE - P.DATA_NAIXEMENT)/365)
FROM PERSONA P, BITLLET B
WHERE B.CODI_VOL = 'AEA2159' AND
      B.DATA = TO_DATE('25/07/2013','DD/MM/YYYY') AND
      P.NIF=B.NIF_PASSATGER
ORDER BY 1,2;

--9.- Tots els models diferents d'aviÛ que existeixen
SELECT DISTINCT TIPUS_AVIO
FROM VOL
ORDER BY 1;

--10.- Les companyies amb vols al mes d'Agost de 2013.
SELECT DISTINCT V.Companyia
FROM VOL V
WHERE V.DATA BETWEEN TO_DATE('01/08/2013','DD/MM/YYYY') 
                 AND TO_DATE('31/08/2013','DD/MM/YYYY')
ORDER BY 1;


--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'‡rees
SELECT A.NOM, P.TERMINAL, COUNT (DISTINCT P.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT P
WHERE A.CODI_AEROPORT=P.CODI_AEROPORT
GROUP BY A.NOM, P.TERMINAL
ORDER BY 1,2,3

--Quantes maletes ha facturat en Aitor Tilla al vol amb destinaciÛ Pekin de la data 1/10/2013?
SELECT COUNT(DISTINCT M.CODI_MALETA)
FROM MALETA M, PERSONA P, VOL V
WHERE P.NIF=M.NIF_PASSATGER AND
M.CODI_VOL=V.CODI_VOL AND
M.DATA=V.DATA AND
V.DESTINACIO='Pekin' AND
V.DATA=TO_DATE('01/10/2013', 'DD/MM/YYYY') 

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT AVG(COUNT(DISTINCT PORTA))
FROM PORTA_EMBARCAMENT
GROUP BY CODI_AEROPORT

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P 
WHERE P.PAIS = 'Espanya' 
  AND M.NIF_PASSATGER = P.NIF
ORDER BY 1


 
--8. Nom de la(/es) companyia(/es) que t�(nen) m�s bitllets reservats en algun dels
seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.     ///1
SELECT V.COMPANYIA, V.CODI_VOL, V.DATA, COUNT(*) AS N_SEIENTS_RESERVATS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4
 
--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers  /////////1
--del(s) vol(s) que t�(nen) el nombre de passatgers m�s petit.      
SELECT V.ORIGEN, V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY'), COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(*)=(SELECT MIN(COUNT(*))
 FROM BITLLET B, VOL V
 WHERE B.CODI_VOL=V.CODI_VOL AND
 B.DATA=V.DATA
 GROUP BY V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4
 
--NIF i nom del(s) passatger(s) que t� mes maletes registrades al seu nom.       /////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                      FROM PERSONA P, MALETA M
                      WHERE M.NIF_PASSATGER = P.NIF
                      GROUP BY P.NIF, P.NOM)
ORDER BY 1                      

--Nom(es) de la(es) companyia(es) que opera(n) amb m�s tipus d'avio.        ////////////1
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(TIPUS_AVIO)>=ALL(SELECT COUNT(TIPUS_AVIO)
FROM VOL
GROUP BY COMPANYIA)
 
--Nom(es) de la(es) companyia(es) que ha(n) estat reservada(es) per m�s clients (persones) d'Amsterdam ////////////1
SELECT V.COMPANYIA
FROM VOL V, BITLLET B, RESERVA R, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, PERSONA P, BITLLET B, RESERVA R
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
GROUP BY V.COMPANYIA)
GROUP BY V.COMPANYIA
ORDER BY 1
 
--Nom(es) del passatger/s que ha/n facturat la maleta de m�s pes. ////////////1
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
M.PES=(SELECT MAX(PES)
FROM MALETA)

 
--NIF i nom de la(/es) persona(/es) que ha(n) reservat m�s bitllets. //////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM PERSONA P, RESERVA R
WHERE P.NIF=R.NIF_CLIENT
GROUP BY P.NOM)
ORDER BY 1, 2
 
--Color(s) de la(/es) maleta(/es) facturada(/es) m�s pesada(/es). //1
SELECT DISTINCT COLOR
FROM MALETA M
WHERE M.PES =(SELECT MAX(PES)
FROM MALETA)
ORDER BY 1
 
--9. Tipus d'avi�(ns) que t�(nen) m�s passatgers no espanyols.          //////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM BITLLET B, VOL V, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.NIF_PASSATGER = P.NIF AND
P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                FROM BITLLET B, VOL V, PERSONA P
                WHERE V.CODI_VOL=B.CODI_VOL AND
                V.DATA=B.DATA AND
                B.NIF_PASSATGER = P.NIF AND
                P.PAIS = 'Espanya')
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS = 'Espanya')
                    GROUP BY V.TIPUS_AVIO)
 
--Nom(es) de la companyia(es) que fa(n) m�s vols amb origen Italia.       ////////////1
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA)
ORDER BY 1
 
--Tipus d'avi�(ns) que fa(n) mes vols amb origen Espanya.     ////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1
 
--///1
--Dates de vol (DD/MM/YYYY) del(s) avi�(ns) amb mes capacitat (major nombre de seients)
SELECT TO_CHAR(DATA,'DD/MM/YYYY')
FROM SEIENT
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM SEIENT
GROUP BY CODI_VOL, DATA)
GROUP BY CODI_VOL, DATA
ORDER BY 1

--2.-Tipus d'avio amb mes files.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA AND
S.FILA=(SELECT MAX(FILA) FROM SEIENT)
ORDER BY 1;


SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA AND
S.FILA>=ALL(SELECT FILA FROM SEIENT)
ORDER BY 1;


--////////////////1

SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA)>=ALL(SELECT COUNT(DISTINCT S.FILA) 
FROM VOL V, SEIENT S 
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;

 
--1. Nom de totes les persones que son del mateix pa�s que Domingo Diaz Festivo (ell incl�s). /////////1
SELECT DISTINCT NOM
FROM PERSONA
WHERE PAIS=(SELECT PAIS 
FROM PERSONA WHERE NOM='Domingo Diaz Festivo')
ORDER BY 1;
 
--3.-Nom del(s) aeroport(s) amb m�s terminals.          ////////////1
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)>=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1;
 

--4.-NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets///1
SELECT P.NIF, P.NOM, SUM(M.PES)
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES)<=ALL(SELECT (SUM(M.PES))
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM)
 
--5. Nombre de vegades que apareix la nacionalitat m�s freq�ent entre els passatgers de la
--BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.       ///////1
SELECT DISTINCT PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA
GROUP BY PAIS
HAVING COUNT(*)= (SELECT MAX(COUNT(*)) 
                  FROM PERSONA
                  GROUP BY PAIS);

--////
--Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/13.
SELECT S.FILA, S.LLETRA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('31/07/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2195' AND
        S.NIF_PASSATGER IS NULL
ORDER BY 1,2
--////////
--Companyia que vola a tots els aeroports espanyols	///1
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND
        A.PAIS='Espanya'
HAVING COUNT (DISTINCT A.CODI_AEROPORT)=(SELECT COUNT(CODI_AEROPORT)
FROM AEROPORT
WHERE PAIS='Espanya')
GROUP BY V.COMPANYIA;

SELECT DISTINCT V.COMPANYIA
FROM VOL V
WHERE V.COMPANYIA NOT IN
(SELECT COMPANYIA FROM VOL V, AEROPORT A
WHERE A.PAIS = 'Espanya' AND
(A.CODI_AEROPORT, V.COMPANYIA) NOT IN
(SELECT DESTINACIO, COMPANYIA FROM VOL) )
--///////
--Nom del/s passatger/s que sempre vola amb primera classe.	///1

SELECT DISTINCT P.NOM
FROM BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
P.NOM NOT IN 
(SELECT P.NOM 
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND 
B.CLASSE = '2');
---////////
--Numero de places lliures pel vol AEA2195 amb data 31/07/13.	///1
SELECT (T1.TOTAL - T2.OCUPATS)
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2;

--/////////
--Per al vol AEA2195 amb data 31/07/13, quin �s el percentatge d'ocupaci�?///1
SELECT (T2.OCUPATS / T1.TOTAL) * 100 
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2;

--/////
--Noms dels aeroports i nom de les ciutats de destinaci� d'aquells vols a les que els falta un passatger per estar plens///1
SELECT DISTINCT A.NOM, A.CIUTAT
FROM (SELECT CODI_VOL, DATA, COUNT(*) AS S_TOTALS
FROM SEIENT
GROUP BY CODI_VOL, DATA) ST,
(SELECT CODI_VOL, DATA, COUNT(*) AS S_OCUPATS
FROM BITLLET
GROUP BY CODI_VOL, DATA) SO, VOL V, AEROPORT A
WHERE ST.CODI_VOL = SO.CODI_VOL AND
ST.DATA = SO.DATA AND
SO.S_OCUPATS = ST.S_TOTALS + 1 AND
SO.CODI_VOL = V.CODI_VOL AND
SO.DATA = V.DATA AND
V.DESTINACIO = A.CODI_AEROPORT
ORDER BY 1, 2
--////////////
--En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a �s dels passatgers? //////1
SELECT S.FILA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('01/08/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2159' AND
        S.NIF_PASSATGER IS NULL AND
        S.LLETRA = 'A'
ORDER BY 1

--////////////////
--Numero de portes que queden lliures de la terminal 3, area 3, del aeroport 'Berlin-Sch�nefeld International Airport' el dia 04/07/2013	///1
SELECT (TOTALES - OCUPADAS)
FROM
(SELECT COUNT(*) AS TOTALES FROM PORTA_EMBARCAMENT
WHERE CODI_AEROPORT='SXF' AND TERMINAL='3' AND AREA='3'),
(SELECT COUNT (*) AS OCUPADAS FROM VOL 
WHERE DATA = TO_DATE('04/07/2013','DD/MM/YYYY') AND
TERMINAL='3' AND
AREA='3' AND
ORIGEN ='SXF')
--///////
--Aeroports que no tenen vol en los tres primeros meses de 2014	///1
SELECT DISTINCT NOM
FROM AEROPORT
WHERE CODI_AEROPORT NOT IN
(SELECT DESTINACIO FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY')) AND CODI_AEROPORT NOT IN
(SELECT ORIGEN FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY'))
ORDER BY 1

--//////////Del total de maletes facturades, quin percentatge s�n de color vermell. (NOTA: el percentatge �s sempre un numero entre 0 i 100)//////1
SELECT (T1.ROJAS / T2.TOTAL) * 100
FROM (SELECT COUNT(*) AS TOTAL FROM MALETA) T2,
(SELECT COUNT(*) AS ROJAS FROM MALETA WHERE COLOR = 'vermell') T1

--///////////////////
--Companyies que tenen, almenys, els mateixos tipus d'avions que Vueling Airlines. (El resultat tamb� ha de tornar  Vueling airlines)

SELECT DISTINCT COMPANYIA
FROM VOL
WHERE TIPUS_AVIO IN 
(SELECT DISTINCT TIPUS_AVIO FROM VOL WHERE COMPANYIA='VUELING AIRLINES')
ORDER BY 1

////////////

--Dades de contacte (nom, email, tel�fon) de les persones que no han volat mai a la Xina.	///1
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM PERSONA P
MINUS
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA AND
V.DESTINACIO='PEK'
ORDER BY 1, 2, 3

--/////////////////////////////////////////////////

--1. Nom de totes les persones que son del mateix pa�s que Domingo Diaz Festivo (ell incl�s). /////////1
SELECT DISTINCT NOM
FROM PERSONA
WHERE PAIS=(SELECT PAIS 
FROM PERSONA WHERE NOM='Domingo Diaz Festivo')
ORDER BY 1;
 
--3.-Nom del(s) aeroport(s) amb m�s terminals.          ////////////1
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)>=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1;
 
--//1
--4.-NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets
SELECT P.NIF, P.NOM, SUM(M.PES)
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES)<=ALL(SELECT (SUM(M.PES))
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM)
 
5--. Nombre de vegades que apareix la nacionalitat m�s freq�ent entre els passatgers de la
--BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.       //////////1
SELECT DISTINCT PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA
GROUP BY PAIS
HAVING COUNT(*)= (SELECT MAX(COUNT(*)) 
                  FROM PERSONA
                  GROUP BY PAIS);

--7.-NIF i nom de la(/es) persona(/es) que ha(n) reservat m�s bitllets. //////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM PERSONA P, RESERVA R
WHERE P.NIF=R.NIF_CLIENT
GROUP BY P.NOM)
ORDER BY 1, 2
 
--Color(s) de la(/es) maleta(/es) facturada(/es) m�s pesada(/es). //1
SELECT DISTINCT COLOR
FROM MALETA M
WHERE M.PES =(SELECT MAX(PES)
FROM MALETA)
ORDER BY 1
 
---9. Tipus d'avi�(ns) que t�(nen) m�s passatgers no espanyols.          //////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM BITLLET B, VOL V, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.NIF_PASSATGER = P.NIF AND
P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                FROM BITLLET B, VOL V, PERSONA P
                WHERE V.CODI_VOL=B.CODI_VOL AND
                V.DATA=B.DATA AND
                B.NIF_PASSATGER = P.NIF AND
                P.PAIS = 'Espanya')
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS = 'Espanya')
                    GROUP BY V.TIPUS_AVIO)
 
--Nom(es) de la companyia(es) que fa(n) m�s vols amb origen Italia.       ////////////1
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA)
ORDER BY 1
 
--Tipus d'avi�(ns) que fa(n) mes vols amb origen Espanya.     ////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1
 
--///1
--Dates de vol (DD/MM/YYYY) del(s) avi�(ns) amb mes capacitat (major nombre de seients)
SELECT TO_CHAR(DATA,'DD/MM/YYYY')
FROM SEIENT
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM SEIENT
GROUP BY CODI_VOL, DATA)
GROUP BY CODI_VOL, DATA
ORDER BY 1

--8. Nom de la(/es) companyia(/es) que t�(nen) m�s bitllets reservats en algun dels seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.     ///1
SELECT V.COMPANYIA, V.CODI_VOL, V.DATA, COUNT(*) AS N_SEIENTS_RESERVATS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4
 
--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers  /////////1
--del(s) vol(s) que t�(nen) el nombre de passatgers m�s petit.      
SELECT V.ORIGEN, V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY'), COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(*)=(SELECT MIN(COUNT(*))
 FROM BITLLET B, VOL V
 WHERE B.CODI_VOL=V.CODI_VOL AND
 B.DATA=V.DATA
 GROUP BY V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4
 
NIF i nom del(s) passatger(s) que t� mes maletes registrades al seu nom.       /////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                      FROM PERSONA P, MALETA M
                      WHERE M.NIF_PASSATGER = P.NIF
                      GROUP BY P.NIF, P.NOM)
ORDER BY 1                      
 
Nom(es) de la(es) companyia(es) que opera(n) amb m�s tipus d'avio.        ////////////1
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(TIPUS_AVIO)>=ALL(SELECT COUNT(TIPUS_AVIO)
FROM VOL
GROUP BY COMPANYIA)
 
Nom(es) de la(es) companyia(es) que ha(n) estat reservada(es) per m�s clients (persones) d'Amsterdam ////////////1
SELECT V.COMPANYIA
FROM VOL V, BITLLET B, RESERVA R, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, PERSONA P, BITLLET B, RESERVA R
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
GROUP BY V.COMPANYIA)
GROUP BY V.COMPANYIA
ORDER BY 1
 
--Nom(es) del passatger/s que ha/n facturat la maleta de m�s pes. ////////////1
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
M.PES=(SELECT MAX(PES)
FROM MALETA)

--(Reserves vols) NIF, nom, poblaci� i pa�s de totes les persones existents a la base de dades.	///1
SELECT NIF, NOM, POBLACIO, PAIS
FROM PERSONA					
ORDER BY 1, 2, 3, 4

--/////////////////////////////////////////////////

--Companyies que volen a tots els aeroports de Espanya ///1
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.DESTINACIO = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'					
GROUP BY V.COMPANYIA
HAVING COUNT(DISTINCT A.CODI_AEROPORT)=(SELECT COUNT(CODI_AEROPORT)
FROM AEROPORT WHERE PAIS='Espanya')

--/////////////////////////////////////////////////
	
--Nombre de persones mexicanes.		///1
SELECT COUNT(*)
FROM PERSONA					
WHERE PAIS = 'Mexic'

--//////////////////////////////////////////////////

--NIF i nom del(s) passatger(s) que t� mes maletes registrades al seu nom.	///1
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM, M.NIF_PASSATGER			
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM, M.NIF_PASSATGER)
ORDER BY 1, 2

--/////////////////////////////////////////////////

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.	///1
SELECT DISTINCT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V						
GROUP BY V.COMPANYIA
ORDER BY 1,2

--//////////////////////////////////////////////////

--Nom de la(s) companyia(es) amb vols d'origen a tots els aeroports de la Xina.	///1
(SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Xina')
MINUS
(SELECT DISTINCT V.COMPANYIA				
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Xina' AND V.COMPANYIA NOT IN (SELECT V.COMPANYIA FROM VOL V, AEROPORT A
WHERE A.PAIS = 'Xina'))

--//////////////////////////////////////////////////
--Nom dels passatgers que tenen bitllet, per� no han fet mai cap reserva	///1
(SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER)
MINUS (SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, RESERVA R			
WHERE P.NIF = B.NIF_PASSATGER AND
R.NIF_CLIENT = P.NIF AND
B.NIF_PASSATGER = R.NIF_CLIENT)

--//////////////////////////////////////////////////

--El pes de les maletes (com a pes_total) dels passatgers italians.///1
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE P.NIF = M.NIF_PASSATGER AND			
P.PAIS = 'Italia'
GROUP BY M.NIF_PASSATGER

--//////////////////////////////////////////////////

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'�rea i el nombre de portes d'aquesta �rea.///1
SELECT DISTINCT A.NOM, PE.TERMINAL, PE.AREA, COUNT(PE.PORTA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE			
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3, 4

--//////////////////////////////////////////////////
--Percentatge d'espanyols als vols amb destinaci� Munich. (NOTA: el percentatge �s sempre un numero entre 0 i 100)///1

SELECT (T2.ESPANYOLS / T1.TOTAL) * 100 
FROM
(SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
V.DESTINACIO='MUC') T1,
(SELECT COUNT(*) AS ESPANYOLS
FROM BITLLET B, VOL V, PERSONA P
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
P.NIF=B.NIF_PASSATGER AND
V.DESTINACIO='MUC' AND
P.PAIS='Espanya') T2

--////////////

--Nombre de passatgers dels vols d'IBERIA per pa�sos. Es demana pa�s i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Nombre m�xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

--Nom dels passatgers que han facturat m�s d'una maleta vermella
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Nom de les Companyies amb m�s de 10 vols.
SELECT V.COMPANYIA
FROM VOL V
GROUP BY V.COMPANYIA
HAVING COUNT(*)>10
ORDER BY 1

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT COUNT(*)/(COUNT(DISTINCT PE.CODI_AEROPORT))
FROM PORTA_EMBARCAMENT PE
ORDER BY 1

--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco t� el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGER
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL ='IBE2119' AND
B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
AND P.NOM LIKE '%Blanco%'

--Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar tamb� el nombre de vegades que han volat com a N_VEGADES.
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) > 4
ORDER BY 2,1

--Nombre de passatgers agrupats segons pa�s de destinaci� (en el vol). Mostreu pa�s i nombre de passatgers.
SELECT A.PAIS, COUNT(*)
FROM BITLLET B, VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND 
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA
GROUP BY A.PAIS
ORDER BY 1,2

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'�rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Nacionalitats representades per m�s de dos passatgers amb bitllets.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(P.PAIS)>2
ORDER BY 1

--Nombre de passatgers dels vols d'IBERIA per pa�sos. Es demana pa�s i nombre de passatgers
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Quantes maletes ha facturat en Aitor Tilla al vol amb destinaci� Pekin de la data 1/10/2013?
SELECT COUNT(*)
FROM BITLLET B, VOL V, MALETA M, PERSONA P
WHERE V.DESTINACIO='PEK' AND
P.NOM='Aitor Tilla' AND
P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.DATA=V.DATA AND
B.DATA=M.DATA AND
B.CODI_VOL=M.CODI_VOL AND
M.DATA=TO_DATE('01/10/2013','DD/MM/YYYY')

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--Nom dels passatgers que han facturat m�s d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--N�mero de porta amb m�s d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN=PE.CODI_AEROPORT AND
V.TERMINAL=PE.TERMINAL AND
V.AREA=PE.AREA AND
V.PORTA=PE.PORTA
HAVING COUNT(*)>1
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
ORDER BY 1,2,3,4

--Nombre de persones mexicanes.
SELECT COUNT(*)
FROM PERSONA 
WHERE PAIS='Mexic'

--Nombre de passatgers dels vols d'IBERIA per pa�sos. Es demana pa�s i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

-- Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--NIF i nom dels passatgers que han facturat m�s de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, NOM, CODI_VOL, DATA i PES_TOTAL
SELECT M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.CODI_VOL=M.CODI_VOL AND
B.DATA=M.DATA
GROUP BY M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA
HAVING SUM(M.PES)>15
ORDER BY 1, 2, 3, 4, 5

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1

-- El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Italia'
ORDER BY 1

--Nacionalitats representades per m�s de dos passatgers amb bitllets.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(P.PAIS)>2
ORDER BY 1

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'�rea i el nombre de portes d'aquesta �rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'�rea i el nombre de portes d'aquesta �rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3


---Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'�rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--Tipus de avions que volen a l'aeroport de Orly
SELECT DISTINCT TIPUS_AVIO
FROM VOL
WHERE DESTINACIO = 'ORY'
ORDER BY 1

--Nom i edat dels passatgers del vol AEA2159 del 25/07/2013.
SELECT DISTINCT P.NOM, ROUND((SYSDATE - P.DATA_NAIXEMENT)/365)
FROM PERSONA P, BITLLET B
WHERE B.CODI_VOL = 'AEA2159'
AND B.NIF_PASSATGER = P.NIF
AND B.DATA = TO_DATE('25/07/2013', 'DD/MM/YYYY')
ORDER BY 1, 2

--Nom de les passatgers que han nascut entre el 05/06/1964 i el 03/09/1985.
SELECT DISTINCT NOM
FROM PERSONA
WHERE DATA_NAIXEMENT >= TO_DATE('05/06/1964','DD/MM/YYYY')
AND DATA_NAIXEMENT <= TO_DATE('03/09/1985','DD/MM/YYYY')
ORDER BY 1

--(Reserves Vols) Localitzador(s) i nom de les persones pel vol KLM4303.
SELECT DISTINCT B.LOCALITZADOR, P.NOM
FROM PERSONA P, VOL V, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
AND V.CODI_VOL = 'KLM4303'
AND B.CODI_VOL = V.CODI_VOL
ORDER BY 1, 2

--Forma de pagament de les reserves fetes per a persones de nacionalitat alemana
SELECT R.MODE_PAGAMENT
FROM RESERVA R, PERSONA P
WHERE R.NIF_CLIENT = P.NIF
AND P.PAIS = 'Alemanya'
ORDER BY 1

--Ciutats espanyoles amb aeroports.
SELECT DISTINCT CIUTAT
FROM AEROPORT
WHERE PAIS = 'Espanya'
ORDER BY 1

--Nom del passatger i pes de les maletes facturades per passatgers espanyols.
SELECT DISTINCT P.NOM, M.PES
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
AND P.PAIS = 'Espanya'
ORDER BY 1, 2

--Nacionalitat dels passatgers dels vols amb destinaci� Paris Orly Airport.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B, VOL V
WHERE B.NIF_PASSATGER = P.NIF
AND B.CODI_VOL = V.CODI_VOL
AND V.DESTINACIO = 'ORY'
ORDER BY 1

--Codi de vol i data dels bitllets del passatger Alan Brito.
SELECT B.CODI_VOL, B.DATA
FROM BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND P.NOM = 'Alan Brito'
ORDER BY 1, 2

--(Reserves Vol) Nom dels passatgers que han nascut abans de l'any 1975.
SELECT DISTINCT NOM
FROM PERSONA
WHERE DATA_NAIXEMENT < TO_DATE('01/01/1975','DD/MM/YYYY')
ORDER BY 1

--El codi de la maleta, el pes i les mides de les maletes que ha facturat el passatger Jose Luis Lamata Feliz en els seus vols.
SELECT M.CODI_MALETA, M.PES, M.MIDES
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER = P.NIF
AND P.NOM = 'Jose Luis Lamata Feliz'
ORDER BY 1, 2, 3

--Nom dels passatgers que tenen alguna maleta que pesa m�s de 10 kilos. Especifica tamb� en quin vol.
SELECT DISTINCT P.NOM, M.CODI_VOL
FROM PERSONA P, MALETA M, VOL V
WHERE M.CODI_VOL = V.CODI_VOL
AND P.NIF = M.NIF_PASSATGER
AND M.PES > 10
ORDER BY 1, 2

--(Reserves Vols) Codi de tots els vols amb bitllets amb data d'abans de l'1 de Desembre de 2013.
SELECT DISTINCT V.CODI_VOL
FROM RESERVA R, VOL V, BITLLET B, PERSONA P
WHERE R.NIF_CLIENT = P.NIF
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA < TO_DATE('01/12/2013', 'DD/MM/YYYY')
ORDER BY 1

--Les companyies amb vols al mes d'Agost de 2013.
SELECT DISTINCT COMPANYIA
FROM VOL
WHERE DATA >= TO_DATE('01/08/2013', 'DD/MM/YYYY')
AND DATA <= TO_DATE('31/08/2013', 'DD/MM/YYYY')
ORDER BY 1

--Nom i Poblaci� dels passatgers que han viatjat algun cop en 1a classe.
SELECT DISTINCT P.NOM, P.POBLACIO
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
AND B.CLASSE = '1'
ORDER BY 1, 2

--(Reserves Vols) Localitzador(s) i nom de les persones pel vol KLM4303.
SELECT DISTINCT B.LOCALITZADOR, P.NOM
FROM PERSONA P, VOL V, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
AND V.CODI_VOL = 'KLM4303'
AND B.CODI_VOL = V.CODI_VOL
ORDER BY 1, 2

--Nom dels persones vegetarians.
SELECT DISTINCT P.NOM
FROM PERSONA P
WHERE P.OBSERVACIONS = 'Vegetaria/na'
ORDER BY 1

--El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS pes_total
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER=P.NIF AND P.PAIS='Italia'

--Quan pes ha facturat l'Alberto Carlos Huevos al vol amb destinaci� Rotterdam de la data 02 de juny del 2013?
SELECT SUM(M.PES)
FROM MALETA M, PERSONA P, VOL V
WHERE M.NIF_PASSATGER = P.NIF AND
M.CODI_VOL = V.CODI_VOL AND
P.NOM = 'Alberto Carlos Huevos' AND
V.DESTINACIO = 'RTM' AND
V.DATA = M.DATA AND
M.DATA = TO_DATE('02/06/2013','DD/MM/YYYY')

Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

Nombre m�xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

Nombre de passatgers (com a N_Passatgers) amb cognom Blanco t� el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGER
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL ='IBE2119' AND
B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
AND P.NOM LIKE '%Blanco%'

Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

Nacionalitats representades per m�s de dos passatgers amb bitllets.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(P.PAIS)>2
ORDER BY 1

--Nom dels passatgers que han facturat m�s d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT COUNT(*)/(COUNT(DISTINCT PE.CODI_AEROPORT))
FROM PORTA_EMBARCAMENT PE
ORDER BY 1

--Nombre de passatgers dels vols d'IBERIA per pa�sos. Es demana pa�s i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'�rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'�rea i el nombre de portes d'aquesta �rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'�rea i el nombre de portes d'aquesta �rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

--El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Italia'
ORDER BY 1

--Nombre m�xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

--NIF i nom dels passatgers que han facturat m�s de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, NOM, CODI_VOL, DATA i PES_TOTAL
SELECT M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.CODI_VOL=M.CODI_VOL AND
B.DATA=M.DATA
GROUP BY M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA
HAVING SUM(M.PES)>15
ORDER BY 1, 2, 3, 4, 5

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Nombre de passatgers dels vols d'IBERIA per pa�sos. Es demana pa�s i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'�rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar tamb� el nombre de vegades que han volat com a N_VEGADES.
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) > 4
ORDER BY 2,1

--Nombre de passatgers agrupats segons pa�s de destinaci� (en el vol). Mostreu pa�s i nombre de passatgers
SELECT A.PAIS, COUNT(*)
FROM BITLLET B, VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND 
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA
GROUP BY A.PAIS
ORDER BY 1,2

Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1

Nom de les Companyies amb m�s de 10 vols.
SELECT V.COMPANYIA
FROM VOL V
GROUP BY V.COMPANYIA
HAVING COUNT(*)>10
ORDER BY 1

Quantes maletes ha facturat en Aitor Tilla al vol amb destinaci� Pekin de la data 1/10/2013?
SELECT COUNT(*)
FROM BITLLET B, VOL V, MALETA M, PERSONA P
WHERE V.DESTINACIO='PEK' AND
P.NOM='Aitor Tilla' AND
P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.DATA=V.DATA AND
B.DATA=M.DATA AND
B.CODI_VOL=M.CODI_VOL AND
M.DATA=TO_DATE('01/10/2013','DD/MM/YYYY')

Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

Nombre de passatgers (com a N_Passatgers) amb cognom Blanco t� el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGER
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL ='IBE2119' AND
B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
AND P.NOM LIKE '%Blanco%'

Nombre m�xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

--N�mero de porta amb m�s d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN=PE.CODI_AEROPORT AND
V.TERMINAL=PE.TERMINAL AND
V.AREA=PE.AREA AND
V.PORTA=PE.PORTA
HAVING COUNT(*)>1
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
ORDER BY 1,2,3,4

--Nom dels passatgers que han facturat m�s d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Nombre de passatgers agrupats segons pa�s de destinaci� (en el vol). Mostreu pa�s i nombre de passatgers.
SELECT A.PAIS, COUNT(*)
FROM BITLLET B, VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND 
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA
GROUP BY A.PAIS
ORDER BY 1,2

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT COUNT(*)/(COUNT(DISTINCT PE.CODI_AEROPORT))
FROM PORTA_EMBARCAMENT PE
ORDER BY 1

--Nom dels passatgers que han facturat m�s d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')


--Tipus d'avi�(ns) amb m�s files.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA)>=ALL(SELECT COUNT(DISTINCT S.FILA) 
FROM VOL V, SEIENT S 
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Nom del(s) aeroport(s) amb el m�nim n�mero de terminals.
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)<=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1

--NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets.

SELECT P.NIF, P.NOM, SUM(M.PES)
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES)<=ALL(SELECT (SUM(M.PES))
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM)

--Nom(es) del(s) aeroport(s) amb m�s terminals.
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)>=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1

--Nom(es) del passatger/s que ha/n facturat la maleta de m�s pes.
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
M.PES=(SELECT MAX(PES)
FROM MALETA)

Nom de totes les persones que son del mateix pa�s que "Domingo Diaz Festivo" (ell incl�s).
SELECT DISTINCT NOM
FROM PERSONA
WHERE PAIS=(SELECT PAIS 
FROM PERSONA WHERE NOM='Domingo Diaz Festivo')
ORDER BY 1

Nom(es) de la companyia(es) que fa(n) m�s vols amb origen Italia.
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA)
ORDER BY 1

--Tipus d'avi�(ns) que fa(n) mes vols amb origen Espanya.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers del(s) vol(s) que t�(nen) el nombre de passatges m�s petit.
SELECT V.ORIGEN, V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY'), COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(*)=(SELECT MIN(COUNT(*))
 FROM BITLLET B, VOL V
 WHERE B.CODI_VOL=V.CODI_VOL AND
 B.DATA=V.DATA
 GROUP BY V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

--NIF i nom del(s) passatger(s) que t� mes maletes registrades al seu nom.
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                      FROM PERSONA P, MALETA M
                      WHERE M.NIF_PASSATGER = P.NIF
                      GROUP BY P.NIF, P.NOM)
ORDER BY 1

--Nom(es) de la companyia(es) que fa(n) m�s vols amb origen Italia.
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1

--Tipus d'avi�(ns) que t�(nen) m�s passatgers no espanyols. 
SELECT DISTINCT V.TIPUS_AVIO
FROM BITLLET B, VOL V, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.NIF_PASSATGER = P.NIF AND
P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                FROM BITLLET B, VOL V, PERSONA P
                WHERE V.CODI_VOL=B.CODI_VOL AND
                V.DATA=B.DATA AND
                B.NIF_PASSATGER = P.NIF AND
                P.PAIS = 'Espanya')
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS = 'Espanya')
                    GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Nom de la(/es) companyia(/es) que t�(nen) m�s bitllets reservats en algun dels seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.
SELECT V.COMPANYIA, V.CODI_VOL, V.DATA, COUNT(*) AS N_SEIENTS_RESERVATS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

--NIF i nom del(s) passatger(s) que t� mes maletes registrades al seu nom.
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>= ALL(SELECT COUNT(*)
                    FROM PERSONA P, MALETA M
                    WHERE M.NIF_PASSATGER = P.NIF
                    GROUP BY P.NIF, P.NOM)

--Tipus d'avi�(ns) que fa(n) mes vols amb origen Espanya.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Nom(es) de la(es) companyia(es) que opera(n) amb m�s tipus d'avio.
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(TIPUS_AVIO)>=ALL(SELECT COUNT(TIPUS_AVIO)
FROM VOL
GROUP BY COMPANYIA)

--Nombre de vegades que apareix la nacionalitat m�s freq�ent entre les persones de la BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.
SELECT DISTINCT PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA
GROUP BY PAIS
HAVING COUNT(*)= (SELECT MAX(COUNT(*)) 
                  FROM PERSONA
                  GROUP BY PAIS)



--Per al vol AEA2195 amb data 31/07/13, quin �s el percentatge d'ocupaci�? (NOTA: el percentatge �s sempre un numero entre 0 i 100)
SELECT (T2.OCUPATS / T1.TOTAL) * 100 
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2

--Noms dels aeroports i nom de les ciutats de destinaci� d'aquells vols a les que els falta un passatger per estar plens
SELECT DISTINCT A.NOM, A.CIUTAT
FROM (SELECT CODI_VOL, DATA, COUNT(*) AS S_TOTALS
FROM SEIENT
GROUP BY CODI_VOL, DATA) ST,
(SELECT CODI_VOL, DATA, COUNT(*) AS S_OCUPATS
FROM BITLLET
GROUP BY CODI_VOL, DATA) SO, VOL V, AEROPORT A
WHERE ST.CODI_VOL = SO.CODI_VOL AND
ST.DATA = SO.DATA AND
SO.S_OCUPATS = ST.S_TOTALS + 1 AND
SO.CODI_VOL = V.CODI_VOL AND
SO.DATA = V.DATA AND
V.DESTINACIO = A.CODI_AEROPORT
ORDER BY 1, 2

--Dades de contacte (nom, email, tel�fon) de les persones que no han volat mai a la Xina.
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM PERSONA P
MINUS
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA AND
V.DESTINACIO='PEK'
ORDER BY 1, 2, 3

--Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT (T1.TOTAL - T2.OCUPATS)
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2

--Nom dels passatgers que tenen bitllet, per� no han fet mai cap reserva
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
P.NIF NOT IN
(SELECT P.NIF
FROM PERSONA P, BITLLET B, RESERVA R
WHERE P.NIF = B.NIF_PASSATGER AND
B.NIF_PASSATGER = R.NIF_CLIENT)
ORDER BY 1

--Del total de maletes facturades, quin percentatge s�n de color vermell. (NOTA: el percentatge �s sempre un numero entre 0 i 100)
SELECT (T1.ROJAS / T2.TOTAL) * 100
FROM (SELECT COUNT(*) AS TOTAL FROM MALETA) T2,
(SELECT COUNT(*) AS ROJAS FROM MALETA WHERE COLOR = 'vermell') T1

--Percentatge d'espanyols als vols amb destinaci� Munich. (NOTA: el percentatge �s sempre un numero entre 0 i 100)
SELECT (T2.ESPANYOLS / T1.TOTAL) * 100 
FROM
(SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
V.DESTINACIO='MUC') T1,
(SELECT COUNT(*) AS ESPANYOLS
FROM BITLLET B, VOL V, PERSONA P
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
P.NIF=B.NIF_PASSATGER AND
V.DESTINACIO='MUC' AND
P.PAIS='Espanya') T2

--En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a �s dels passatgers?
SELECT S.FILA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('01/08/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2159' AND
        S.NIF_PASSATGER IS NULL AND
        S.LLETRA = 'A'
ORDER BY 1

--Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT (T1.TOTAL - T2.OCUPATS)
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2

--Companyies que volen a tots els aeroports de Espanya
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND
        A.PAIS='Espanya'
HAVING COUNT (DISTINCT A.CODI_AEROPORT)=(SELECT COUNT(CODI_AEROPORT)
FROM AEROPORT
WHERE PAIS='Espanya')
GROUP BY V.COMPANYIA

--Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/2013
SELECT S.FILA, S.LLETRA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('31/07/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2195' AND
        S.NIF_PASSATGER IS NULL
ORDER BY 1,2

--Nom del(s) passatger(s) que sempre vola amb primera classe
SELECT DISTINCT P.NOM
FROM BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
P.NOM NOT IN 
(SELECT P.NOM 
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND 
B.CLASSE = '2')

--Nom dels passatgers que tenen bitllet, per� no han fet mai cap reserva
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
P.NIF NOT IN
(SELECT P.NIF
FROM PERSONA P, BITLLET B, RESERVA R
WHERE P.NIF = B.NIF_PASSATGER AND
B.NIF_PASSATGER = R.NIF_CLIENT)
ORDER BY 1

--Numero de portes lliures de la terminal 3, area 3, del aeroport "Berlin-Sch�nefeld International Airport" el dia 04/07/2013.
SELECT (TOTALES - OCUPADAS)
FROM
(SELECT COUNT(*) AS TOTALES FROM PORTA_EMBARCAMENT
WHERE CODI_AEROPORT='SXF' AND TERMINAL='3' AND AREA='3'),
(SELECT COUNT (*) AS OCUPADAS FROM VOL 
WHERE DATA = TO_DATE('04/07/2013','DD/MM/YYYY') AND
TERMINAL='3' AND
AREA='3' AND
ORIGEN ='SXF')

--En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a �s dels passatgers?
SELECT S.FILA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('01/08/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2159' AND
        S.NIF_PASSATGER IS NULL AND
        S.LLETRA = 'A'
ORDER BY 1

--Aeroports que no tenen vol en los tres primeros meses de 2014
SELECT DISTINCT NOM
FROM AEROPORT
WHERE CODI_AEROPORT NOT IN
(SELECT DESTINACIO FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY')) AND CODI_AEROPORT NOT IN
(SELECT ORIGEN FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY'))
ORDER BY 1

--Del total de maletes facturades, quin percentatge s�n de color vermell. (NOTA: el percentatge �s sempre un numero entre 0 i 100)
SELECT (T1.ROJAS / T2.TOTAL) * 100
FROM (SELECT COUNT(*) AS TOTAL FROM MALETA) T2,
(SELECT COUNT(*) AS ROJAS FROM MALETA WHERE COLOR = 'vermell') T1

---Percentatge d'espanyols als vols amb destinaci� Munich. (NOTA: el percentatge �s sempre un numero entre 0 i 100)
SELECT (T2.ESPANYOLS / T1.TOTAL) * 100 
FROM
(SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
V.DESTINACIO='MUC') T1,
(SELECT COUNT(*) AS ESPANYOLS
FROM BITLLET B, VOL V, PERSONA P
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
P.NIF=B.NIF_PASSATGER AND
V.DESTINACIO='MUC' AND
P.PAIS='Espanya') T2

--Numero m�xim de bitllets (com a Max_Bitllets) que ha comprat en una sola reserva en Narciso Blanco.
SELECT MAX(COUNT(*)) AS MAX_BITLLETS
FROM PERSONA P, BITLLET B, RESERVA R
WHERE P.NIF = B.NIF_PASSATGER AND
P.NIF = R.NIF_CLIENT AND
B.LOCALITZADOR = R.LOCALITZADOR AND
P.NOM = 'Narciso Blanco'
GROUP BY R.DATA


8. Nom de la(/es) companyia(/es) que tÈ(nen) mÈs bitllets reservats en algun dels
seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.     ///1
SELECT V.COMPANYIA, V.CODI_VOL, V.DATA, COUNT(*) AS N_SEIENTS_RESERVATS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers  /////////1
del(s) vol(s) que tÈ(nen) el nombre de passatgers mÈs petit.      
SELECT V.ORIGEN, V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY'), COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(*)=(SELECT MIN(COUNT(*))
 FROM BITLLET B, VOL V
 WHERE B.CODI_VOL=V.CODI_VOL AND
 B.DATA=V.DATA
 GROUP BY V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

NIF i nom del(s) passatger(s) que tÈ mes maletes registrades al seu nom.       /////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                      FROM PERSONA P, MALETA M
                      WHERE M.NIF_PASSATGER = P.NIF
                      GROUP BY P.NIF, P.NOM)
ORDER BY 1                      

--Nom(es) de la(es) companyia(es) que opera(n) amb mÈs tipus d'avio.        ////////////1
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(TIPUS_AVIO)>=ALL(SELECT COUNT(TIPUS_AVIO)
FROM VOL
GROUP BY COMPANYIA)

--Nom(es) de la(es) companyia(es) que ha(n) estat reservada(es) per mÈs clients (persones) d'Amsterdam ////////////1
SELECT V.COMPANYIA
FROM VOL V, BITLLET B, RESERVA R, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, PERSONA P, BITLLET B, RESERVA R
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
GROUP BY V.COMPANYIA)
GROUP BY V.COMPANYIA
ORDER BY 1

--Nom(es) del passatger/s que ha/n facturat la maleta de mÈs pes. ////////////1
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
M.PES=(SELECT MAX(PES)
FROM MALETA)


7.-NIF i nom de la(/es) persona(/es) que ha(n) reservat mÈs bitllets. //////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM PERSONA P, RESERVA R
WHERE P.NIF=R.NIF_CLIENT
GROUP BY P.NOM)
ORDER BY 1, 2

--Color(s) de la(/es) maleta(/es) facturada(/es) mÈs pesada(/es). //1
SELECT DISTINCT COLOR
FROM MALETA M
WHERE M.PES =(SELECT MAX(PES)
FROM MALETA)
ORDER BY 1

--9. Tipus d'aviÛ(ns) que tÈ(nen) mÈs passatgers no espanyols.          //////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM BITLLET B, VOL V, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.NIF_PASSATGER = P.NIF AND
P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                FROM BITLLET B, VOL V, PERSONA P
                WHERE V.CODI_VOL=B.CODI_VOL AND
                V.DATA=B.DATA AND
                B.NIF_PASSATGER = P.NIF AND
                P.PAIS = 'Espanya')
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS = 'Espanya')
                    GROUP BY V.TIPUS_AVIO)

--Nom(es) de la companyia(es) que fa(n) mÈs vols amb origen Italia.       ////////////1
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA)
ORDER BY 1

--Tipus d'aviÛ(ns) que fa(n) mes vols amb origen Espanya.     ////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

///1
Dates de vol (DD/MM/YYYY) del(s) aviÛ(ns) amb mes capacitat (major nombre de seients)
SELECT TO_CHAR(DATA,'DD/MM/YYYY')
FROM SEIENT
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM SEIENT
GROUP BY CODI_VOL, DATA)
GROUP BY CODI_VOL, DATA
ORDER BY 1

--2.-Tipus d'avio amb mes files.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA AND
S.FILA=(SELECT MAX(FILA) FROM SEIENT)
ORDER BY 1;


SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA AND
S.FILA>=ALL(SELECT FILA FROM SEIENT)
ORDER BY 1;


////////////////1

SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA)>=ALL(SELECT COUNT(DISTINCT S.FILA) 
FROM VOL V, SEIENT S 
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;


1. Nom de totes les persones que son del mateix paÌs que Domingo Diaz Festivo (ell inclòs). /////////1
SELECT DISTINCT NOM
FROM PERSONA
WHERE PAIS=(SELECT PAIS 
FROM PERSONA WHERE NOM='Domingo Diaz Festivo')
ORDER BY 1;

3.-Nom del(s) aeroport(s) amb mÈs terminals.          ////////////1
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)>=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1;


4.-NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets///1
SELECT P.NIF, P.NOM, SUM(M.PES)
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES)<=ALL(SELECT (SUM(M.PES))
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM)

5. Nombre de vegades que apareix la nacionalitat mÈs freq¸ent entre els passatgers de la
BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.       ///////1
SELECT DISTINCT PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA
GROUP BY PAIS
HAVING COUNT(*)= (SELECT MAX(COUNT(*)) 
                  FROM PERSONA
                  GROUP BY PAIS);

////
Seients
disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/13.
SELECT S.FILA, S.LLETRA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('31/07/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2195' AND
        S.NIF_PASSATGER IS NULL
ORDER BY 1,2
////////
Companyia que vola a tots els aeroports espanyols	///1
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND
        A.PAIS='Espanya'
HAVING COUNT (DISTINCT A.CODI_AEROPORT)=(SELECT COUNT(CODI_AEROPORT)
FROM AEROPORT
WHERE PAIS='Espanya')
GROUP BY V.COMPANYIA;

SELECT DISTINCT V.COMPANYIA
FROM VOL V
WHERE V.COMPANYIA NOT IN
(SELECT COMPANYIA FROM VOL V, AEROPORT A
WHERE A.PAIS = 'Espanya' AND
(A.CODI_AEROPORT, V.COMPANYIA) NOT IN
(SELECT DESTINACIO, COMPANYIA FROM VOL) )
///////
Nom del/s passatger/s que sempre vola amb primera classe.	///1

SELECT DISTINCT P.NOM
FROM BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
P.NOM NOT IN 
(SELECT P.NOM 
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND 
B.CLASSE = '2');
////////
Numero de 
places lliures pel vol AEA2195 amb data 31/07/13.	///1
SELECT (T1.TOTAL - T2.OCUPATS)
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2;
/////////
--Per al vol AEA2195 amb data 31/07/13, quin Ès el percentatge d'ocupaciÛ?///1
SELECT (T2.OCUPATS / T1.TOTAL) * 100 
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2;
/////
--Noms dels aeroports i nom de les ciutats de destinaciÛ d'aquells vols a les que els falta un passatger per estar plens///1
SELECT DISTINCT A.NOM, A.CIUTAT
FROM (SELECT CODI_VOL, DATA, COUNT(*) AS S_TOTALS
FROM SEIENT
GROUP BY CODI_VOL, DATA) ST,
(SELECT CODI_VOL, DATA, COUNT(*) AS S_OCUPATS
FROM BITLLET
GROUP BY CODI_VOL, DATA) SO, VOL V, AEROPORT A
WHERE ST.CODI_VOL = SO.CODI_VOL AND
ST.DATA = SO.DATA AND
SO.S_OCUPATS = ST.S_TOTALS + 1 AND
SO.CODI_VOL = V.CODI_VOL AND
SO.DATA = V.DATA AND
V.DESTINACIO = A.CODI_AEROPORT
ORDER BY 1, 2
////////////
En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a ˙s dels passatgers? //////1
SELECT S.FILA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('01/08/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2159' AND
        S.NIF_PASSATGER IS NULL AND
        S.LLETRA = 'A'
ORDER BY 1

////////////////
Numero de portes que queden lliures de la terminal 3, area 3, del aeroport 'Berlin-Schˆnefeld International Airport' el dia 04/07/2013	///1
SELECT (TOTALES - OCUPADAS)
FROM
(SELECT COUNT(*) AS TOTALES FROM PORTA_EMBARCAMENT
WHERE CODI_AEROPORT='SXF' AND TERMINAL='3' AND AREA='3'),
(SELECT COUNT (*) AS OCUPADAS FROM VOL 
WHERE DATA = TO_DATE('04/07/2013','DD/MM/YYYY') AND
TERMINAL='3' AND
AREA='3' AND
ORIGEN ='SXF')
///////
--Aeroports que no tenen vol en los tres primeros meses de 2014	///1
SELECT DISTINCT NOM
FROM AEROPORT
WHERE CODI_AEROPORT NOT IN
(SELECT DESTINACIO FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY')) AND CODI_AEROPORT NOT IN
(SELECT ORIGEN FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY'))
ORDER BY 1
//////////

--Del total de maletes facturades, quin percentatge sÛn de color vermell. (NOTA: el percentatge Ès sempre un numero entre 0 i 100)//////1
SELECT (T1.ROJAS / T2.TOTAL) * 100
FROM (SELECT COUNT(*) AS TOTAL FROM MALETA) T2,
(SELECT COUNT(*) AS ROJAS FROM MALETA WHERE COLOR = 'vermell') T1

///////////////////

--Companyies que tenen, almenys, els mateixos tipus d'avions que Vueling Airlines. (El resultat tambÈ ha de tornar  Vueling airlines)

SELECT DISTINCT COMPANYIA
FROM VOL
WHERE TIPUS_AVIO IN 
(SELECT DISTINCT TIPUS_AVIO FROM VOL WHERE COMPANYIA='VUELING AIRLINES')
ORDER BY 1

////////////

Dades de contacte (nom, email, telËfon) de les persones que no han volat mai a la Xina.	///1
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM PERSONA P
MINUS
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA AND
V.DESTINACIO='PEK'
ORDER BY 1, 2, 3

/////////////////////////////////////////////////

1. Nom de totes les persones que son del mateix paÌs que Domingo Diaz Festivo (ell inclòs). /////////1
SELECT DISTINCT NOM
FROM PERSONA
WHERE PAIS=(SELECT PAIS 
FROM PERSONA WHERE NOM='Domingo Diaz Festivo')
ORDER BY 1;

3.-Nom del(s) aeroport(s) amb mÈs terminals.          ////////////1
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)>=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1;

//1
4.-NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets
SELECT P.NIF, P.NOM, SUM(M.PES)
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES)<=ALL(SELECT (SUM(M.PES))
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM)

5. Nombre de vegades que apareix la nacionalitat mÈs freq¸ent entre els passatgers de la
BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.       //////////1
SELECT DISTINCT PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA
GROUP BY PAIS
HAVING COUNT(*)= (SELECT MAX(COUNT(*)) 
                  FROM PERSONA
                  GROUP BY PAIS);

7.-NIF i nom de la(/es) persona(/es) que ha(n) reservat mÈs bitllets. //////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM PERSONA P, RESERVA R
WHERE P.NIF=R.NIF_CLIENT
GROUP BY P.NOM)
ORDER BY 1, 2

Color(s) de la(/es) maleta(/es) facturada(/es) mÈs pesada(/es). //1
SELECT DISTINCT COLOR
FROM MALETA M
WHERE M.PES =(SELECT MAX(PES)
FROM MALETA)
ORDER BY 1

--9. Tipus d'aviÛ(ns) que tÈ(nen) mÈs passatgers no espanyols.          //////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM BITLLET B, VOL V, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.NIF_PASSATGER = P.NIF AND
P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                FROM BITLLET B, VOL V, PERSONA P
                WHERE V.CODI_VOL=B.CODI_VOL AND
                V.DATA=B.DATA AND
                B.NIF_PASSATGER = P.NIF AND
                P.PAIS = 'Espanya')
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS = 'Espanya')
                    GROUP BY V.TIPUS_AVIO)

Nom(es) de la companyia(es) que fa(n) mÈs vols amb origen Italia.       ////////////1
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA)
ORDER BY 1

--Tipus d'aviÛ(ns) que fa(n) mes vols amb origen Espanya.     ////////1
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

///1
--Dates de vol (DD/MM/YYYY) del(s) aviÛ(ns) amb mes capacitat (major nombre de seients)
SELECT TO_CHAR(DATA,'DD/MM/YYYY')
FROM SEIENT
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM SEIENT
GROUP BY CODI_VOL, DATA)
GROUP BY CODI_VOL, DATA
ORDER BY 1

--8. Nom de la(/es) companyia(/es) que tÈ(nen) mÈs bitllets reservats en algun dels
seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.     ///1
SELECT V.COMPANYIA, V.CODI_VOL, V.DATA, COUNT(*) AS N_SEIENTS_RESERVATS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers  /////////1
del(s) vol(s) que tÈ(nen) el nombre de passatgers mÈs petit.      
SELECT V.ORIGEN, V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY'), COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(*)=(SELECT MIN(COUNT(*))
 FROM BITLLET B, VOL V
 WHERE B.CODI_VOL=V.CODI_VOL AND
 B.DATA=V.DATA
 GROUP BY V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

--NIF i nom del(s) passatger(s) que tÈ mes maletes registrades al seu nom.       /////////1
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                      FROM PERSONA P, MALETA M
                      WHERE M.NIF_PASSATGER = P.NIF
                      GROUP BY P.NIF, P.NOM)
ORDER BY 1                      

--Nom(es) de la(es) companyia(es) que opera(n) amb mÈs tipus d'avio.        ////////////1
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(TIPUS_AVIO)>=ALL(SELECT COUNT(TIPUS_AVIO)
FROM VOL
GROUP BY COMPANYIA)

--Nom(es) de la(es) companyia(es) que ha(n) estat reservada(es) per mÈs clients (persones) d'Amsterdam ////////////1
SELECT V.COMPANYIA
FROM VOL V, BITLLET B, RESERVA R, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, PERSONA P, BITLLET B, RESERVA R
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.LOCALITZADOR=R.LOCALITZADOR AND
R.NIF_CLIENT=P.NIF AND
P.POBLACIO='Amsterdam'
GROUP BY V.COMPANYIA)
GROUP BY V.COMPANYIA
ORDER BY 1

--Nom(es) del passatger/s que ha/n facturat la maleta de mÈs pes. ////////////1
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
M.PES=(SELECT MAX(PES)
FROM MALETA)

(Reserves vols) NIF, nom, poblaciÛ i paÌs de totes les persones existents a la base de dades.	///1
SELECT NIF, NOM, POBLACIO, PAIS
FROM PERSONA					
ORDER BY 1, 2, 3, 4

/////////////////////////////////////////////////

--Companyies que volen a tots els aeroports de Espanya ///1
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.DESTINACIO = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'					
GROUP BY V.COMPANYIA
HAVING COUNT(DISTINCT A.CODI_AEROPORT)=(SELECT COUNT(CODI_AEROPORT)
FROM AEROPORT WHERE PAIS='Espanya')

/////////////////////////////////////////////////
	
--Nombre de persones mexicanes.		///1
SELECT COUNT(*)
FROM PERSONA					
WHERE PAIS = 'Mexic'

//////////////////////////////////////////////////

--NIF i nom del(s) passatger(s) que tÈ mes maletes registrades al seu nom.	///1
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM, M.NIF_PASSATGER			
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM, M.NIF_PASSATGER)
ORDER BY 1, 2

/////////////////////////////////////////////////

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.	///1
SELECT DISTINCT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V						
GROUP BY V.COMPANYIA
ORDER BY 1,2

//////////////////////////////////////////////////

--Nom de la(s) companyia(es) amb vols d'origen a tots els aeroports de la Xina.	///1
(SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Xina')
MINUS
(SELECT DISTINCT V.COMPANYIA				
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Xina' AND V.COMPANYIA NOT IN (SELECT V.COMPANYIA FROM VOL V, AEROPORT A
WHERE A.PAIS = 'Xina'))

//////////////////////////////////////////////////

Nom dels passatgers que tenen bitllet, però no han fet mai cap reserva	///1
(SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER)
MINUS (SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, RESERVA R			
WHERE P.NIF = B.NIF_PASSATGER AND
R.NIF_CLIENT = P.NIF AND
B.NIF_PASSATGER = R.NIF_CLIENT)

//////////////////////////////////////////////////

El pes de les maletes (com a pes_total) dels passatgers italians.///1
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE P.NIF = M.NIF_PASSATGER AND			
P.PAIS = 'Italia'
GROUP BY M.NIF_PASSATGER

//////////////////////////////////////////////////

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'‡rea i el nombre de portes d'aquesta ‡rea.///1
SELECT DISTINCT A.NOM, PE.TERMINAL, PE.AREA, COUNT(PE.PORTA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE			
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3, 4

//////////////////////////////////////////////////

--Percentatge d'espanyols als vols amb destinaciÛ Munich. (NOTA: el percentatge Ès sempre un numero entre 0 i 100)///1

SELECT (T2.ESPANYOLS / T1.TOTAL) * 100 
FROM
(SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
V.DESTINACIO='MUC') T1,
(SELECT COUNT(*) AS ESPANYOLS
FROM BITLLET B, VOL V, PERSONA P
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
P.NIF=B.NIF_PASSATGER AND
V.DESTINACIO='MUC' AND
P.PAIS='Espanya') T2

////////////



--Nombre de passatgers dels vols d'IBERIA per paÔsos. Es demana paÌs i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Nombre m‡xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

--Nom dels passatgers que han facturat mÈs d'una maleta vermella
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Nom de les Companyies amb mËs de 10 vols.
SELECT V.COMPANYIA
FROM VOL V
GROUP BY V.COMPANYIA
HAVING COUNT(*)>10
ORDER BY 1

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT COUNT(*)/(COUNT(DISTINCT PE.CODI_AEROPORT))
FROM PORTA_EMBARCAMENT PE
ORDER BY 1

--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco tÈ el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGER
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL ='IBE2119' AND
B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
AND P.NOM LIKE '%Blanco%'

--Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar tambÈ el nombre de vegades que han volat com a N_VEGADES.
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) > 4
ORDER BY 2,1

--Nombre de passatgers agrupats segons paÌs de destinaciÛ (en el vol). Mostreu paÌs i nombre de passatgers.
SELECT A.PAIS, COUNT(*)
FROM BITLLET B, VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND 
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA
GROUP BY A.PAIS
ORDER BY 1,2

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'‡rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Nacionalitats representades per mÈs de dos passatgers amb bitllets.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(P.PAIS)>2
ORDER BY 1

--Nombre de passatgers dels vols d'IBERIA per paÔsos. Es demana paÌs i nombre de passatgers
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1
Quantes maletes ha facturat en Aitor Tilla al vol amb destinaciÛ Pekin de la data 1/10/2013?
SELECT COUNT(*)
FROM BITLLET B, VOL V, MALETA M, PERSONA P
WHERE V.DESTINACIO='PEK' AND
P.NOM='Aitor Tilla' AND
P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.DATA=V.DATA AND
B.DATA=M.DATA AND
B.CODI_VOL=M.CODI_VOL AND
M.DATA=TO_DATE('01/10/2013','DD/MM/YYYY')
Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--Nom dels passatgers que han facturat mÈs d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN=PE.CODI_AEROPORT AND
V.TERMINAL=PE.TERMINAL AND
V.AREA=PE.AREA AND
V.PORTA=PE.PORTA
HAVING COUNT(*)>1
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
ORDER BY 1,2,3,4
Nombre de persones mexicanes.
SELECT COUNT(*)
FROM PERSONA 
WHERE PAIS='Mexic'

--Nombre de passatgers dels vols d'IBERIA per paÔsos. Es demana paÌs i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--NIF i nom dels passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, NOM, CODI_VOL, DATA i PES_TOTAL
SELECT M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.CODI_VOL=M.CODI_VOL AND
B.DATA=M.DATA
GROUP BY M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA
HAVING SUM(M.PES)>15
ORDER BY 1, 2, 3, 4, 5

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1

--El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Italia'
ORDER BY 1


--Nacionalitats representades per mÈs de dos passatgers amb bitllets.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(P.PAIS)>2
ORDER BY 1

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'‡rea i el nombre de portes d'aquesta ‡rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'‡rea i el nombre de portes d'aquesta ‡rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'‡rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--Tipus de avions que volen a l'aeroport de Orly
SELECT DISTINCT TIPUS_AVIO
FROM VOL
WHERE DESTINACIO = 'ORY'
ORDER BY 1

Nom i edat dels passatgers del vol AEA2159 del 25/07/2013.
SELECT DISTINCT P.NOM, ROUND((SYSDATE - P.DATA_NAIXEMENT)/365)
FROM PERSONA P, BITLLET B
WHERE B.CODI_VOL = 'AEA2159'
AND B.NIF_PASSATGER = P.NIF
AND B.DATA = TO_DATE('25/07/2013', 'DD/MM/YYYY')
ORDER BY 1, 2

Nom de les passatgers que han nascut entre el 05/06/1964 i el 03/09/1985.
SELECT DISTINCT NOM
FROM PERSONA
WHERE DATA_NAIXEMENT >= TO_DATE('05/06/1964','DD/MM/YYYY')
AND DATA_NAIXEMENT <= TO_DATE('03/09/1985','DD/MM/YYYY')
ORDER BY 1

(Reserves Vols) Localitzador(s) i nom de les persones pel vol KLM4303.
SELECT DISTINCT B.LOCALITZADOR, P.NOM
FROM PERSONA P, VOL V, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
AND V.CODI_VOL = 'KLM4303'
AND B.CODI_VOL = V.CODI_VOL
ORDER BY 1, 2

Forma de pagament de les reserves fetes per a persones de nacionalitat alemana
SELECT R.MODE_PAGAMENT
FROM RESERVA R, PERSONA P
WHERE R.NIF_CLIENT = P.NIF
AND P.PAIS = 'Alemanya'
ORDER BY 1

Ciutats espanyoles amb aeroports.
SELECT DISTINCT CIUTAT
FROM AEROPORT
WHERE PAIS = 'Espanya'
ORDER BY 1

Nom del passatger i pes de les maletes facturades per passatgers espanyols.
SELECT DISTINCT P.NOM, M.PES
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
AND P.PAIS = 'Espanya'
ORDER BY 1, 2

--Nacionalitat dels passatgers dels vols amb destinaciÛ Paris Orly Airport.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B, VOL V
WHERE B.NIF_PASSATGER = P.NIF
AND B.CODI_VOL = V.CODI_VOL
AND V.DESTINACIO = 'ORY'
ORDER BY 1

--Codi de vol i data dels bitllets del passatger Alan Brito.
SELECT B.CODI_VOL, B.DATA
FROM BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND P.NOM = 'Alan Brito'
ORDER BY 1, 2

--(Reserves Vol) Nom dels passatgers que han nascut abans de l'any 1975.
SELECT DISTINCT NOM
FROM PERSONA
WHERE DATA_NAIXEMENT < TO_DATE('01/01/1975','DD/MM/YYYY')
ORDER BY 1

--El codi de la maleta, el pes i les mides de les maletes que ha facturat el passatger Jose Luis Lamata Feliz en els seus vols.
SELECT M.CODI_MALETA, M.PES, M.MIDES
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER = P.NIF
AND P.NOM = 'Jose Luis Lamata Feliz'
ORDER BY 1, 2, 3

--Nom dels passatgers que tenen alguna maleta que pesa mÈs de 10 kilos. Especifica tambÈ en quin vol.
SELECT DISTINCT P.NOM, M.CODI_VOL
FROM PERSONA P, MALETA M, VOL V
WHERE M.CODI_VOL = V.CODI_VOL
AND P.NIF = M.NIF_PASSATGER
AND M.PES > 10
ORDER BY 1, 2

--(Reserves Vols) Codi de tots els vols amb bitllets amb data d'abans de l'1 de Desembre de 2013.
SELECT DISTINCT V.CODI_VOL
FROM RESERVA R, VOL V, BITLLET B, PERSONA P
WHERE R.NIF_CLIENT = P.NIF
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA < TO_DATE('01/12/2013', 'DD/MM/YYYY')
ORDER BY 1

--Les companyies amb vols al mes d'Agost de 2013.
SELECT DISTINCT COMPANYIA
FROM VOL
WHERE DATA >= TO_DATE('01/08/2013', 'DD/MM/YYYY')
AND DATA <= TO_DATE('31/08/2013', 'DD/MM/YYYY')
ORDER BY 1

--Nom i PoblaciÛ dels passatgers que han viatjat algun cop en 1a classe.
SELECT DISTINCT P.NOM, P.POBLACIO
FROM PERSONA P, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
AND B.CLASSE = '1'
ORDER BY 1, 2

(Reserves Vols) Localitzador(s) i nom de les persones pel vol KLM4303.
SELECT DISTINCT B.LOCALITZADOR, P.NOM
FROM PERSONA P, VOL V, BITLLET B
WHERE B.NIF_PASSATGER = P.NIF
AND V.CODI_VOL = 'KLM4303'
AND B.CODI_VOL = V.CODI_VOL
ORDER BY 1, 2

--Nom dels persones vegetarians.
SELECT DISTINCT P.NOM
FROM PERSONA P
WHERE P.OBSERVACIONS = 'Vegetaria/na'
ORDER BY 1

--El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS pes_total
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER=P.NIF AND P.PAIS='Italia'

--Quan pes ha facturat l'Alberto Carlos Huevos al vol amb destinaciÛ Rotterdam de la data 02 de juny del 2013?
SELECT SUM(M.PES)
FROM MALETA M, PERSONA P, VOL V
WHERE M.NIF_PASSATGER = P.NIF AND
M.CODI_VOL = V.CODI_VOL AND
P.NOM = 'Alberto Carlos Huevos' AND
V.DESTINACIO = 'RTM' AND
V.DATA = M.DATA AND
M.DATA = TO_DATE('02/06/2013','DD/MM/YYYY')

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--Nombre m‡xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco tÈ el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGER
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL ='IBE2119' AND
B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
AND P.NOM LIKE '%Blanco%'

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')

--Nacionalitats representades per mÈs de dos passatgers amb bitllets.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.PAIS
HAVING COUNT(P.PAIS)>2
ORDER BY 1

--Nom dels passatgers que han facturat mÈs d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT COUNT(*)/(COUNT(DISTINCT PE.CODI_AEROPORT))
FROM PORTA_EMBARCAMENT PE
ORDER BY 1

--Nombre de passatgers dels vols d'IBERIA per paÔsos. Es demana paÌs i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'‡rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'‡rea i el nombre de portes d'aquesta ‡rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'‡rea i el nombre de portes d'aquesta ‡rea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(*)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1, 2, 3

El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Italia'
ORDER BY 1

Nombre m‡xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

NIF i nom dels passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, NOM, CODI_VOL, DATA i PES_TOTAL
SELECT M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.CODI_VOL=M.CODI_VOL AND
B.DATA=M.DATA
GROUP BY M.NIF_PASSATGER, P.NOM, M.CODI_VOL, M.DATA
HAVING SUM(M.PES)>15
ORDER BY 1, 2, 3, 4, 5

--Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(M.PES) 
FROM MALETA M, PERSONA P
WHERE P.NIF=M.NIF_PASSATGER AND P.PAIS='Espanya'
ORDER BY 1

--Nombre de passatgers dels vols d'IBERIA per paÔsos. Es demana paÌs i nombre de passatgers.
SELECT DISTINCT P.PAIS, COUNT(*)
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF=B.NIF_PASSATGER AND B.CODI_VOL=V.CODI_VOL AND B.DATA=V.DATA
AND V.COMPANYIA='IBERIA'
GROUP BY P.PAIS
ORDER BY 1

--Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'‡rees
SELECT DISTINCT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA)
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1, 2

--Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar tambÈ el nombre de vegades que han volat com a N_VEGADES.
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) > 4
ORDER BY 2,1

--Nombre de passatgers agrupats segons paÌs de destinaciÛ (en el vol). Mostreu paÌs i nombre de passatgers
SELECT A.PAIS, COUNT(*)
FROM BITLLET B, VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND 
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA
GROUP BY A.PAIS
ORDER BY 1,2

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1

--Nom de les Companyies amb mËs de 10 vols.
SELECT V.COMPANYIA
FROM VOL V
GROUP BY V.COMPANYIA
HAVING COUNT(*)>10
ORDER BY 1

--Quantes maletes ha facturat en Aitor Tilla al vol amb destinaciÛ Pekin de la data 1/10/2013?
SELECT COUNT(*)
FROM BITLLET B, VOL V, MALETA M, PERSONA P
WHERE V.DESTINACIO='PEK' AND
P.NOM='Aitor Tilla' AND
P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
B.DATA=V.DATA AND
B.DATA=M.DATA AND
B.CODI_VOL=M.CODI_VOL AND
M.DATA=TO_DATE('01/10/2013','DD/MM/YYYY')

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.
SELECT COUNT(*) AS N_BUTAQUES_RESERVADES
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
B.CODI_VOL ='VLG2117' AND
B.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1

--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco tÈ el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGER
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL ='IBE2119' AND
B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
AND P.NOM LIKE '%Blanco%'

---Nombre m‡xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS MAX_MALETES
FROM MALETA
GROUP BY DATA, NIF_PASSATGER

--N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN=PE.CODI_AEROPORT AND
V.TERMINAL=PE.TERMINAL AND
V.AREA=PE.AREA AND
V.PORTA=PE.PORTA
HAVING COUNT(*)>1
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA 
ORDER BY 1,2,3,4

--Nom dels passatgers que han facturat mÈs d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Nombre de passatgers agrupats segons paÌs de destinaciÛ (en el vol). Mostreu paÌs i nombre de passatgers.
SELECT A.PAIS, COUNT(*)
FROM BITLLET B, VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND 
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA
GROUP BY A.PAIS
ORDER BY 1,2

--Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
SELECT COUNT(*)/(COUNT(DISTINCT PE.CODI_AEROPORT))
FROM PORTA_EMBARCAMENT PE
ORDER BY 1

--Nom dels passatgers que han facturat mÈs d'una maleta vermella.
SELECT P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER AND
M.COLOR='vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1

--Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M
WHERE M.CODI_VOL='KLM4304' AND
M.DATA=TO_DATE('09/10/2013','DD/MM/YYYY')


--Tipus d'aviÛ(ns) amb mÈs files.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA)>=ALL(SELECT COUNT(DISTINCT S.FILA) 
FROM VOL V, SEIENT S 
WHERE V.CODI_VOL=S.CODI_VOL AND
V.DATA=S.DATA
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Nom del(s) aeroport(s) amb el mÌnim n˙mero de terminals.
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)<=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1

--NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets.

SELECT P.NIF, P.NOM, SUM(M.PES)
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES)<=ALL(SELECT (SUM(M.PES))
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
      B.NIF_PASSATGER=M.NIF_PASSATGER AND
      B.CODI_VOL=M.CODI_VOL AND
      B.DATA=M.DATA
GROUP BY P.NIF, P.NOM)

--Nom(es) del(s) aeroport(s) amb mÈs terminals.
SELECT DISTINCT A.NOM
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM
HAVING COUNT(DISTINCT PE.TERMINAL)>=ALL(SELECT COUNT(DISTINCT PE.TERMINAL) 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT=PE.CODI_AEROPORT
GROUP BY A.NOM)
ORDER BY 1

--Nom(es) del passatger/s que ha/n facturat la maleta de mÈs pes.
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF=B.NIF_PASSATGER AND
B.NIF_PASSATGER=M.NIF_PASSATGER AND
M.PES=(SELECT MAX(PES)
FROM MALETA)

--Nom de totes les persones que son del mateix paÌs que "Domingo Diaz Festivo" (ell inclòs).
SELECT DISTINCT NOM
FROM PERSONA
WHERE PAIS=(SELECT PAIS 
FROM PERSONA WHERE NOM='Domingo Diaz Festivo')
ORDER BY 1

--Nom(es) de la companyia(es) que fa(n) mÈs vols amb origen Italia.
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA
HAVING COUNT(*)=(SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN=A.CODI_AEROPORT AND
A.PAIS='Italia'
GROUP BY COMPANYIA)
ORDER BY 1

--Tipus d'aviÛ(ns) que fa(n) mes vols amb origen Espanya.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers del(s) vol(s) que tÈ(nen) el nombre de passatges mÈs petit.
SELECT V.ORIGEN, V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY'), COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(*)=(SELECT MIN(COUNT(*))
 FROM BITLLET B, VOL V
 WHERE B.CODI_VOL=V.CODI_VOL AND
 B.DATA=V.DATA
 GROUP BY V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

--NIF i nom del(s) passatger(s) que tÈ mes maletes registrades al seu nom.
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                      FROM PERSONA P, MALETA M
                      WHERE M.NIF_PASSATGER = P.NIF
                      GROUP BY P.NIF, P.NOM)
ORDER BY 1

--Nom(es) de la companyia(es) que fa(n) mÈs vols amb origen Italia.
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1

--Tipus d'aviÛ(ns) que tÈ(nen) mÈs passatgers no espanyols. 
SELECT DISTINCT V.TIPUS_AVIO
FROM BITLLET B, VOL V, PERSONA P
WHERE V.CODI_VOL=B.CODI_VOL AND
V.DATA=B.DATA AND
B.NIF_PASSATGER = P.NIF AND
P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                FROM BITLLET B, VOL V, PERSONA P
                WHERE V.CODI_VOL=B.CODI_VOL AND
                V.DATA=B.DATA AND
                B.NIF_PASSATGER = P.NIF AND
                P.PAIS = 'Espanya')
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS NOT IN (SELECT DISTINCT V.TIPUS_AVIO
                    FROM BITLLET B, VOL V, PERSONA P
                    WHERE V.CODI_VOL=B.CODI_VOL AND
                    V.DATA=B.DATA AND
                    B.NIF_PASSATGER = P.NIF AND
                    P.PAIS = 'Espanya')
                    GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Nom de la(/es) companyia(/es) que tÈ(nen) mÈs bitllets reservats en algun dels seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.
SELECT V.COMPANYIA, V.CODI_VOL, V.DATA, COUNT(*) AS N_SEIENTS_RESERVATS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA
HAVING COUNT(*)>=ALL(SELECT COUNT(*)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA
GROUP BY V.COMPANYIA, V.CODI_VOL, V.DATA)
ORDER BY 1, 2, 3, 4

--NIF i nom del(s) passatger(s) que tÈ mes maletes registrades al seu nom.
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE M.NIF_PASSATGER = P.NIF
GROUP BY P.NIF, P.NOM
HAVING COUNT(*)>= ALL(SELECT COUNT(*)
                    FROM PERSONA P, MALETA M
                    WHERE M.NIF_PASSATGER = P.NIF
                    GROUP BY P.NIF, P.NOM)

--Tipus d'aviÛ(ns) que fa(n) mes vols amb origen Espanya.
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(V.CODI_VOL)>=ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT AND
A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1

--Nom(es) de la(es) companyia(es) que opera(n) amb mÈs tipus d'avio.
SELECT COMPANYIA
FROM VOL
GROUP BY COMPANYIA
HAVING COUNT(TIPUS_AVIO)>=ALL(SELECT COUNT(TIPUS_AVIO)
FROM VOL
GROUP BY COMPANYIA)

--Nombre de vegades que apareix la nacionalitat mÈs freq¸ent entre les persones de la BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.
SELECT DISTINCT PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA
GROUP BY PAIS
HAVING COUNT(*)= (SELECT MAX(COUNT(*)) 
                  FROM PERSONA
                  GROUP BY PAIS)









--Per al vol AEA2195 amb data 31/07/13, quin Ès el percentatge d'ocupaciÛ? (NOTA: el percentatge Ès sempre un numero entre 0 i 100)
SELECT (T2.OCUPATS / T1.TOTAL) * 100 
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2

--Noms dels aeroports i nom de les ciutats de destinaciÛ d'aquells vols a les que els falta un passatger per estar plens
SELECT DISTINCT A.NOM, A.CIUTAT
FROM (SELECT CODI_VOL, DATA, COUNT(*) AS S_TOTALS
FROM SEIENT
GROUP BY CODI_VOL, DATA) ST,
(SELECT CODI_VOL, DATA, COUNT(*) AS S_OCUPATS
FROM BITLLET
GROUP BY CODI_VOL, DATA) SO, VOL V, AEROPORT A
WHERE ST.CODI_VOL = SO.CODI_VOL AND
ST.DATA = SO.DATA AND
SO.S_OCUPATS = ST.S_TOTALS + 1 AND
SO.CODI_VOL = V.CODI_VOL AND
SO.DATA = V.DATA AND
V.DESTINACIO = A.CODI_AEROPORT
ORDER BY 1, 2

--Dades de contacte (nom, email, telËfon) de les persones que no han volat mai a la Xina.
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM PERSONA P
MINUS
SELECT DISTINCT P.NOM, P.MAIL, P.TELEFON
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
B.CODI_VOL=V.CODI_VOL AND
B.DATA=V.DATA AND
V.DESTINACIO='PEK'
ORDER BY 1, 2, 3

Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT (T1.TOTAL - T2.OCUPATS)
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2

Nom dels passatgers que tenen bitllet, però no han fet mai cap reserva
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
P.NIF NOT IN
(SELECT P.NIF
FROM PERSONA P, BITLLET B, RESERVA R
WHERE P.NIF = B.NIF_PASSATGER AND
B.NIF_PASSATGER = R.NIF_CLIENT)
ORDER BY 1

Del total de maletes facturades, quin percentatge sÛn de color vermell. (NOTA: el percentatge Ès sempre un numero entre 0 i 100)
SELECT (T1.ROJAS / T2.TOTAL) * 100
FROM (SELECT COUNT(*) AS TOTAL FROM MALETA) T2,
(SELECT COUNT(*) AS ROJAS FROM MALETA WHERE COLOR = 'vermell') T1

--Percentatge d'espanyols als vols amb destinaciÛ Munich. (NOTA: el percentatge Ès sempre un numero entre 0 i 100)
SELECT (T2.ESPANYOLS / T1.TOTAL) * 100 
FROM
(SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
V.DESTINACIO='MUC') T1,
(SELECT COUNT(*) AS ESPANYOLS
FROM BITLLET B, VOL V, PERSONA P
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
P.NIF=B.NIF_PASSATGER AND
V.DESTINACIO='MUC' AND
P.PAIS='Espanya') T2

En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a ˙s dels passatgers?
SELECT S.FILA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('01/08/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2159' AND
        S.NIF_PASSATGER IS NULL AND
        S.LLETRA = 'A'
ORDER BY 1

Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT (T1.TOTAL - T2.OCUPATS)
FROM (SELECT COUNT(*) AS TOTAL
FROM SEIENT S, VOL V
WHERE S.CODI_VOL = V.CODI_VOL AND
S.DATA = V.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T1,
(SELECT COUNT(*) AS OCUPATS
FROM VOL V, BITLLET B
WHERE V.CODI_VOL = B.CODI_VOL AND
V.DATA = B.DATA AND
V.CODI_VOL = 'AEA2195' AND
TO_CHAR(V.DATA,'DD/MM/YY') = '31/07/13') T2

Companyies que volen a tots els aeroports de Espanya
SELECT DISTINCT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.DESTINACIO=A.CODI_AEROPORT AND
        A.PAIS='Espanya'
HAVING COUNT (DISTINCT A.CODI_AEROPORT)=(SELECT COUNT(CODI_AEROPORT)
FROM AEROPORT
WHERE PAIS='Espanya')
GROUP BY V.COMPANYIA

Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/2013
SELECT S.FILA, S.LLETRA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('31/07/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2195' AND
        S.NIF_PASSATGER IS NULL
ORDER BY 1,2

Nom del(s) passatger(s) que sempre vola amb primera classe
SELECT DISTINCT P.NOM
FROM BITLLET B, PERSONA P
WHERE P.NIF=B.NIF_PASSATGER AND
P.NOM NOT IN 
(SELECT P.NOM 
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER AND 
B.CLASSE = '2')

Nom dels passatgers que tenen bitllet, però no han fet mai cap reserva
SELECT DISTINCT P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER AND
P.NIF NOT IN
(SELECT P.NIF
FROM PERSONA P, BITLLET B, RESERVA R
WHERE P.NIF = B.NIF_PASSATGER AND
B.NIF_PASSATGER = R.NIF_CLIENT)
ORDER BY 1

Numero de portes lliures de la terminal 3, area 3, del aeroport "Berlin-Schˆnefeld International Airport" el dia 04/07/2013.
SELECT (TOTALES - OCUPADAS)
FROM
(SELECT COUNT(*) AS TOTALES FROM PORTA_EMBARCAMENT
WHERE CODI_AEROPORT='SXF' AND TERMINAL='3' AND AREA='3'),
(SELECT COUNT (*) AS OCUPADAS FROM VOL 
WHERE DATA = TO_DATE('04/07/2013','DD/MM/YYYY') AND
TERMINAL='3' AND
AREA='3' AND
ORIGEN ='SXF')

En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a ˙s dels passatgers?
SELECT S.FILA
FROM SEIENT S, VOL V
WHERE S.CODI_VOL=V.CODI_VOL AND
        S.DATA=V.DATA AND
        TO_DATE('01/08/2013','DD/MM/YYYY')=S.DATA AND
        S.CODI_VOL='AEA2159' AND
        S.NIF_PASSATGER IS NULL AND
        S.LLETRA = 'A'
ORDER BY 1

Aeroports que no tenen vol en los tres primeros meses de 2014
SELECT DISTINCT NOM
FROM AEROPORT
WHERE CODI_AEROPORT NOT IN
(SELECT DESTINACIO FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY')) AND CODI_AEROPORT NOT IN
(SELECT ORIGEN FROM VOL
WHERE DATA >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND
DATA < TO_DATE('01/04/2014', 'DD/MM/YYYY'))
ORDER BY 1

Del total de maletes facturades, quin percentatge sÛn de color vermell. (NOTA: el percentatge Ès sempre un numero entre 0 i 100)
SELECT (T1.ROJAS / T2.TOTAL) * 100
FROM (SELECT COUNT(*) AS TOTAL FROM MALETA) T2,
(SELECT COUNT(*) AS ROJAS FROM MALETA WHERE COLOR = 'vermell') T1

--Percentatge d'espanyols als vols amb destinaciÛ Munich. (NOTA: el percentatge Ès sempre un numero entre 0 i 100)
SELECT (T2.ESPANYOLS / T1.TOTAL) * 100 
FROM
(SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
V.DESTINACIO='MUC') T1,
(SELECT COUNT(*) AS ESPANYOLS
FROM BITLLET B, VOL V, PERSONA P
WHERE B.CODI_VOL = V.CODI_VOL AND
B.DATA = V.DATA AND
P.NIF=B.NIF_PASSATGER AND
V.DESTINACIO='MUC' AND
P.PAIS='Espanya') T2

--Numero m‡xim de bitllets (com a Max_Bitllets) que ha comprat en una sola reserva en Narciso Blanco.
SELECT MAX(COUNT(*)) AS MAX_BITLLETS
FROM PERSONA P, BITLLET B, RESERVA R
WHERE P.NIF = B.NIF_PASSATGER AND
P.NIF = R.NIF_CLIENT AND
B.LOCALITZADOR = R.LOCALITZADOR AND
P.NOM = 'Narciso Blanco'
GROUP BY R.DATA

--/*1 (Reserves Vol) Nombre de persones mexicanes.*/
SELECT COUNT(NOM)
FROM PERSONA
WHERE PAIS = 'Mexic'
ORDER BY 1;

--/*2 (Reserves vol) Nombre de maletes que ha facturat Chema Pamundi per cada vol que ha fet. Volem recuperar codi del vol, data i nombre de maletes.*/
SELECT B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, COUNT(M.CODI_MALETA) AS N_MALETES
FROM BITLLET B, PERSONA P, MALETA M
WHERE B.NIF_PASSATGER = P.NIF
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND P.NOM = 'Chema Pamundi'
GROUP BY B.CODI_VOL, B.DATA, M.CODI_MALETA
ORDER BY 1, 2, 3;

--/*3 (Reserves Vol) Nom dels passatgers que han facturat mÈs d'una maleta vermella.*/
SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
  AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(P.NOM) > 1
ORDER BY 1;

/*4 (Reserves vol) N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.*/
SELECT PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
  AND V.TERMINAL = PE.TERMINAL
  AND V.AREA = PE.AREA
  AND V.PORTA = PE.PORTA
GROUP BY PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA, PE.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;

/*5 (Reserves Vol) Passatgers que han facturat mÈs de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total*/
SELECT DISTINCT P.NIF, P.NOM, B.CODI_VOL, TO_CHAR(B.DATA, 'DD/MM/YYYY') AS DATA, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, BITLLET B, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
  AND B.NIF_PASSATGER = M.NIF_PASSATGER
  AND B.CODI_VOL = M.CODI_VOL
  AND B.DATA = M.DATA
GROUP BY P.NIF, P.NOM, B.CODI_VOL, B.DATA
HAVING SUM(M.PES) > 15
ORDER BY 1, 2, 3, 4, 5;

/*6 (Espectacles) Capacitat total del teatre Romea. */
SELECT ZR.CAPACITAT
FROM ZONES_RECINTE ZR, RECINTES R
WHERE ZR.CODI_RECINTE = R.CODI
AND R.NOM = 'Romea'
ORDER BY 1;

/*7 (Espectacles) Preu m‡xim i mÌnim de les entrades per líespectacle Jazz a la tardor.*/
SELECT MAX(PREU), MIN(PREU)
FROM ESPECTACLES E, PREUS_ESPECTACLES PE
WHERE PE.CODI_ESPECTACLE = E.CODI
  AND E.NOM = 'Jazz a la tardor'
ORDER BY 1,2;

/*8 (Espectacles) Data, hora i nombre díentrades venudes de cadascuna de les representacions de El M‡gic díOz.*/
SELECT TO_CHAR(E.DATA, 'DD/MM/YYYY') AS DATA, 
TO_CHAR(E.HORA, 'HH24:MI:SS') AS HORA,
COUNT(E.CODI_ESPECTACLE) AS N_ENTRADES
FROM ESPECTACLES ESP, ENTRADES E, REPRESENTACIONS R
WHERE E.CODI_ESPECTACLE = R.CODI_ESPECTACLE
  AND E.DATA = R.DATA
  AND E.HORA = E.HORA
  AND R.CODI_ESPECTACLE = ESP.CODI
  AND ESP.NOM = 'El M‡gic d''Oz'
GROUP BY E.DATA, E.HORA
ORDER BY 1, 2, 3;

/*9 (Espectacles) Nom dels espectacles on el preu de l'entrada mÈs econòmica sigui superior a 18Ä.*/
SELECT E.NOM
FROM ESPECTACLES E, PREUS_ESPECTACLES PE
WHERE PE.CODI_ESPECTACLE = E.CODI
GROUP BY E.NOM
HAVING MIN(PE.PREU) > 18
ORDER BY 1;

/*10 (Espectacles) DNI, nom i cognoms dels espectadors que sëhan gastat mÈs de 500Ä en espectacles.*/
SELECT ES.DNI, ES.NOM, ES.COGNOMS
FROM ESPECTADORS ES, ENTRADES E, REPRESENTACIONS R, ESPECTACLES ESP, PREUS_ESPECTACLES PE
WHERE ES.DNI = E.DNI_CLIENT
  AND E.CODI_ESPECTACLE = R.CODI_ESPECTACLE
  AND E.DATA = R.DATA
  AND E.HORA = E.HORA
  AND R.CODI_ESPECTACLE = ESP.CODI
  AND PE.CODI_ESPECTACLE = ESP.CODI
GROUP BY ES.DNI, ES.NOM, ES.COGNOMS
HAVING SUM(PE.PREU) > 500
ORDER BY 1, 2, 3


Nom i Població dels passatgers que han viatjat algun cop en 1a classe.
SELECT P.NOM, P.POBLACIO
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = 1
GROUP BY P.NOM, P.POBLACIO
ORDER BY 1

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'àrea i el nombre de portes d'aquesta àrea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(DISTINCT PE.PORTA) AS PORTES
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA 
ORDER BY 1,2,3,4

--Nombre màxim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF, B.DATA
ORDER BY 1;

SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF)
ORDER BY 1;







--Del total de maletes facturades, quin percentatge són de color vermell.
--(NOTA: el percentatge és sempre un numero entre 0 i 100)
SELECT T1.M_VERMELLES/T2.M_TOTAL AS PERCENTATGE
FROM(SELECT COUNT(*) AS M_VERMELLES
FROM MALETA M
WHERE M.COLOR = 'vermell') T1, (SELECT COUNT(*) AS M_TOTAL
FROM MALETA M) T2
ORDER BY 1;

--Nombre de vegades que apareix la nacionalitat més freqüent entre les 
--persones de la BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.
SELECT P.PAIS, COUNT(*) AS N_NACIONALITATS
FROM PERSONA P
GROUP BY P.PAIS
HAVING COUNT(*)>= ALL(SELECT COUNT(*)
FROM PERSONA P
GROUP BY P.PAIS)
ORDER BY 1,2;

--Quan pes ha facturat l'Alberto Carlos Huevos al vol amb destinació Rotterdam
de la data 02 de juny del 2013?
SELECT M.PES
FROM MALETA M, PERSONA P, VOL V, AEROPORT A
WHERE P.NOM = 'Alberto Carlos Huevos'
AND P.NIF = M.NIF_PASSATGER
AND M.CODI_VOL = V.CODI_VOL
AND V.DATA = TO_DATE('02/06/2013', 'DD/MM/YYYY')
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Rotterdam'
ORDER BY 1;

Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/2013
SELECT S.FILA, S.LLETRA
FROM SEIENT S
WHERE S.CODI_VOL = 'AEA2195'
AND S.EMBARCAT != 'SI'
AND S.DATA = TO_DATE('31/07/2013','DD/MM/YYYY')
GROUP BY S.FILA, S.LLETRA
ORDER BY 1;

Nom dels passatgers que tenen alguna maleta que pesa més de 10 kilos. 
Especifica també en quin vol.
SELECT P.NOM, M.CODI_VOL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND M.PES > 10
GROUP BY P.NOM, M.CODI_VOL
ORDER BY 1,2;




Nombre de Bitllets per compradors. Es demana nom del comprador i 
nombre de bitllets comprats.
SELECT P.NOM, COUNT(B.NIF_PASSATGER) AS N_BITLLETS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
ORDER BY 1;

Nom(es) de la(es) persona(es) que més cops ha(n) pagat amb Paypal.
Retorneu també el nòmero de cops que ho ha(n) fet.
SELECT P.NOM, COUNT(*) AS COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM RESERVA
WHERE MODE_PAGAMENT = 'Paypal'
GROUP BY NIF_CLIENT)
ORDER BY 1;

--Tots els models diferents d'avió que existeixen
SELECT DISTINCT TIPUS_AVIO
FROM VOL
ORDER BY 1;


--Nom i telèfon de tots les persones portugueses.
SELECT P.NOM, P.TELEFON
FROM PERSONA P
WHERE P.PAIS = 'Portugal'
ORDER BY 1,2;


--Companyies que tenen avions 'Airbus A380' de tots els tipus. Nota: per restringir
--els tipus d'avio a 'Airbus A380' useu l'operador "like":  tipus_avio like 'Airbus A380%'.
SELECT DISTINCT V.COMPANYIA
FROM VOL V
WHERE V.TIPUS_AVIO LIKE 'Airbus A380%'
GROUP BY V.COMPANYIA
ORDER BY 1;

--NIF i nom del(s) passatger(s) que ha(n) facturat menys pes en el total dels seus vols.
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND M.PES <= ALL (SELECT PES
FROM MALETA)
GROUP BY P.NIF, P.NOM
ORDER BY 1,2;


--Nombre de vegades que apareix la nacionalitat més freqüent entre les persones de la BD. 
--Es demana la  nacionalitat i nombre de vegades que apareix.
SELECT P.PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA P
GROUP BY P.PAIS
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_VEGADES
FROM PERSONA P
GROUP BY P.PAIS)
ORDER BY 1,2;

--Total de bitllets venuts per la companyia IBERIA.
SELECT COUNT(*) AS N_BITLLETS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA = 'IBERIA'
ORDER BY 1;

--Nombre de bitllets per compradors. Es demana nom del comprador i nombre de bitllets comprats.
SELECT P.NOM, COUNT(*) AS N_BITLLETS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
ORDER BY 1,2;


--Nom dels clients que sempre paguem amb Visa Electron i dels que sempre paguem amb Paypal.
SELECT P.NOM
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND (R.MODE_PAGAMENT = 'Visa Electron' OR R.MODE_PAGAMENT = 'Paypal')
GROUP BY P.NOM
ORDER BY 1;


--Data i codi de vol dels vols amb més de 3 passatgers. ESTA MALAMENT
SELECT V.DATA, V.CODI_VOL
FROM VOL V
WHERE N_PASSATGERS = (SELECT COUNT(*) AS N
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA)
GROUP BY V.DATA, V.CODI_VOL
HAVING N_PASSATGERS > 3
ORDER BY 1,2;

--El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER = P.NIF
AND P.PAIS = 'Italia'
ORDER BY 1;

--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco té el vol
--amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGERS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = 'IBE2119'
AND P.NOM LIKE '%Blanco%'
AND B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
ORDER BY 1;

--Codi, data i pes total dels vols que han facturat, en total, un pes igual o superior a 26 Kgs.
SELECT M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM MALETA M
GROUP BY M.CODI_VOL, M.DATA
HAVING SUM(M.PES) > 26
ORDER BY 1,2,3;

--Quantes maletes ha facturat en Aitor Tilla al vol amb destinació Pekin de la data 1/10/2013
SELECT COUNT(*) AS N_MALETES
FROM MALETA M, PERSONA P, VOL V, AEROPORT A
WHERE M.NIF_PASSATGER = P.NIF
AND P.NOM = 'Aitor Tilla'
AND M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Pekin'
AND V.DATA = TO_DATE('01/10/2013','DD/MM/YYYY')
ORDER BY 1;

--Nom dels passatgers que han facturat més d'una maleta vermella.
SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1;





--Nòmero de porta amb més d'un vol assignat. Mostrar codi de l'aeroport al que pertany,
--terminal, area i porta.
SELECT P.CODI_AEROPORT, P.TERMINAL, P.AREA, P.PORTA
FROM PORTA_EMBARCAMENT P, VOL V
WHERE V.ORIGEN = P.CODI_AEROPORT
AND V.TERMINAL = P.TERMINAL
AND V.AREA = P.AREA
AND V.PORTA = P.PORTA
GROUP BY P.CODI_AEROPORT, P.TERMINAL, P.AREA, P.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;

--Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1,2;

--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117
--del 27 de Agosto de 2013.
SELECT COUNT(R.LOCALITZADOR) AS N_SEIENTS_RESERVADES
FROM RESERVA R, BITLLET B, VOL V
WHERE R.LOCALITZADOR = B.LOCALITZADOR
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA = 'VUELING AIRLINES'
AND V.CODI_VOL = 'VLG2117'
AND V.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1;


--Tipus d'avió(ns) que té(nen) més passatgers no espanyols. 
SI FUNCIONA:

SELECT DISTINCT V.TIPUS_AVIO
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS != 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_PASS
FROM BITLLET B, VOL V, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS != 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;


SI FUNCIONA:

SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL((SELECT COUNT(*) AS N_PASS
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.TIPUS_AVIO)
MINUS
(SELECT COUNT(*) AS N_ESP
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO))
ORDER BY 1;

--NIF i nom de la(/es) persona(/es) que ha(n) reservat més bitllets.
SELECT P.NIF, P.NOM
FROM PERSONA P, RESERVA R, BITLLET B
WHERE P.NIF = R.NIF_CLIENT
AND B.LOCALITZADOR = R.LOCALITZADOR
GROUP BY P.NIF, P.NOM
HAVING COUNT(*) >= ALL (SELECT COUNT (*) AS N_RESERVES
FROM PERSONA P, RESERVA R, BITLLET B
WHERE P.NIF = R.NIF_CLIENT
AND B.LOCALITZADOR = R.LOCALITZADOR
GROUP BY P.NIF)
ORDER BY 1,2;

--Nom(es) de la(es) persona(es) que més cops ha(n) pagat amb Paypal. PER QUE NO CAL BITLLET B?
--Retorneu també el número de cops que ho ha(n) fet.
SELECT P.NOM, COUNT(*) AS N_COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM)
ORDER BY 1;


--Tipus d'avió(ns) que fa(n) mes vols amb origen Espanya.
SELECT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;

--Tipus d'avió(ns) amb més files. PER QUE DISTINCT?
SELECT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA) >= ALL(SELECT COUNT(DISTINCT S.FILA)
FROM SEIENT S
GROUP BY S.CODI_VOL)
ORDER BY 1;

--Nom(es) de la companyia(es) que fa(n) més vols amb origen Italia.
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(V.CODI_VOL) >= ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1;

--ciutat de França amb més arribades
SELECT A.CIUTAT
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY A.CIUTAT
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY V.CODI_VOL)
ORDER BY 1;


--NIF, nom del(s) passatger(s) i pes total que ha(n) facturat 
--menys pes en el conjunt de tots els seus bitllets.
SELECT P.NIF, P.NOM, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES) <= ALL(SELECT SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY M.NIF_PASSATGER)
ORDER BY 1,2;

--Nom de totes les persones que son del mateix país que "Domingo Diaz Festivo" (ell inclñs).
SELECT P.NOM
FROM PERSONA P
WHERE P.PAIS = (SELECT P.PAIS
FROM PERSONA P
WHERE P.NOM = 'Domingo Diaz Festivo')
GROUP BY P.NOM
ORDER BY 1;

--Color(s) de la(/es) maleta(/es) facturada(/es) més pesada(/es).
SELECT M.COLOR
FROM MALETA M
WHERE M.PES = (SELECT MAX(M.PES)
FROM MALETA M)
ORDER BY 1; 

--Dates de vol (DD/MM/YYYY) del(s) avió(ns) amb mes capacitat (major nombre de seients)

NO FUNCIONA:

SELECT V.CODI_VOL
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL
HAVING COUNT(S.FILA) >= ALL(SELECT MAX(COUNT(S.FILA)) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1;

NO FUNCIONA:

SELECT V.CODI_VOL, TO_CHAR(V.DATA, 'DD/MM/YYYY') AS DATA
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL, TO_CHAR(V.DATA, 'DD/MM/YYYY')
HAVING COUNT(S.FILA) >= ALL(SELECT MAX(COUNT(S.FILA)) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1;


--Nombre de seients reservats al vol ROYAL AIR MAROC RAM964 02/06/2014
SELECT COUNT(*) AS N_RESERVES
FROM SEIENT S, BITLLET B
WHERE S.CODI_VOL = 'RAM964'
AND S.DATA = TO_DATE('02/07/2014','DD/MM/YYYY')
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
ORDER BY 1;

--Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT COUNT(*) AS N_LLIURES
FROM SEIENT S
WHERE S.CODI_VOL = 'AEA2195'
AND S.DATA = TO_DATE('31/07/2013', 'DD/MM/YYYY')
AND (S.FILA, S.LLETRA) NOT IN (SELECT S.FILA, S.LLETRA
FROM SEIENT S, BITLLET B
WHERE S.CODI_VOL = 'AEA2195'
AND S.DATA = TO_DATE('31/07/2013', 'DD/MM/YYYY')
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
AND B.FILA = S.FILA
AND B.LLETRA = S.LLETRA)
ORDER BY 1;

--Aeroports que no tenen vol en los tres primeros meses de 2014
SELECT A.CODI_AEROPORT
FROM AEROPORT A, VOL V
WHERE V.ORIGEN = A.CODI_AEROPORT
AND V.CODI_VOL NOT IN (SELECT V.CODI_VOL
FROM VOL V
WHERE V.DATA BETWEEN TO_DATE('01/01/2014','DD/MM/YYYY') AND TO_DATE('31/3/2014','DD/MM/YYYY'))
GROUP BY A.CODI_AEROPORT
ORDER BY 1;

--Noms dels aeroports i nom de les ciutats de destinació d'aquells 
--vols a les que els falta un passatger per estar plens
SELECT A.NOM, A.CIUTAT
FROM AEROPORT A, VOL V, (SELECT COUNT(*) AS TOTAL_1, B.CODI_VOL, B.DATA
FROM BITLLET B
GROUP BY B.CODI_VOL, B.DATA) T1, --SUBCONSULTA Q ENS RETORNA EL NUMERO DE BITLLETS AMB CODI I DATA
(SELECT COUNT(*) AS TOTAL_2, S.CODI_VOL, S.DATA
FROM SEIENT S
GROUP BY S.CODI_VOL, S.DATA) T2 --SUBCONSULTA Q ENS RETORNA EL NUMERO DE SEIENTS AMB CODI I DATA
WHERE T1.DATA = T2.DATA --JOINT DELS BITLLETS AMB ELS SEIENTS
AND T1.CODI_VOL = T2.CODI_VOL --JOINT DELS BITLLETS AMB ELS SEIENTS
AND T1.DATA = V.DATA --JOINT DEL BITLLET AMB EL VOL
AND T1.CODI_VOL = V.CODI_VOL --JOINT DEL BITLLET AMB EL VOL
AND V.DESTINACIO = A.CODI_AEROPORT --JOINT DEL VOL AMB EL AEROPORT
AND (T1.TOTAL_1 - T2.TOTAL_2) = 1 --CONDICIO NUM BITLLETS - NUM SEIENTS = 1 (1 LLOC LLIURE)
GROUP BY A.NOM, A.CIUTAT
ORDER BY 1, 2;

--Codi de vol i data de tots els bitllets d'anada.
SELECT CODI_VOL, TO_CHAR(DATA, 'DD/MM/YYYY') AS DATA
FROM BITLLET
WHERE ANADA_TORNADA = 'ANADA'
ORDER BY 1,2;

--Nombre de seients reservats al vol de ROYAL AIR MAROC codi RAM964 del 2 de Juny de 2014.
SELECT COUNT(*) AS N_BITLLETS
FROM RESERVA R, SEIENT S, BITLLET B
WHERE R.LOCALITZADOR = B.LOCALITZADOR
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
AND B.CODI_VOL = 'RAM964'
AND B.DATA = TO_DATE('02/06/2014','DD/MM/YYYY')
ORDER BY 1;

--Nom, email (mail) i adreça dels passatgers que han volat amb les mateixes companyies que 
--Andres Trozado (poden haver volat amb més companyies).
SELECT P.NOM, P.MAIL, P.ADREÇA, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA IN (SELECT V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.NOM = 'Andres Trozado'
GROUP BY V.COMPANYIA)
GROUP BY P.NOM, P.MAIL, P.ADREÇA, V.COMPANYIA
ORDER BY 1,2,3;

--Nom i NIF de les persones que volen a França i que no han facturat mai maletes de color vermell.
--NOTA: Ha d'incloure també les persones que volen a França però que no han facturat mai.
SELECT P.NOM, P.NIF
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
AND P.NIF NOT IN (SELECT DISTINCT P.NIF
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell')
GROUP BY P.NOM, P.NIF;

SELECT DISTINCT P.NIF
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell'
ORDER BY 1;

SELECT P.NOM, P.NIF
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY P.NOM, P.NIF;

--Per a cada vol, calculi el nombre de maletes facturades. Retorneu el codi de vol, la data, 
-- i el nombre de maletes. (NOTA: les dates s'han de presentar al format 'DD/MM/YYYY')
SELECT V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY') AS DATA, COUNT(*) AS N_MALETES
FROM VOL V, MALETA M
WHERE M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA 
GROUP BY V.CODI_VOL, V.DATA
ORDER BY 1,2,3;

--Preu promig de les reserves pagats amb Visa. El numerador es 
--el preu total de les reserves pagades amb Visa i el denominador és el numero de
--reserves pagats amb Visa.
SELECT T1.PREU_TOTAL/T2.N_RESERVES AS PROMIG_RESERVES
FROM (SELECT SUM(PREU) AS PREU_TOTAL
FROM RESERVA
WHERE MODE_PAGAMENT = 'Visa') T1,
(SELECT COUNT(*) AS N_RESERVES
FROM RESERVA
WHERE MODE_PAGAMENT = 'Visa') T2
ORDER BY 1;






--3. Pais dels passatgers que han viatjat en primera classe més de 5 vegades.
SELECT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND P.NIF IN (SELECT P.NIF
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = '1'
GROUP BY P.NIF
HAVING COUNT(*) > 5)
GROUP BY P.PAIS
ORDER BY 1;

--passatgers que han viatjat en primeraclasse mes de 5 cops:
SELECT P.NIF
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = '1'
GROUP BY P.NIF
HAVING COUNT(*) > 5;

--4. Codi(s) de vol del(s) avion(s) més petit(s) (menor nombre de seients)
SELECT DISTINCT V.CODI_VOL
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA
HAVING COUNT(*) <= ALL(SELECT COUNT(*) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA);

--nombre seients de cada avio:
SELECT COUNT(*) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA
ORDER BY 1;

--8. Nombre de maletes de cada passatger. Retorna el nom del passatger, el seu NIF i el número
--de maletes que posseeix
SELECT P.NOM, P.NIF, COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM
ORDER BY 1,2,3;



--nombre maletes de cada passatger amb maleta
SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF;

--5. Noms de les persones que han facturat tots els mateixos colors de maleta que 
--Jose Luis Lamata Feliz (també poden haver facturat maletes d'altres colors)
SELECT DISTINCT R1.NOM
FROM (SELECT DISTINCT P.NOM, M.COLOR
        FROM PERSONA P, MALETA M
        WHERE P.NIF=M.NIF_PASSATGER) R1
WHERE (R1.NOM) NOT IN (SELECT R2.NOM
                        FROM (SELECT DISTINCT M.COLOR
                                FROM PERSONA P, MALETA M
                                WHERE P.NIF = M.NIF_PASSATGER 
                                AND P.NOM = 'Jose Luis Lamata Feliz') S,
                            (SELECT DISTINCT P.NOM, M.COLOR
                                FROM PERSONA P, MALETA M
                                WHERE P.NIF = M.NIF_PASSATGER) R2
                        WHERE (S.COLOR, R2.NOM) NOT IN (SELECT R3.COLOR, R3.NOM
                                                        FROM (SELECT DISTINCT P.NOM, M.COLOR
                                                                FROM PERSONA P, MALETA M
                                                                WHERE P.NIF = M.NIF_PASSATGER) R3))
ORDER BY 1;

--colors de les maletes que ha facturat Jose Luis Lamata Feliz:
SELECT M.COLOR
FROM MALETA M, PERSONA P
WHERE P.NIF = M.NIF_PASSATGER
AND P.NOM = 'Jose Luis Lamata Feliz'
GROUP BY M.COLOR
ORDER BY 1;

--2. Nombre de vols promig de totes les companyies. 
--NOTA: el numerador és el nombre de vols totals i el denominador el nombre total de companyies.
SELECT T1.VOLS_TOTALS/T2.N_COMPANYIES AS PROMIG
FROM (SELECT COUNT(*) AS VOLS_TOTALS
FROM VOL V) T1, 
(SELECT COUNT(DISTINCT V.COMPANYIA) AS N_COMPANYIES
FROM VOL V) T2
ORDER BY 1;

SELECT DISTINCT TV.TOTAL_VOLS/TC.TOTAL_COMPANYIES
FROM (SELECT SUM(COUNT(*)) AS TOTAL_VOLS
FROM VOL V
GROUP BY V.COMPANYIA) TV, (SELECT SUM(COUNT(DISTINCT V.COMPANYIA)) AS TOTAL_COMPANYIES
FROM VOL V
GROUP BY V.COMPANYIA) TC
ORDER BY 1;
--7. Percentage del pes total de de maletes que correspon als passatgers francesos. 
--NOTA: un percentatge s'expressa amb nombres entre 0 i 100.
SELECT (T1.PES_FRANCESOS*100)/T2.PES_TOTAL AS PERCENTATGE
FROM (SELECT SUM(M.PES) AS PES_FRANCESOS
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND P.PAIS = 'França') T1,
(SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M) T2
ORDER BY 1;

--6. Per a tots els vols de l'any 2014 mostra el NIF dels que volen i dels que han comprat 
--cada bitllet.

SELECT NIF_PASSATGER, NIF_CLIENT
FROM BITLLET B, RESERVA R
WHERE B.LOCALITZADOR = R.LOCALITZADOR
AND B.DATA BETWEEN TO_DATE('01/01/2014','DD/MM/YYYY') AND TO_DATE('31/12/2014','DD/MM/YYYY')
GROUP BY NIF_CLIENT, NIF_PASSATGER
ORDER BY 1,2;
 
Quantes maletes ha facturat en Aitor Tilla al vol amb destinació Pekin de la data 1/10/2013? 
SELECT COUNT(M.CODI_MALETA) 
FROM MALETA M, VOL V, PERSONA P, AEROPORT A 
WHERE M.NIF_PASSATGER = P.NIF 
AND M.CODI_VOL = V.CODI_VOL 
AND M.DATA = V.DATA 
AND V.DESTINACIO = A.CODI_AEROPORT 
AND P.NOM = 'Aitor Tilla' 
AND A.CIUTAT = 'Pekin' 
AND V.DATA = TO_DATE('01/10/2013', 'DD/MM/YYYY') 
ORDER BY 1;

Codi_Vol i data dels vols amb origen a algún aeroport rus 
SELECT V.CODI_VOL, TO_DATE(V.DATA, 'DD/MM/YYYY') AS DATA 
FROM VOL V 
WHERE ORIGEN IN (SELECT CODI_AEROPORT 
FROM AEROPORT A 
WHERE PAIS = 'Russia') 
ORDER BY 1;

Portes d'embarcament (terminal, area, porta) de l'aeroport JFK on no han sortit mai cap vol amb destinació França
(SELECT PE.TERMINAL, PE.AREA, PE.PORTA
FROM PORTA_EMBARCAMENT PE
WHERE PE.CODI_AEROPORT = 'JFK')
MINUS
(SELECT PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, AEROPORT A, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
AND V.TERMINAL = PE.TERMINAL
AND V.AREA = PE.AREA
AND V.PORTA = PE.PORTA
AND PE.CODI_AEROPORT ='JFK'
AND A.PAIS = 'França'
AND V.DESTINACIO = A.CODI_AEROPORT)
ORDER BY 1,2,3;


Per les reserves de vols amb destinació el Marroc, aquelles on només viatja un únic passatger. Mostreu localitzador i nom del passatger que hi viatja.
SELECT B. LOCALITZADOR, P.NOM 
FROM PERSONA P, BITLLET B 
WHERE P.NIF = B.NIF_PASSATGER 
AND B.CODI_VOL IN (SELECT CODI_VOL 
FROM BITLLET B 
WHERE CODI_VOL IN (SELECT V.CODI_VOL 
FROM AEROPORT A, VOL V 
WHERE V.DESTINACIO = A.CODI_AEROPORT 
AND A.PAIS = 'Marroc') 
GROUP BY CODI_VOL 
HAVING COUNT(*) = 1) 
ORDER BY 1;

--Companyies que tenen, almenys, els mateixos tipus d'avions que Vueling airlines (El resultat també ha de tornar Vueling Airlines)
SELECT DISTINCT COMPANYIA
FROM VOL
WHERE TIPUS_AVIO IN (SELECT DISTINCT TIPUS_AVIO
FROM VOL
WHERE COMPANYIA = 'VUELING AIRLINES')
ORDER BY 1;

--Nom de la companyia que fa més vols amb origen Italia 
SELECT V.COMPANYIA 
FROM VOL V, AEROPORT A 
WHERE V.ORIGEN = A.CODI_AEROPORT 
AND A.PAIS = 'Italia' 
GROUP BY V.COMPANYIA 
HAVING COUNT(*) = (SELECT MAX(COUNT(*)) 
FROM VOL V, AEROPORT A 
WHERE V.ORIGEN = A.CODI_AEROPORT 
AND A.PAIS = 'Italia' 
GROUP BY V.COMPANYIA) 
ORDER BY 1;

--Per a cada aeroport i terminal, nom de l'aeroport, terminal i nombre d'arees. 
SELECT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA) AS AREAS 
FROM PORTA_EMBARCAMENT PE, AEROPORT A 
WHERE PE.CODI_AEROPORT = A.CODI_AEROPORT 
GROUP BY A.NOM, PE.TERMINAL 
ORDER BY 1,2;

--Promitg de vols per a cada companyia. Aclariment: el numerador es el nombre de vols totals i el denominador el nombre total de companyies
SELECT N.VOLS/D.COMPANYIES AS PROMITG
FROM (SELECT COUNT(*) AS VOLS FROM VOL) N,
(SELECT COUNT(DISTINCT COMPANYIA) AS COMPANYIES FROM VOL) D
ORDER BY 1;

--Promig d'espanyols als vols amb destinació Munich. 
SELECT E.ESPANYOLS/T.TOTAL 
FROM (SELECT COUNT(*) AS ESPANYOLS 
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A 
WHERE P.NIF = B.NIF_PASSATGER 
AND B.CODI_VOL = V.CODI_VOL 
AND B.DATA = V.DATA 
AND V.DESTINACIO = A.CODI_AEROPORT 
AND A.CIUTAT = 'Munich' 
AND P.PAIS = 'Espanya') E, (SELECT COUNT(*) AS TOTAL 
FROM BITLLET B, VOL V, AEROPORT A 
WHERE B.CODI_VOL = V.CODI_VOL 
AND B.DATA = V.DATA 
AND V.DESTINACIO = A.CODI_AEROPORT 
AND A.CIUTAT = 'Munich') T 
ORDER BY 1;

--Nom, correu i companyia aeria dels clients (persones) que han fet reserves de 2 bitllets d’anada i 2 bitlles de tornada
(SELECT P.NOM, P.MAIL, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.ANADA_TORNADA = 'ANADA'
GROUP BY P.NOM, P.MAIL, V.COMPANYIA
HAVING COUNT(B.ANADA_TORNADA) = 2)
INTERSECT
(SELECT P.NOM, P.MAIL, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.ANADA_TORNADA = 'TORNADA'
GROUP BY P.NOM, P.MAIL, V.COMPANYIA
HAVING COUNT(B.ANADA_TORNADA) = 2)
ORDER BY 1,2,3;



--Pes facturat als vols de 'AIR FRANCE' del 2013. 
SELECT SUM(PES) AS PES 
FROM MALETA M, VOL V 
WHERE M.CODI_VOL = V.CODI_VOL 
AND M.DATA = V.DATA 
AND V.COMPANYIA = 'AIR FRANCE' 
AND TO_CHAR(V.DATA, 'YYYY') = '2013' 
ORDER BY 1;

--Nacionalitat dels passatgers del vols amb destinació Paris Orly Airport.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.NOM = 'Paris Orly Airport'
ORDER BY 1;

--En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a ús dels passatgers? 
SELECT FILA 
FROM SEIENT 
WHERE CODI_VOL = 'AEA2159' 
AND TO_CHAR(DATA, 'DD/MM/YYYY') = '01/08/2013' 
AND LLETRA = 'A' 
ORDER BY 1;

--Número de terminals dels aeroports francesos (retorneu el nom de l’aeroport i la quantitat de terminals). 
SELECT A.NOM, COUNT(DISTINCT PE.TERMINAL) AS TERMINALS 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT 
AND A.PAIS = 'França' 
GROUP BY A.NOM 
ORDER BY 1;


--Nom de la persona que més cops ha pagat amb Paypal. Retorneu també el número de cops que ho ha fet.
SELECT P.NOM, COUNT(*) AS COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
and R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) > ANY (SELECT COUNT(*)
FROM RESERVA
WHERE MODE_PAGAMENT = 'Paypal'
GROUP BY NIF_CLIENT)
ORDER BY 1;

Destinació dels vols que tenen entre els seus passatgers persones de totes les ciutats de Turquia. 
SELECT DISTINCT V.DESTINACIO 
FROM BITLLET B, VOL V 
WHERE B.CODI_VOL = V.CODI_VOL 
AND B.DATA = V.DATA 
AND B.NIF_PASSATGER IN (SELECT NIF 
FROM PERSONA 
WHERE PAIS = 'Turquia') 
ORDER BY 1;

--Pais, codi_aeroport, terminal i àrea dels aeroport amb 3 portes d'embarcament i el nom del pais comenci per 'E' 
SELECT DISTINCT A.PAIS, PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA 
FROM AEROPORT A, PORTA_EMBARCAMENT PE 
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT 
AND A.PAIS LIKE 'E%' 
AND PE.CODI_AEROPORT IN (SELECT PE.CODI_AEROPORT 
FROM PORTA_EMBARCAMENT PE 
GROUP BY PE.CODI_AEROPORT 
HAVING COUNT(DISTINCT PORTA) = 3) 
ORDER BY 1, 2 ,3, 4;

--Mostreu el numero de passatgers que no han escollit els seus seients 
SELECT COUNT(*) 
FROM BITLLET 
WHERE FILA IS NULL 
AND LLETRA IS NULL 
ORDER BY 1;


--Nombre de passatgers agrupats segons pais de destinació (en el vol). Mostreu pais i nombre de passatgers 
SELECT A.PAIS, COUNT(B.NIF_PASSATGER) AS NOMBRE_PASSATGERS 
FROM VOL V, AEROPORT A, BITLLET B 
WHERE V.DESTINACIO = A.CODI_AEROPORT 
AND B.CODI_VOL = V.CODI_VOL 
AND B.DATA = V.DATA 
GROUP BY A.PAIS 
ORDER BY 1;

--Nom dels clients que sempre paguem amb Visa Electron 
SELECT P.NOM 
FROM PERSONA P, RESERVA R 
WHERE P.NIF = R.NIF_CLIENT 
AND R.MODE_PAGAMENT = 'Visa Electron' 
GROUP BY P.NOM 
HAVING (P.NOM, COUNT(*)) IN (SELECT P.NOM, COUNT(*) 
FROM PERSONA P, RESERVA R 
WHERE P.NIF = R.NIF_CLIENT 
GROUP BY P.NOM) 
ORDER BY 1;

--Ciutats espanyoles amb aeroports 
SELECT CIUTAT 
FROM AEROPORT 
WHERE PAIS = 'Espanya';

--Noms de les companyies que volen als mateixos aeroports on ha volat Estela Gartija. 
SELECT DISTINCT COMPANYIA 
FROM VOL V 
WHERE DESTINACIO IN (SELECT V.DESTINACIO 
FROM PERSONA P, BITLLET B, VOL V 
WHERE P.NIF = B.NIF_PASSATGER 
AND B.CODI_VOL = V.CODI_VOL 
AND B.DATA = V.DATA 
AND P.NOM = 'Estela Gartija') 
ORDER BY 1;

--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers dels vols que tenen el nombre de passatges més petit.
SELECT V.ORIGEN, V.CODI_VOL, V.DATA, COUNT(B.NIF_PASSATGER)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(B.NIF_PASSATGER) = (SELECT MIN(COUNT (*))
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1,2,3,4;

--Percentatge d'ús de l'aeroport de Barcelona. Indicació: el numerador és el nombre de vols ambos origen, o destinació, Barcelona; el denominador el nombre total de vols. 
SELECT T1.BARCELONA/T2.TOTAL 
FROM (SELECT COUNT(*) AS BARCELONA 
FROM VOL 
WHERE ORIGEN = (SELECT CODI_AEROPORT 
FROM AEROPORT 
WHERE CIUTAT = 'Barcelona') 
OR DESTINACIO = (SELECT CODI_AEROPORT 
FROM AEROPORT 
WHERE CIUTAT = 'Barcelona')) T1, (SELECT COUNT(*) AS TOTAL 
FROM VOL) T2 
ORDER BY 1;

Nom i PoblaciÛ dels passatgers que han viatjat algun cop en 1a classe.

SELECT P.NOM, P.POBLACIO
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = 1
GROUP BY P.NOM, P.POBLACIO
ORDER BY 1


--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'‡rea i el nombre de portes d'aquesta ‡rea.

SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(DISTINCT PE.PORTA) AS PORTES
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA 
ORDER BY 1,2,3,4

--Nombre m‡xim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger. ???? MALAMENT

SELECT MAX(COUNT(*)) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF
ORDER BY 1;

SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF)
ORDER BY 1;


Del total de maletes facturades, quin percentatge sÛn de color vermell.
(NOTA: el percentatge Ès sempre un numero entre 0 i 100)

SELECT T1.M_VERMELLES/T2.M_TOTAL AS PERCENTATGE
FROM(SELECT COUNT(*) AS M_VERMELLES
FROM MALETA M
WHERE M.COLOR = 'vermell') T1, (SELECT COUNT(*) AS M_TOTAL
FROM MALETA M) T2
ORDER BY 1;


--Nombre de vegades que apareix la nacionalitat mÈs freq¸ent entre les persones de la BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.

SELECT P.PAIS, COUNT(*) AS N_NACIONALITATS
FROM PERSONA P
GROUP BY P.PAIS
HAVING COUNT(*)>= ALL(SELECT COUNT(*)
FROM PERSONA P
GROUP BY P.PAIS)
ORDER BY 1,2;


--Quan pes ha facturat l'Alberto Carlos Huevos al vol amb destinaciÛ Rotterdam de la data 02 de juny del 2013?

SELECT M.PES
FROM MALETA M, PERSONA P, VOL V, AEROPORT A
WHERE P.NOM = 'Alberto Carlos Huevos'
AND P.NIF = M.NIF_PASSATGER
AND M.CODI_VOL = V.CODI_VOL
AND V.DATA = TO_DATE('02/06/2013', 'DD/MM/YYYY')
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Rotterdam'
ORDER BY 1;


--Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/2013

SELECT S.FILA, S.LLETRA
FROM SEIENT S
WHERE S.CODI_VOL = 'AEA2195'
AND S.EMBARCAT != 'SI'
AND S.DATA = TO_DATE('31/07/2013','DD/MM/YYYY')
GROUP BY S.FILA, S.LLETRA
ORDER BY 1;


--Nom dels passatgers que tenen alguna maleta que pesa mÈs de 10 kilos. 
--Especifica tambÈ en quin vol.

SELECT P.NOM, M.CODI_VOL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND M.PES > 10
GROUP BY P.NOM, M.CODI_VOL
ORDER BY 1,2;


---Nombre de Bitllets per compradors. Es demana nom del comprador i nombre de bitllets comprats.

SELECT P.NOM, COUNT(B.NIF_PASSATGER) AS N_BITLLETS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
ORDER BY 1;


--Nom(es) de la(es) persona(es) que mÈs cops ha(n) pagat amb Paypal.
--Retorneu tambÈ el n˙mero de cops que ho ha(n) fet.

SELECT P.NOM, COUNT(*) AS COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM RESERVA
WHERE MODE_PAGAMENT = 'Paypal'
GROUP BY NIF_CLIENT)
ORDER BY 1;


--Tots els models diferents d'aviÛ que existeixen

SELECT DISTINCT TIPUS_AVIO
FROM VOL
ORDER BY 1;


--Nom i telËfon de tots les persones portugueses.

SELECT P.NOM, P.TELEFON
FROM PERSONA P
WHERE P.PAIS = 'Portugal'
ORDER BY 1,2;


--Companyies que tenen avions 'Airbus A380' de tots els tipus. Nota: per restringir
--els tipus d'avio a 'Airbus A380' useu l'operador "like":  tipus_avio like 'Airbus A380%'.

SELECT DISTINCT V.COMPANYIA
FROM VOL V
WHERE V.TIPUS_AVIO LIKE 'Airbus A380%'
GROUP BY V.COMPANYIA
ORDER BY 1;


--NIF i nom del(s) passatger(s) que ha(n) facturat menys pes en el total dels seus vols.

SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND M.PES <= ALL (SELECT PES
FROM MALETA)
GROUP BY P.NIF, P.NOM
ORDER BY 1,2;


--Nombre de vegades que apareix la nacionalitat mÈs freq¸ent entre les persones de la BD. 
--Es demana la  nacionalitat i nombre de vegades que apareix.

SELECT P.PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA P
GROUP BY P.PAIS
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_VEGADES
FROM PERSONA P
GROUP BY P.PAIS)
ORDER BY 1,2;


--Total de bitllets venuts per la companyia IBERIA.
SELECT COUNT(*) AS N_BITLLETS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA = 'IBERIA'
ORDER BY 1;


--Nombre de bitllets per compradors. Es demana nom del comprador i nombre de bitllets comprats.

SELECT P.NOM, COUNT(*) AS N_BITLLETS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
ORDER BY 1,2;


--Nom dels clients que sempre paguem amb Visa Electron i dels que sempre paguem amb Paypal.

SELECT P.NOM
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND (R.MODE_PAGAMENT = 'Visa Electron' OR R.MODE_PAGAMENT = 'Paypal')
GROUP BY P.NOM
ORDER BY 1;



--Data i codi de vol dels vols amb mÈs de 3 passatgers. ESTA MALAMENT

SELECT V.DATA, V.CODI_VOL
FROM VOL V
WHERE N_PASSATGERS = (SELECT COUNT(*) AS N
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA)
GROUP BY V.DATA, V.CODI_VOL
HAVING N_PASSATGERS > 3
ORDER BY 1,2;


--El pes de les maletes (com a pes_total) dels passatgers italians.

SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER = P.NIF
AND P.PAIS = 'Italia'
ORDER BY 1;


--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco tÈ el vol
--amb codi IBE2119 i data 30/10/13.

SELECT COUNT(*) AS N_PASSATGERS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = 'IBE2119'
AND P.NOM LIKE '%Blanco%'
AND B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
ORDER BY 1;


--Codi, data i pes total dels vols que han facturat, en total, un pes igual o superior a 26 Kgs.

SELECT M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM MALETA M
GROUP BY M.CODI_VOL, M.DATA
HAVING SUM(M.PES) > 26
ORDER BY 1,2,3;


--Quantes maletes ha facturat en Aitor Tilla al vol amb destinaciÛ Pekin de la data 1/10/2013

SELECT COUNT(*) AS N_MALETES
FROM MALETA M, PERSONA P, VOL V, AEROPORT A
WHERE M.NIF_PASSATGER = P.NIF
AND P.NOM = 'Aitor Tilla'
AND M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Pekin'
AND V.DATA = TO_DATE('01/10/2013','DD/MM/YYYY')
ORDER BY 1;


--Nom dels passatgers que han facturat mÈs d'una maleta vermella.

SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1;


--N˙mero de porta amb mÈs d'un vol assignat. Mostrar codi de l'aeroport al que pertany,
--terminal, area i porta.

SELECT P.CODI_AEROPORT, P.TERMINAL, P.AREA, P.PORTA
FROM PORTA_EMBARCAMENT P, VOL V
WHERE V.ORIGEN = P.CODI_AEROPORT
AND V.TERMINAL = P.TERMINAL
AND V.AREA = P.AREA
AND V.PORTA = P.PORTA
GROUP BY P.CODI_AEROPORT, P.TERMINAL, P.AREA, P.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;


--Nom de la companyia i nombre de vols que ha fet as N_VOLS.

SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1,2;


--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117
--del 27 de Agosto de 2013.

SELECT COUNT(R.LOCALITZADOR) AS N_SEIENTS_RESERVADES
FROM RESERVA R, BITLLET B, VOL V
WHERE R.LOCALITZADOR = B.LOCALITZADOR
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA = 'VUELING AIRLINES'
AND V.CODI_VOL = 'VLG2117'
AND V.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1;



--Tipus d'aviÛ(ns) que tÈ(nen) mÈs passatgers no espanyols. 

SI FUNCIONA:

SELECT DISTINCT V.TIPUS_AVIO
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS != 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_PASS
FROM BITLLET B, VOL V, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS != 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;

SI FUNCIONA:

SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL((SELECT COUNT(*) AS N_PASS
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.TIPUS_AVIO)
MINUS
(SELECT COUNT(*) AS N_ESP
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO))
ORDER BY 1;


--NIF i nom de la(/es) persona(/es) que ha(n) reservat mÈs bitllets.
SELECT P.NIF, P.NOM
FROM PERSONA P, RESERVA R, BITLLET B
WHERE P.NIF = R.NIF_CLIENT
AND B.LOCALITZADOR = R.LOCALITZADOR
GROUP BY P.NIF, P.NOM
HAVING COUNT(*) >= ALL (SELECT COUNT (*) AS N_RESERVES
FROM PERSONA P, RESERVA R, BITLLET B
WHERE P.NIF = R.NIF_CLIENT
AND B.LOCALITZADOR = R.LOCALITZADOR
GROUP BY P.NIF)
ORDER BY 1,2;


--Nom(es) de la(es) persona(es) que mÈs cops ha(n) pagat amb Paypal. PER QUE NO CAL BITLLET B?
--Retorneu tambÈ el n˙mero de cops que ho ha(n) fet.
SELECT P.NOM, COUNT(*) AS N_COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM)
ORDER BY 1;


--Tipus d'aviÛ(ns) que fa(n) mes vols amb origen Espanya.
SELECT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;


--Tipus d'aviÛ(ns) amb mÈs files. PER QUE DISTINCT?
SELECT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA) >= ALL(SELECT COUNT(DISTINCT S.FILA)
FROM SEIENT S
GROUP BY S.CODI_VOL)
ORDER BY 1;



--Nom(es) de la companyia(es) que fa(n) mÈs vols amb origen Italia.
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(V.CODI_VOL) >= ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1;


--ciutat de FranÁa amb mÈs arribades

SELECT A.CIUTAT
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'FranÁa'
GROUP BY A.CIUTAT
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'FranÁa'
GROUP BY V.CODI_VOL)
ORDER BY 1;


--NIF, nom del(s) passatger(s) i pes total que ha(n) facturat 
--menys pes en el conjunt de tots els seus bitllets.
SELECT P.NIF, P.NOM, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES) <= ALL(SELECT SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY M.NIF_PASSATGER)
ORDER BY 1,2;

--Nom de totes les persones que son del mateix paÌs que "Domingo Diaz Festivo" (ell inclÚs).
SELECT P.NOM
FROM PERSONA P
WHERE P.PAIS = (SELECT P.PAIS
FROM PERSONA P
WHERE P.NOM = 'Domingo Diaz Festivo')
GROUP BY P.NOM
ORDER BY 1;


--Color(s) de la(/es) maleta(/es) facturada(/es) mÈs pesada(/es).
SELECT M.COLOR
FROM MALETA M
WHERE M.PES = (SELECT MAX(M.PES)
FROM MALETA M)
ORDER BY 1; 


--Dates de vol (DD/MM/YYYY) del(s) aviÛ(ns) amb mes capacitat (major nombre de seients)

NO FUNCIONA:

SELECT V.CODI_VOL
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL
HAVING COUNT(S.FILA) >= ALL(SELECT MAX(COUNT(S.FILA)) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1;

NO FUNCIONA:

SELECT V.CODI_VOL, TO_CHAR(V.DATA, 'DD/MM/YYYY') AS DATA
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL, TO_CHAR(V.DATA, 'DD/MM/YYYY')
HAVING COUNT(S.FILA) >= ALL(SELECT MAX(COUNT(S.FILA)) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1;


Nombre de seients reservats al vol ROYAL AIR MAROC RAM964 02/06/2014
SELECT COUNT(*) AS N_RESERVES
FROM SEIENT S, BITLLET B
WHERE S.CODI_VOL = 'RAM964'
AND S.DATA = TO_DATE('02/07/2014','DD/MM/YYYY')
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
ORDER BY 1;


--Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT COUNT(*) AS N_LLIURES
FROM SEIENT S
WHERE S.CODI_VOL = 'AEA2195'
AND S.DATA = TO_DATE('31/07/2013', 'DD/MM/YYYY')
AND (S.FILA, S.LLETRA) NOT IN (SELECT S.FILA, S.LLETRA
FROM SEIENT S, BITLLET B
WHERE S.CODI_VOL = 'AEA2195'
AND S.DATA = TO_DATE('31/07/2013', 'DD/MM/YYYY')
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
AND B.FILA = S.FILA
AND B.LLETRA = S.LLETRA)
ORDER BY 1;

--Aeroports que no tenen vol en los tres primeros meses de 2014
SELECT A.CODI_AEROPORT
FROM AEROPORT A, VOL V
WHERE V.ORIGEN = A.CODI_AEROPORT
AND V.CODI_VOL NOT IN (SELECT V.CODI_VOL
FROM VOL V
WHERE V.DATA BETWEEN TO_DATE('01/01/2014','DD/MM/YYYY') AND TO_DATE('31/3/2014','DD/MM/YYYY'))
GROUP BY A.CODI_AEROPORT
ORDER BY 1;


--Noms dels aeroports i nom de les ciutats de destinaciÛ d'aquells 
--vols a les que els falta un passatger per estar plens
SELECT A.NOM, A.CIUTAT
FROM AEROPORT A, VOL V, (SELECT COUNT(*) AS TOTAL_1, B.CODI_VOL, B.DATA
FROM BITLLET B
GROUP BY B.CODI_VOL, B.DATA) T1, --SUBCONSULTA Q ENS RETORNA EL NUMERO DE BITLLETS AMB CODI I DATA
(SELECT COUNT(*) AS TOTAL_2, S.CODI_VOL, S.DATA
FROM SEIENT S
GROUP BY S.CODI_VOL, S.DATA) T2 --SUBCONSULTA Q ENS RETORNA EL NUMERO DE SEIENTS AMB CODI I DATA
WHERE T1.DATA = T2.DATA --JOINT DELS BITLLETS AMB ELS SEIENTS
AND T1.CODI_VOL = T2.CODI_VOL --JOINT DELS BITLLETS AMB ELS SEIENTS
AND T1.DATA = V.DATA --JOINT DEL BITLLET AMB EL VOL
AND T1.CODI_VOL = V.CODI_VOL --JOINT DEL BITLLET AMB EL VOL
AND V.DESTINACIO = A.CODI_AEROPORT --JOINT DEL VOL AMB EL AEROPORT
AND (T1.TOTAL_1 - T2.TOTAL_2) = 1 --CONDICIO NUM BITLLETS - NUM SEIENTS = 1 (1 LLOC LLIURE)
GROUP BY A.NOM, A.CIUTAT
ORDER BY 1, 2;


--Codi de vol i data de tots els bitllets d'anada.
SELECT CODI_VOL, TO_CHAR(DATA, 'DD/MM/YYYY') AS DATA
FROM BITLLET
WHERE ANADA_TORNADA = 'ANADA'
ORDER BY 1,2;

--Nombre de seients reservats al vol de ROYAL AIR MAROC codi RAM964 del 2 de Juny de 2014.
SELECT COUNT(*) AS N_BITLLETS
FROM RESERVA R, SEIENT S, BITLLET B
WHERE R.LOCALITZADOR = B.LOCALITZADOR
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
AND B.CODI_VOL = 'RAM964'
AND B.DATA = TO_DATE('02/06/2014','DD/MM/YYYY')
ORDER BY 1;

--Nom, email (mail) i adreÁa dels passatgers que han volat amb les mateixes companyies que 
--Andres Trozado (poden haver volat amb mÈs companyies).
SELECT P.NOM, P.MAIL, P.ADRE«A, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA IN (SELECT V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.NOM = 'Andres Trozado'
GROUP BY V.COMPANYIA)
GROUP BY P.NOM, P.MAIL, P.ADRE«A, V.COMPANYIA
ORDER BY 1,2,3;


--Nom i NIF de les persones que volen a FranÁa i que no han facturat mai maletes de color vermell.
--NOTA: Ha d'incloure tambÈ les persones que volen a FranÁa però que no han facturat mai.
SELECT P.NOM, P.NIF
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'FranÁa'
AND P.NIF NOT IN (SELECT DISTINCT P.NIF
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell')
GROUP BY P.NOM, P.NIF;

SELECT DISTINCT P.NIF
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell'
ORDER BY 1;

SELECT P.NOM, P.NIF
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'FranÁa'
GROUP BY P.NOM, P.NIF;



--Per a cada vol, calculi el nombre de maletes facturades. Retorneu el codi de vol, la data, 
-- i el nombre de maletes. (NOTA: les dates s'han de presentar al format 'DD/MM/YYYY')
SELECT V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY') AS DATA, COUNT(*) AS N_MALETES
FROM VOL V, MALETA M
WHERE M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA 
GROUP BY V.CODI_VOL, V.DATA
ORDER BY 1,2,3;


--Preu promig de les reserves pagats amb Visa. El numerador es 
--el preu total de les reserves pagades amb Visa i el denominador Ès el numero de
--reserves pagats amb Visa.
SELECT T1.PREU_TOTAL/T2.N_RESERVES AS PROMIG_RESERVES
FROM (SELECT SUM(PREU) AS PREU_TOTAL
FROM RESERVA
WHERE MODE_PAGAMENT = 'Visa') T1,
(SELECT COUNT(*) AS N_RESERVES
FROM RESERVA
WHERE MODE_PAGAMENT = 'Visa') T2
ORDER BY 1;


--3. Pais dels passatgers que han viatjat en primera classe mÈs de 5 vegades.
SELECT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND P.NIF IN (SELECT P.NIF
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = '1'
GROUP BY P.NIF
HAVING COUNT(*) > 5)
GROUP BY P.PAIS
ORDER BY 1;

--passatgers que han viatjat en primeraclasse mes de 5 cops:
SELECT P.NIF
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = '1'
GROUP BY P.NIF
HAVING COUNT(*) > 5;

--4. Codi(s) de vol del(s) avion(s) mÈs petit(s) (menor nombre de seients)
SELECT DISTINCT V.CODI_VOL
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA
HAVING COUNT(*) <= ALL(SELECT COUNT(*) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA);

--nombre seients de cada avio:
SELECT COUNT(*) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA
ORDER BY 1;


--8. Nombre de maletes de cada passatger. Retorna el nom del passatger, el seu NIF i el n˙mero
--de maletes que posseeix
SELECT P.NOM, P.NIF, COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM
ORDER BY 1,2,3;

--nombre maletes de cada passatger amb maleta
SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF;


--5. Noms de les persones que han facturat tots els mateixos colors de maleta que 
--Jose Luis Lamata Feliz (tambÈ poden haver facturat maletes d'altres colors)
SELECT DISTINCT R1.NOM
FROM (SELECT DISTINCT P.NOM, M.COLOR
        FROM PERSONA P, MALETA M
        WHERE P.NIF=M.NIF_PASSATGER) R1
WHERE (R1.NOM) NOT IN (SELECT R2.NOM
                        FROM (SELECT DISTINCT M.COLOR
                                FROM PERSONA P, MALETA M
                                WHERE P.NIF = M.NIF_PASSATGER 
                                AND P.NOM = 'Jose Luis Lamata Feliz') S,
                            (SELECT DISTINCT P.NOM, M.COLOR
                                FROM PERSONA P, MALETA M
                                WHERE P.NIF = M.NIF_PASSATGER) R2
                        WHERE (S.COLOR, R2.NOM) NOT IN (SELECT R3.COLOR, R3.NOM
                                                        FROM (SELECT DISTINCT P.NOM, M.COLOR
                                                                FROM PERSONA P, MALETA M
                                                                WHERE P.NIF = M.NIF_PASSATGER) R3))
ORDER BY 1;

--colors de les maletes que ha facturat Jose Luis Lamata Feliz:
SELECT M.COLOR
FROM MALETA M, PERSONA P
WHERE P.NIF = M.NIF_PASSATGER
AND P.NOM = 'Jose Luis Lamata Feliz'
GROUP BY M.COLOR
ORDER BY 1;


--2. Nombre de vols promig de totes les companyies. 
--NOTA: el numerador Ès el nombre de vols totals i el denominador el nombre total de companyies.
SELECT T1.VOLS_TOTALS/T2.N_COMPANYIES AS PROMIG
FROM (SELECT COUNT(*) AS VOLS_TOTALS
FROM VOL V) T1, 
(SELECT COUNT(DISTINCT V.COMPANYIA) AS N_COMPANYIES
FROM VOL V) T2
ORDER BY 1;

SELECT DISTINCT TV.TOTAL_VOLS/TC.TOTAL_COMPANYIES
FROM (SELECT SUM(COUNT(*)) AS TOTAL_VOLS
FROM VOL V
GROUP BY V.COMPANYIA) TV, (SELECT SUM(COUNT(DISTINCT V.COMPANYIA)) AS TOTAL_COMPANYIES
FROM VOL V
GROUP BY V.COMPANYIA) TC
ORDER BY 1;


--7. Percentage del pes total de de maletes que correspon als passatgers francesos. 
--NOTA: un percentatge s'expressa amb nombres entre 0 i 100.
SELECT (T1.PES_FRANCESOS*100)/T2.PES_TOTAL AS PERCENTATGE
FROM (SELECT SUM(M.PES) AS PES_FRANCESOS
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND P.PAIS = 'FranÁa') T1,
(SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M) T2
ORDER BY 1;



--6. Per a tots els vols de l'any 2014 mostra el NIF dels que volen i dels que han comprat 
--cada bitllet.

SELECT NIF_PASSATGER, NIF_CLIENT
FROM BITLLET B, RESERVA R
WHERE B.LOCALITZADOR = R.LOCALITZADOR
AND B.DATA BETWEEN TO_DATE('01/01/2014','DD/MM/YYYY') AND TO_DATE('31/12/2014','DD/MM/YYYY')
GROUP BY NIF_CLIENT, NIF_PASSATGER
ORDER BY 1,2;

1- 1 Punt

SELECT E.Nom, E.Interpret
FROM ESPECTACLES E
WHERE E.Tipus = 'Musical' AND
E.Data_Inicial = E.Data_Final
ORDER BY 1,2
-------------------------------------------
2- 1 Punt

SELECT DISTINCT Codi_Espectacle, Data,DNI_Client
FROM Entrades
WHERE Data BETWEEN TO_DATE('01/03/2012','dd/mm/yyyy') AND
TO_DATE('31/03/2012','dd/mm/yyyy') AND
Codi_Recinte IN (101, 103, 105)
ORDER BY Codi_Espectacle, Data
----------------------------------------------
4 - 1 Punt

SELECT R.DATA, TO_CHAR(R.HORA,'HH24:MI:SS') as HoraRep
FROM ESPECTACLES E, REPRESENTACIONS R
WHERE E.NOM = 'La extraña pareja' AND
R.Codi_Espectacle = E.Codi
ORDER BY 1,2
----------------------------------------------
6 - 1 Punt

SELECT Z.Zona, Z.Capacitat, P.Preu
FROM ZONES_RECINTE Z, ESPECTACLES E, PREUS_ESPECTACLES P
WHERE E.Nom = "La extraña pareja" AND
Z.Codi_Recinte = E.Codi_Recinte AND
P.Codi_Recinte = Z.Codi_Recinte AND
P.Zona = Z.Zona
ORDER BY 1,2,3
---------------------------------------------
7- 1 Punt

SELECT R.Ciutat, E.Nom, E.Data_Inicial, E.Data_Final
FROM RECINTES R, ESPECTACLES E
WHERE E.Interpret = 'El Tricicle' AND
E.Codi_Recinte = R.Codi
ORDER BY 1,2,3,4
----------------------------------------------
10- 1 Punt

SELECT S.Zona, S.Fila, S.Numero 
FROM Espectacles E, Seients S
WHERE E.Nom = 'Hamlet' AND
E.Codi_Recinte = S.Codi_Recinte
ORDER BY 1,2,3
----------------------------------------------
13 - 1 Punt

SELECT E2.Nom
FROM Espectacles E1, Espectacles E2
WHERE E1.Nom = 'Cianur i puntes de coixí' AND
E1.Codi_Recinte = E2.Codi_Recinte AND
E2.Data_Final > E1.Data_Final
----------------------------------------------
15 - 1 Punt

SELECT COUNT(*) as Nombre
FROM Espectacles E, Recintes R
WHERE R.Ciutat = 'Girona' AND
E.Codi_Recinte = R.Codi AND
(E.Data_Inicial Between TO_DATE('01/01/2011', 'dd/mm/yyyy') AND TO_DATE('31/12/2011', 'dd/mm/yyyy') OR
E.Data_Final BETWEEN TO_DATE('01/01/2011', 'dd/mm/yyyy') AND TO_DATE('31/12/2011', 'dd/mm/yyyy')) OR
(E.Data_Inicial < TO_DATE('01/01/2011', 'dd/mm/yyyy') AND E.Data_Final > TO_DATE('31/12/2011', 'dd/mm/yyyy'))
ORDER BY Nombre
----------------------------------------------
16- 1 Punt

SELECT COUNT(*) as CapacitatTotal
FROM Recintes R, Seients S
WHERE R.Nom = 'Romea' AND
S.Codi_Recinte = R.Codi
----------------------------------------------
17 - 1 Punt

SELECT MAX(P.Preu), MIN(P.Preu)
FROM Espectacles E, Preus_Espectacles P
WHERE E.Nom = 'Jazz a la tardor' AND
E.Codi = P.Codi_Espectacle
----------------------------------------------
22 - 1 Punt NO SE SI VA

SELECT COUNT (*) AS Num
FROM Entrades EN, Espectacles EC
WHERE E.Nom = "Mar i Cel" AND
EN.Codi_Espectacle = E.Codi
ORDER BY 1
----------------------------------------------
27 - 1 Punt

SELECT MAX(P.Preu) as MaxPreu, MIN(P.Preu) as MinPreu, E.Nom, E.Data_Inicial, E.Data_Final
FROM ESPECTACLES E, PREUS_ESPECTACLES P, RECINTES R
WHERE R.Nom = 'Liceu' AND
R.Codi = E.Codi_Recinte AND
E.Codi = P.Codi_Espectacle AND
R.Codi = P.Codi_Recinte
GROUP BY E.Nom, E.Data_Inicial, E.Data_final
ORDER BY 1,2,3,4
----------------------------------------------
32 - 2 Punts

SELECT r.nom
FROM RECINTES R, ZONES_RECINTE ZR
WHERE r.codi = zr.codi_recinte
AND R.CIUTAT = 'Barcelona'
GROUP BY R.nom
HAVING SUM(ZR.CAPACITAT)>
(
SELECT SUM(ZR2.CAPACITAT)
FROM RECINTES R2, ZONES_RECINTE ZR2
WHERE R2.CODI=ZR2.CODI_RECINTE
AND R2.nom='Victòria'
)
ORDER BY 1
-----------------------------------------------
42 - 2 Punts

SELECT DISTINCT E.CODI, E.NOM, pe.zona
FROM ESPECTACLES E, PREUS_ESPECTACLES PE
WHERE E.codi = pe.codi_espectacle
AND PE.PREU >=
(
  SELECT DISTINCT MAX (pe2.preu)
  FROM PREUS_ESPECTACLES PE2
)
ORDER BY 1,2,3
-----------------------------------------------
56 - 1 Punt ARA NO VA NOSE PQ (Va quan li dona la gana)

SELECT E.Nom, COUNT(*) AS Nombre
FROM Espectacles E, Entrades EN, Recintes R
WHERE R.Nom = ëLa Farandulaí AND
E.Codi_Recinte = R.Codi AND
EN.Codi_Espectacle = E.Codi
GROUP BY E.Nom
ORDER BY Nombre DESC
-----------------------------------------------
58 - 2 Punts COMPROBAR!!! NO FICAR QUE PETA

SELECT ES.DNI, ES.Nom, ES.Cognoms
FROM ESPECTADORS ES, ENTRADES EN
WHERE EN.DNI_Client = ES.DNI
GROUP BY EN.DNI_Client
HAVING COUNT(*) >=
ALL (SELECT COUNT(*)
FROM ENTRADES EN2,
GROUP BY EN2.DNI_Client)
ORDER BY 1,2,3
-----------------------------------------------
61 - 2 Punts

SELECT R.Nom, R.Ciutat, SUM (ZR.Capacitat) as CapacitatFinal
FROM RECINTES R, ZONES_RECINTE ZR
WHERE R.Codi = ZR.Codi_Recinte
GROUP BY R.Nom, R.Ciutat
HAVING SUM(ZR.Capacitat) >= 
ALL ( SELECT SUM(ZR2.Capacitat)
FROM Zones_Recinte ZR2
GROUP BY ZR2.Codi_Recinte)
ORDER BY 1,2,3
-----------------------------------------------
76 - 2 Punts

SELECT DISTINCT E.NOM
FROM ESPECTACLES E, PREUS_ESPECTACLES PE
WHERE E.CODI = PE.CODI_ESPECTACLE
AND PE.PREU <=
(
SELECT MIN (PE2.PREU)
FROM ESPECTACLES E2, PREUS_ESPECTACLES PE2
WHERE E2.CODI  = PE2.CODI_ESPECTACLE
)
ORDER BY 1
------------------------------------------------
97 - 1 Punt

SELECT r.codi, r.nom, r.ciutat
FROM RECINTES R, ZONES_RECINTE Z
WHERE r.codi = z.codi_recinte
GROUP BY r.codi, r.nom, r.ciutat
HAVING COUNT(*) > 1
ORDER BY 1,2,3
------------------------------------------------

-- Per a cada aeroport i terminal, troba el nom de l'aeroport, terminal i nombre d'àrees
SELECT a.nom, pe.terminal, SUM(pe.area) AS nombre_arees
FROM aeroport a INNER JOIN porta_embarcament pe ON pe.codi_aeroport=a.codi_aeroport
GROUP BY a.nom, pe.terminal
ORDER BY 1,2,3

-- Per cada aeroport llisti el nom de l'aeroport, la terminal, l'àrea i el nombre de portes d'aquesta àrea.
SELECT a.nom, pe.terminal, pe.area, MAX (pe.porta) AS Nombre_Portes
FROM aeroport a INNER JOIN porta_embarcament pe ON pe.codi_aeroport=a.codi_aeroport
GROUP BY a.nom, pe.terminal, pe.area
ORDER BY 1,2,3,4

-- Nombre de maletes que ha facturat Chema Pamundi per cada vol que ha fet. Volem recuperar codi del vol, data i nombre de maletes.
SELECT m.codi_vol, TO_CHAR(m.data,'DD/MM/YYYY') AS DATA, COUNT (m.codi_maleta) AS MALETES
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE p.nom = 'Chema Pamundi'
GROUP BY m.codi_vol, m.data
ORDER BY 1,2,3

-- Pes total de les maletes facturades per passatgers espanyols.

SELECT SUM(m.pes) AS PES
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE p.pais = 'Espanya'
ORDER BY 1

-- Quan pes ha facturat l'Alberto Carlos Huevos al vol amb destinació Rotterdam de la data 02 de juny del 2013?

SELECT SUM(m.pes) AS PES
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE p.nom = 'Alberto Carlos Huevos' 
AND b.codi_vol IN (SELECT v.codi_vol
FROM vol v INNER JOIN aeroport a ON v.destinacio = a.codi_aeroport
WHERE a.ciutat = 'Rotterdam' AND  v.data = TO_DATE('02/06/2013','DD/MM/YYYY') )
AND b.data IN (SELECT v.data
FROM vol v INNER JOIN aeroport a ON v.destinacio = a.codi_aeroport
WHERE a.ciutat = 'Rotterdam' AND  v.data = TO_DATE('02/06/2013','DD/MM/YYYY') )
ORDER BY 1

-- Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117 del 27 de Agosto de 2013.

SELECT COUNT (s.codi_vol) AS N_Seients_Reservades
FROM seient s
WHERE s.codi_vol = 'VLG2117' AND s.data=TO_DATE('27/08/2013','DD/MM/YYYY') AND s.nif_passatger IS NOT NULL
ORDER BY 1

-- Nombre de persones mexicanes.

SELECT COUNT(nif)
FROM persona
WHERE pais = 'Mexic'
ORDER BY 1

-- Número de porta amb més d'un vol assignat. Mostrar codi de l'aeroport al que pertany, terminal, area i porta.

SELECT origen, terminal, area, porta
FROM vol
WHERE porta IS NOT NULL 
HAVING COUNT (codi_vol) > 1
GROUP BY origen, terminal, area, porta
ORDER BY 1,2,3,4

-- Nom de la companyia i nombre de vols que ha fet as N_VOLS.

SELECT companyia, COUNT (codi_vol) AS N_VOLS
FROM vol
GROUP BY companyia
ORDER BY 1,2

-- Nom dels passatgers que han volat almenys 5 vegades, ordenats pel nombre de vegades que han volat. Mostrar també el nombre de vegades que han volat com a N_VEGADES.
-- **OK
SELECT P.NOM, COUNT(*) AS N_VEGADES
FROM PERSONA P, BITLLET B
WHERE P.NIF=B.NIF_PASSATGER
GROUP BY P.NOM
HAVING COUNT(*) > 4
ORDER BY 2,1

-- ** ERROR
SELECT COUNT (*) AS N_VEGADES, b.nif_passatger, p.nom
FROM bitllet b LEFT JOIN persona p ON b.nif_passatger = p.nif
GROUP BY b.nif_passatger, p.nom
HAVING COUNT (*) >= 5
ORDER BY 1,2,3

-- Nombre màxim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(max_maletes) AS max_maletes 
FROM (SELECT COUNT (m.codi_maleta) AS max_maletes FROM maleta m GROUP BY m.nif_passatger,m.data)
ORDER BY 1

-- Pes total facturat (com a Pes_Total) pel vol KLM4304 de l'9 d'Octubre de 2013
SELECT SUM(m.pes) AS Pes_Total
FROM vol v INNER JOIN bitllet b ON v.codi_vol=b.codi_vol AND v.data=b.data
INNER JOIN maleta m ON m.nif_passatger = b.nif_passatger AND m.codi_vol=b.codi_vol AND m.data=b.data
WHERE v.codi_vol = 'KLM4304'AND v.data = TO_DATE('09/10/2013', 'DD/MM/YYYY')
ORDER BY 1

-- 	Nom de les Companyies amb mès de 10 vols.
SELECT v.companyia 
FROM vol v
GROUP BY v.companyia
HAVING COUNT(v.codi_vol)>10
ORDER BY 1

-- Passatgers que han facturat més de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, nom, codi_vol, data i pes_total
-- ** ERROR (DESCONEGUT)
SELECT P.NIF, P.NOM, B.CODI_VOL, b.data, SUM(M.PES) AS PES_TOTAL
FROM persona p, bitllet b, maleta m WHERE p.nif = b.nif_passatger AND B.NIF_PASSATGER = M.NIF_PASSATGER AND B.CODI_VOL = M.CODI_VOL  AND B.DATA = M.DATA
GROUP BY P.NIF, P.NOM, B.CODI_VOL, B.DATA
HAVING SUM(M.PES) > 15
ORDER BY 1, 2, 3, 4, 5

-- NIF i nom dels passatgers que han facturat més de 15 kgs. de maletes en algun dels seus vols. Atributs: NIF, NOM, CODI_VOL, DATA i PES_TOTAL
-- ** ERROR (DESCONEGUT)
SELECT p.nif, p.nom, m.codi_vol, m.data, SUM(m.pes) AS pes_total
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger=b.nif_passatger AND m.codi_vol=b.codi_vol AND m.data=b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
GROUP BY p.nif, p.nom, m.codi_vol, m.data
HAVING SUM(m.pes) > 15
ORDER BY 1,2,3,4

-- Nom dels passatgers que han facturat més d'una maleta vermella.
SELECT p.nom
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger=b.nif_passatger AND m.codi_vol=b.codi_vol AND m.data=b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE m.color = 'vermell'
GROUP BY p.nom
HAVING COUNT (m.color)>1
ORDER BY 1

-- Quantes maletes ha facturat en Aitor Tilla al vol amb destinació Pekin de la data 1/10/2013?
SELECT COUNT(m.codi_maleta)
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger=b.nif_passatger AND m.codi_vol=b.codi_vol AND m.data=b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
INNER JOIN vol v ON v.codi_vol=b.codi_vol AND v.data=b.data
INNER JOIN aeroport a ON a.codi_aeroport=v.destinacio
WHERE p.nom='Aitor Tilla' AND a.ciutat='Pekin' AND v.data = TO_DATE('01/10/2013', 'DD/MM/YYYY')
ORDER BY 1

-- Nacionalitats representades per més de dos passatgers amb bitllets.
SELECT p.pais
FROM persona p RIGHT JOIN bitllet b ON p.nif = b.nif_passatger
GROUP BY p.pais
HAVING COUNT(*) > 2
ORDER BY 1

-- El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(m.pes) AS pes_total
FROM maleta m
WHERE m.nif_passatger IN( SELECT nif FROM persona WHERE pais ='Italia')
ORDER BY 1

-- Nombre de Bitllets per compradors. Es demana nom del comprador i nombre de bitllets comprats.
-- **ERROR**
SELECT p.nom, COUNT(*) AS bitllets_comprats
FROM persona p LEFT JOIN reserva r ON p.nif=r.nif_client
GROUP BY p.nom
ORDER BY 1,2

-- Pes total de les maletes facturades per passatgers espanyols.
SELECT SUM(m.pes)
FROM maleta m, persona p WHERE p.nif = m.nif_passatger AND p.pais = 'Espanya'
ORDER BY 1

-- Data i codi de vol dels vols amb més de 3 passatgers.
-- ** ERROR (desconegut) 
SELECT b.data, b.codi_vol
FROM bitllet b
GROUP BY b.data, b.codi_vol
HAVING COUNT(b.nif_passatger) > 3
ORDER BY 1,2

-- Nombre de passatgers dels vols d'IBERIA per països. Es demana país i nombre de passatgers.
SELECT p.pais, COUNT(v.codi_vol)
FROM vol v INNER JOIN bitllet b ON v.codi_vol = b.codi_vol AND v.data=b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE v.companyia ='IBERIA'
GROUP BY p.pais
ORDER BY 1

-- Nombre de passatgers agrupats segons país de destinació (en el vol). Mostreu país i nombre de passatgers.
SELECT a.pais, COUNT(*)
FROM vol v INNER JOIN bitllet b ON v.codi_vol = b.codi_vol AND v.data=b.data
INNER JOIN aeroport a ON a.codi_aeroport = v.destinacio
GROUP BY a.pais
ORDER BY 1,2

-- Codi, data i pes total dels vols que han facturat, en total, un pes igual o superior a 26 Kgs.
-- ** ERRROR ** 
SELECT m.codi_vol, m.data, SUM(m.pes) AS pes_total
FROM maleta m
GROUP BY m.codi_vol, m.data
HAVING SUM (m.pes)>=26
ORDER BY 1,2,3

-- Promig de portes per aeroport. Aclariment: el numerador es el nombre total de portes de tots els aeroports i el denominador el nombre total d'aeroports.
-- **ERROR**
SELECT AVG(pe.porta) AS Promig
FROM aeroport a INNER JOIN porta_embarcament pe ON a.codi_aeroport=pe.codi_aeroport
ORDER BY 1

SELECT (SUM(pe.porta) / COUNT (a.codi_aeroport)) AS Promig
FROM aeroport a INNER JOIN porta_embarcament pe ON a.codi_aeroport=pe.codi_aeroport
ORDER BY 1
-- OK --
SELECT total_portes.n_portes/total_aeroports.n_aeroports AS PROMIG_PORTES
FROM (SELECT COUNT(*) AS n_portes FROM porta_embarcament) total_portes, (SELECT COUNT(*) AS n_aeroports FROM aeroport) total_aeroports
ORDER BY 1

-- Nombre de passatgers (com a N_Passatgers) amb cognom Blanco té el vol amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_Passatgers
FROM bitllet
WHERE nif_passatger IN (SELECT p.nif FROM persona p WHERE p.nom LIKE '%Blanco%') AND codi_vol='IBE2119' AND data = TO_DATE('30/10/2013', 'DD/MM/YYYY')
ORDER BY 1

-- NIF, nom del(s) passatger(s) i pes total que ha(n) facturat menys pes en el conjunt de tots els seus bitllets.
SELECT p.nif, p.nom, SUM(m.pes) AS pes_total
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
GROUP BY p.nif, p.nom
HAVING SUM(m.pes) <= ALL(
    SELECT (SUM(m.pes))
    FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
    INNER JOIN persona p ON p.nif = b.nif_passatger
    GROUP BY p.nif, p.nom)
ORDER BY 1,2,3

-- Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers del(s) vol(s) que té(nen) el nombre de passatges més petit.
SELECT v.origen, v.codi_vol, TO_CHAR(v.data,'DD/MM/YYYY') AS data, COUNT(*) AS n_passatgers
FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
GROUP BY v.origen, v.codi_vol, v.data
HAVING COUNT(*) = (
    SELECT MIN(COUNT(*)) 
    FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
    GROUP BY v.codi_vol, v.data)
ORDER BY 1,2,3,4

-- Tipus d'avió(ns) que fa(n) mes vols amb origen Espanya.

SELECT DISTINCT v.tipus_avio
FROM vol v INNER JOIN aeroport a ON a.codi_aeroport=v.destinacio
WHERE a.pais = 'Espanya'
GROUP BY v.tipus_avio
HAVING COUNT(v.codi_vol) >= ALL(
    SELECT COUNT(v.codi_vol)
    FROM vol v INNER JOIN aeroport a ON a.codi_aeroport=v.destinacio
    WHERE a.pais = 'Espanya'
    GROUP BY v.tipus_avio)
ORDER BY 1

-- Nom(es) de la(es) persona(es) que més cops ha(n) pagat amb Paypal. Retorneu també el número de cops que ho ha(n) fet.
SELECT P.NOM 
FROM RESERVA R, PERSONA P
WHERE R.NIF_CLIENT = P.NIF
AND MODE_PAGAMENT = 'Paypal'
GROUP BY R.NIF_CLIENT, P.NOM
HAVING COUNT() = (SELECT MAX(COUNT())
                    FROM RESERVA 
                    WHERE MODE_PAGAMENT = 'Paypal'
                    GROUP BY NIF_CLIENT)
ORDER BY 1;

-- Nom de totes les persones que son del mateix país que "Domingo Diaz Festivo" (ell inclòs).

SELECT DISTINCT nom
FROM persona
WHERE pais IN (
    SELECT p.pais 
    FROM persona p 
    WHERE p.nom = 'Domingo Diaz Festivo')
ORDER BY 1

-- NIF, nom i pes total facturat del passatger amb menys pes facturat en el conjunt de tots els seus bitllets.

SELECT p.nif, p.nom, SUM(m.pes) AS pes_facturat
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
GROUP BY p.nif, p.nom
HAVING SUM(m.pes) <= ALL(
    SELECT (SUM(m.pes))
    FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
    INNER JOIN persona p ON p.nif = b.nif_passatger
    GROUP BY p.nif, p.nom)
ORDER BY 1,2,3

-- Nombre de vegades que apareix la nacionalitat més freqüent entre les persones de la BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.

SELECT DISTINCT p.pais, COUNT(*) AS n_vegades
FROM persona p
GROUP BY p.pais
HAVING COUNT(*) = (
    SELECT MAX(COUNT(*)) 
    FROM persona p
    GROUP BY p.pais)
ORDER BY 1,2

-- Nom de la(/es) companyia(/es) que té(nen) més bitllets reservats en algun dels seus vols. Atributs: companyia, codi i data de vol, n_seients_reservats.
SELECT v.companyia, v.codi_vol, v.data, COUNT(*) AS n_seients_reservats
FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
GROUP BY v.companyia, v.codi_vol, v.data
HAVING COUNT(*) >= ALL(
    SELECT COUNT(*)
    FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
    GROUP BY v.companyia, v.codi_vol, v.data)
ORDER BY 1,2,3,4

-- Color(s) de la(/es) maleta(/es) facturada(/es) més pesada(/es).
SELECT DISTINCT m.color
FROM maleta m
WHERE m.pes =(
    SELECT MAX(m.pes)
    FROM maleta m)
ORDER BY 1

-- Nom(es) del passatger/s que ha/n facturat la maleta de més pes.

SELECT DISTINCT p.nom
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE m.pes = (
    SELECT MAX(m.pes)
    FROM maleta m)
ORDER BY 1

-- Nom(es) del(s) aeroport(s) amb més terminals.

SELECT DISTINCT a.nom
FROM aeroport a INNER JOIN porta_embarcament pe ON a.codi_aeroport=pe.codi_aeroport
GROUP BY a.nom
HAVING COUNT(DISTINCT pe.terminal) >= ALL(
    SELECT COUNT(DISTINCT PE.TERMINAL) 
    FROM aeroport a INNER JOIN porta_embarcament pe ON a.codi_aeroport=pe.codi_aeroport
    GROUP BY a.nom)
ORDER BY 1

-- Nom(es) de la(es) companyia(es) que ha(n) estat reservada(es) per més clients (persones) d'Amsterdam.
SELECT V.COMPANYIA
FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
INNER JOIN reserva r ON r.localitzador = b.localitzador 
INNER JOIN persona p ON p.nif = r.nif_client
WHERE p.poblacio = 'Amsterdam'
GROUP BY v.companyia
HAVING COUNT(*)=(
    SELECT MAX(COUNT(*))
    FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
    INNER JOIN reserva r ON r.localitzador = b.localitzador 
    INNER JOIN persona p ON p.nif = r.nif_client
    WHERE p.poblacio = 'Amsterdam'
    GROUP BY v.companyia)
ORDER BY 1

-- Dates de vol (DD/MM/YYYY) del(s) avió(ns) amb mes capacitat (major nombre de seients)

SELECT TO_CHAR(s.data,'DD/MM/YYYY')
FROM seient s
GROUP BY s.codi_vol, s.data
HAVING COUNT(*) = (
    SELECT MAX(COUNT(*))
    FROM seient s
    GROUP BY s.codi_vol, s.data)
ORDER BY 1

-- Tipus d'avió(ns) amb més files.
SELECT DISTINCT v.tipus_avio
FROM vol v INNER JOIN seient s ON v.codi_vol = s.codi_vol AND v.data =s.data
WHERE s.fila IN (SELECT MAX(s.fila) FROM seient s)
ORDER BY 1

-- Tipus d'avió(ns) que té(nen) més passatgers no espanyols. 
SELECT DISTINCT v.tipus_avio
FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE p.pais NOT IN (
    SELECT DISTINCT v.tipus_avio
    FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
    INNER JOIN persona p ON p.nif = b.nif_passatger
    WHERE p.pais = 'Espanya')
GROUP BY v.tipus_avio
HAVING COUNT(*) >= ALL(
    SELECT COUNT(*)
    FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
    INNER JOIN persona p ON p.nif = b.nif_passatger
    WHERE p.pais NOT IN (
        SELECT DISTINCT v.tipus_avio
        FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
        INNER JOIN persona p ON p.nif = b.nif_passatger
        WHERE p.pais = 'Espanya')
    GROUP BY V.TIPUS_AVIO)
ORDER BY 1

-- Nom(es) de la(es) companyia(es) que opera(n) amb més tipus d'avio.
SELECT v.companyia
FROM vol v
GROUP BY v.companyia
HAVING COUNT(v.tipus_avio) >= ALL(
    SELECT COUNT(v.tipus_avio)
    FROM vol v
    GROUP BY v.companyia)
ORDER BY 1

-- 	Nom del(s) aeroport(s) amb el mínim número de terminals.
SELECT DISTINCT a.nom
FROM aeroport a INNER JOIN porta_embarcament pe ON a.codi_aeroport=pe.codi_aeroport
GROUP BY a.nom
HAVING COUNT(DISTINCT pe.terminal) <= ALL(
    SELECT COUNT(DISTINCT pe.terminal) 
    FROM aeroport a INNER JOIN porta_embarcament pe ON a.codi_aeroport=pe.codi_aeroport
    GROUP BY a.nom)
ORDER BY 1

-- NIF i nom de la(/es) persona(/es) que ha(n) reservat més bitllets.
-- ERROR --
SELECT p.nif, p.nom
FROM bitllet b INNER JOIN persona p ON p.nif = b.nif_passatger
GROUP BY p.nif, p.nom
HAVING COUNT(*) = (
    SELECT MAX(COUNT(*))
    FROM reserva r 
    GROUP BY r.nif_client)
ORDER BY 1,2

-- NIF i nom del(s) passatger(s) que té mes maletes registrades al seu nom.

SELECT p.nif, p.nom
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
INNER JOIN persona p ON p.nif = b.nif_passatger
GROUP BY p.nif, p.nom
HAVING COUNT(*) >= ALL(
    SELECT COUNT(*)
    FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
    INNER JOIN persona p ON p.nif = b.nif_passatger
    GROUP BY p.nif, p.nom)
ORDER BY 1     


-- Nom(es) de la companyia(es) que fa(n) més vols amb origen Italia.

SELECT v.companyia
FROM vol v INNER JOIN aeroport a ON v.destinacio = a.codi_aeroport
WHERE a.pais = 'Italia'
GROUP BY v.companyia
HAVING COUNT(*) = (
    SELECT MAX(COUNT(*))
    FROM vol v INNER JOIN aeroport a ON v.destinacio = a.codi_aeroport
    WHERE a.pais = 'Italia'
    GROUP BY v.companyia)
ORDER BY 1

-- Nom de la(s) companyia(es) amb vols d'origen a tots els aeroports de la Xina.
(SELECT DISTINCT v.companyia
FROM vol v INNER JOIN aeroport a ON a.codi_aeroport=v.destinacio
WHERE a.pais = 'Xina')
MINUS
(SELECT DISTINCT v.companyia				
FROM vol v INNER JOIN aeroport a ON a.codi_aeroport=v.destinacio
WHERE a.pais = 'Xina' AND v.companyia NOT IN (
    SELECT v.companyia 
    FROM vol v INNER JOIN aeroport a ON a.codi_aeroport=v.destinacio
    WHERE a.pais = 'Xina')
)
ORDER BY 1

-- Dades de contacte (nom, email, telèfon) de les persones que no han volat mai a la Xina.
SELECT DISTINCT p.nom, p.mail, p.telefon 
FROM persona p
MINUS
SELECT DISTINCT p.nom, p.mail, p.telefon 
FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data INNER JOIN persona p ON p.nif = b.nif_passatger 
WHERE v.destinacio = 'PEK'
ORDER BY 1,2,3

-- Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/2013

SELECT s.fila, s.lletra
FROM vol v INNER JOIN seient s ON v.codi_vol = s.codi_vol AND v.data = s.data
WHERE s.data = TO_DATE('31/07/2013','DD/MM/YYYY') AND s.codi_vol = 'AEA2195' AND s.nif_passatger IS NULL
ORDER BY 1,2

-- Noms dels aeroports i nom de les ciutats de destinació d'aquells vols a les que els falta un passatger per estar plens

SELECT DISTINCT a.nom, a.ciutat
FROM
(SELECT codi_vol, data, COUNT(*) AS total_seients FROM seient GROUP BY codi_vol, data) ts,
(SELECT codi_vol, data, COUNT(*) AS seients_ocupats FROM bitllet GROUP BY codi_vol, data) so, 
vol v, aeroport a
WHERE ts.codi_vol = so.codi_vol 
AND ts.data = so.data 
AND so.seients_ocupats = ts.total_seients + 1 
AND so.codi_vol = v.codi_vol 
AND so.data = v.data 
AND v.destinacio = a.codi_aeroport
ORDER BY 1,2

-- Nom del(s) passatger(s) que sempre vola amb primera classe

SELECT DISTINCT p.nom
FROM bitllet b INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE p.nom NOT IN (SELECT p.nom FROM bitllet b INNER JOIN persona p ON p.nif = b.nif_passatger WHERE b.classe = '2')
ORDER BY 1

-- Numero màxim de bitllets (com a Max_Bitllets) que ha comprat en una sola reserva en Narciso Blanco.

SELECT MAX(COUNT(*)) AS Max_Bitllets
FROM bitllet b INNER JOIN reserva r ON r.localitzador = b.localitzador 
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE p.nom = 'Narciso Blanco'
GROUP BY r.data
ORDER BY 1

-- Companyies que volen a tots els aeroports de Espanya

SELECT DISTINCT v.companyia
FROM vol v INNER JOIN aeroport a ON a.codi_aeroport = v.destinacio
WHERE a.pais = 'Espanya'					
GROUP BY v.companyia
HAVING COUNT(DISTINCT a.codi_aeroport) = (SELECT COUNT(a.codi_aeroport) FROM aeroport a WHERE a.pais = 'Espanya')
ORDER BY 1

-- Numero de places lliures pel vol AEA2195 amb data 31/07/13

SELECT (tp.total_places - tot.total_ocupat) AS Places_Lliures
FROM 
(SELECT COUNT(*) AS total_places FROM vol v INNER JOIN seient s ON v.codi_vol = s.codi_vol AND v.data = s.data WHERE v.codi_vol = 'AEA2195' AND TO_CHAR(v.data,'DD/MM/YY') = '31/07/13') tp,
(SELECT COUNT(*) AS total_ocupat FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data WHERE v.codi_vol = 'AEA2195' AND TO_CHAR(v.data,'DD/MM/YY') = '31/07/13') tot
ORDER BY 1

-- Aeroports que no tenen vol en los tres primeros meses de 2014

SELECT DISTINCT a.nom
FROM aeroport a
WHERE a.codi_aeroport NOT IN
(SELECT v.destinacio FROM vol v WHERE v.data >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND v.data < TO_DATE('01/04/2014', 'DD/MM/YYYY')) 
AND a.codi_aeroport NOT IN
(SELECT v.origen FROM vol v WHERE v.data >= TO_DATE('01/01/2014', 'DD/MM/YYYY') AND v.data < TO_DATE('01/04/2014', 'DD/MM/YYYY'))
ORDER BY 1

-- Percentatge d'espanyols als vols amb destinació Munich. (NOTA: el percentatge és sempre un numero entre 0 i 100)
SELECT (total_espanyols / total_viatgers) * 100 
FROM
    (SELECT COUNT(*) AS total_viatgers FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data WHERE v.destinacio = 'MUC'),
    (SELECT COUNT(*) AS total_espanyols FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data INNER JOIN persona p ON p.nif = b.nif_passatger WHERE v.destinacio = 'MUC' AND p.pais = 'Espanya')

-- Numero de portes lliures de la terminal 3, area 3, del aeroport "Berlin-Schönefeld International Airport" el dia 04/07/2013.
SELECT (pt.portes_totals - po.portes_ocupades) AS Portes_Lliures
FROM
(SELECT COUNT(*) AS portes_totals FROM porta_embarcament pe WHERE pe.codi_aeroport = 'SXF' AND pe.terminal = '3' AND pe.area = '3') pt,
(SELECT COUNT (*) AS portes_ocupades FROM vol v WHERE v.data = TO_DATE('04/07/2013','DD/MM/YYYY') AND v.terminal = '3' AND v.area = '3' AND v.origen = 'SXF') po
ORDER BY 1

-- Quines portes queden lliures de la terminal 3, area 3, del aeroport 'Berlin-Schönefeld International Airport' el dia 04/07/2013?
SELECT pe.porta FROM porta_embarcament pe 
WHERE pe.codi_aeroport = 'SXF' AND pe.terminal = '3' AND pe.area = '3' AND pe.porta NOT IN (SELECT v.porta FROM vol v WHERE v.data = TO_DATE('04/07/2013','DD/MM/YYYY') AND v.terminal = '3' AND v.area = '3' AND v.origen = 'SXF')
ORDER BY 1

-- Nom dels passatgers que tenen bitllet, però no han fet mai cap reserva
(SELECT DISTINCT p.nom FROM bitllet b INNER JOIN persona p ON p.nif = b.nif_passatger)
MINUS
(SELECT DISTINCT p.nom FROM persona p, bitllet b, reserva r WHERE p.nif = b.nif_passatger AND r.nif_client = p.nif AND b.nif_passatger = r.nif_client)


-- Companyies que tenen, almenys, els mateixos tipus d'avions que Vueling Airlines. (El resultat també ha de tornar  Vueling airlines).  
-- ERROR
SELECT DISTINCT v.companyia
FROM vol v
WHERE v.tipus_avio IN (SELECT DISTINCT v.tipus_avio FROM vol v WHERE v.companyia = 'VUELING AIRLINES')
ORDER BY 1

-- Per al vol AEA2195 amb data 31/07/13, quin és el percentatge d'ocupació? (NOTA: el percentatge és sempre un numero entre 0 i 100)
SELECT (tot.total_ocupat / tp.total_places) * 100 AS Places_Lliures
FROM 
(SELECT COUNT(*) AS total_places FROM vol v INNER JOIN seient s ON v.codi_vol = s.codi_vol AND v.data = s.data WHERE v.codi_vol = 'AEA2195' AND TO_CHAR(v.data,'DD/MM/YY') = '31/07/13') tp,
(SELECT COUNT(*) AS total_ocupat FROM bitllet b INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data WHERE v.codi_vol = 'AEA2195' AND TO_CHAR(v.data,'DD/MM/YY') = '31/07/13') tot
ORDER BY 1

-- Del total de maletes facturades, quin percentatge són de color vermell. (NOTA: el percentatge és sempre un numero entre 0 i 100)
SELECT (tv.total_vermelles / tm.total_maletes) * 100
FROM (SELECT COUNT(*) AS total_maletes FROM maleta) tm,
(SELECT COUNT(*) AS total_vermelles FROM maleta WHERE color = 'vermell') tv

-- Portes d'embarcament (terminal, area, porta) de l'aeroport JFK on no han sortit mai cap vol amb destinació França.

-- Ciutat de destinació dels vols que tenen entre els seus passatgers persones de totes les ciutats de Argentina.

-- Nom i NIF de les persones que han facturat maletes però cap d'elles de color blau o vermell.
-- Companies amb algun vol al septembre del 2013
-- Noms de les companyies que volen a tots els mateixos aeroports als quals ha volat Carmelo Coton (també poden volar a altres aeroports)
-- Nombre de vols de la companyia CZECH AIRLINES. 
-- Nombre de destinaciones diferents per a cada passatger. Cal mostrar nom del passatger i el nombre de destinacions.
-- NIF dels passatgers que sempre han agafat el vol d'ANADA des del mateix aeroport.
SELECT b.nif_passatger
FROM bitllet b INNER JOIN vol v ON b.codi_vol = v.codi_vol AND b.data = v.data
WHERE b.anada_tornada LIKE 'ANADA'
GROUP BY b.nif_passatger, v.destinacio
HAVING b.nif_passatger IN (
    SELECT b.nif_passatger
    FROM bitllet b INNER JOIN vol v ON b.codi_vol = v.codi_vol AND b.data = v.data
    WHERE b.anada_tornada LIKE 'ANADA'
    GROUP BY b.nif_passatger
    HAVING count(DISTINCT v.destinacio) < 2
)
ORDER BY 1

-- Codi(s) de vol del(s) avion(s) més petit(s) (menor nombre de seients)
-- Pais dels passatgers que han viatjat en primera classe més de 5 vegades.

-- Ciutats espanyoles amb aeroports.

SELECT UNIQUE a.Ciutat 
FROM aeroport a 
WHERE a.pais = 'Espanya'
ORDER BY 1

-- El codi de la maleta, el pes i les mides de les maletes que ha facturat el passatger Jose Luis Lamata Feliz en els seus vols.

SELECT m.codi_maleta, m.pes, m.mides
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger 
AND m.codi_vol = b.codi_vol
AND m.data = b.data 
WHERE b.nif_passatger = (SELECT p.nif FROM persona p WHERE p.nom='Jose Luis Lamata Feliz')
ORDER BY 1,2,3

-- Nom del passatger i pes de les maletes facturades per passatgers espanyols.

SELECT UNIQUE p.nom, m.pes
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger 
AND m.codi_vol = b.codi_vol
AND m.data = b.data 
INNER JOIN persona p ON p.nif = b.nif_passatger
WHERE b.nif_passatger IN (SELECT nif FROM persona WHERE pais='Espanya')
ORDER BY 1

-- Nom dels persones vegetarians.
SELECT UNIQUE nom
FROM persona
WHERE observacions='Vegetaria/na'
ORDER BY 1

-- Tots els models diferents d'avió que existeixen
SELECT DISTINCT tipus_avio
FROM vol
ORDER BY 1

-- Nom de les passatgers que han nascut entre el 05/06/1964 i el 03/09/1985.
SELECT UNIQUE nom
FROM persona
WHERE data_naixement BETWEEN TO_DATE('05/06/1964', 'DD/MM/YYYY') AND TO_DATE('03/09/1985', 'DD/MM/YYYY')
ORDER BY 1

-- (Reserves Vol) El codi de la maleta i el localitzador de la reserva de totes les maletes.

SELECT m.codi_maleta, b.localitzador
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger 
AND m.codi_vol = b.codi_vol
AND m.data = b.data 
ORDER BY 1, 2

-- (Reserves Vol) Nom dels passatgers que han nascut abans de l'any 1975.
SELECT nom
FROM persona
WHERE data_naixement < TO_DATE('01/01/1975','DD/MM/YYYY') 
ORDER BY 1

-- Les companyies amb vols al mes d'Agost de 2013.
SELECT UNIQUE v.companyia
FROM vol v 
WHERE v.data BETWEEN TO_DATE('01/08/2013','DD/MM/YYYY')  AND TO_DATE('31/08/2013','DD/MM/YYYY') 
ORDER BY 1

-- Codi de vol i data dels bitllets del passatger Alan Brito.
SELECT b.codi_vol, TO_CHAR(b.data, 'DD/MM/YYYY') AS data 
FROM bitllet b 
WHERE b.nif_passatger = (SELECT p.nif FROM persona p WHERE p.nom = 'Alan Brito')
ORDER BY 1,2

-- Forma de pagament de les reserves fetes per a persones de nacionalitat alemana
SELECT UNIQUE r.mode_pagament
FROM reserva r
WHERE r.nif_client IN (SELECT nif FROM persona WHERE pais='Alemanya')
ORDER BY 1

-- Nom dels passatgers que tenen alguna maleta que pesa més de 10 kilos. Especifica també en quin vol.

SELECT UNIQUE p.nom, v.codi_vol
FROM maleta m INNER JOIN bitllet b ON m.nif_passatger = b.nif_passatger 
AND m.codi_vol = b.codi_vol
AND m.data = b.data 
INNER JOIN persona p ON p.nif = b.nif_passatger
INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
WHERE m.pes > 10 
ORDER BY 1,2

-- Nom i edat dels passatgers del vol AEA2159 del 25/07/2013.

SELECT UNIQUE p.nom, TO_CHAR(p.data_naixement, 'DD/MM/YYYY') AS data_naixement 
FROM persona p INNER JOIN bitllet b ON p.nif = b.nif_passatger
INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
WHERE v.codi_vol = 'AEA2159' AND v.data=TO_DATE('25/07/2013','DD/MM/YYYY')
ORDER BY 1,2

-- Per al vol amb data 02/06/14 i codi RAM964 digueu el NIF dels passatgers que volen i el dels que han comprat el bitllet.

SELECT b.nif_passatger, r.nif_client 
FROM bitllet b INNER JOIN vol v ON b.codi_vol = v.codi_vol AND b.data = v.data
INNER JOIN reserva r ON r.localitzador = b.localitzador
WHERE v.data = TO_DATE('02/06/14','DD/MM/YYYY') AND v.codi_vol = 'RAM964'
ORDER BY 1,2

-- Nacionalitat dels passatgers dels vols amb destinació Paris Orly Airport.

SELECT UNIQUE p.pais
FROM persona p INNER JOIN bitllet b ON p.nif=b.nif_passatger
INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
WHERE v.destinacio = 'ORY'
ORDER BY 1

-- Codi_vol i data dels vols amb origen algún aeroport rus.

SELECT UNIQUE v.codi_vol, TO_CHAR(v.data, 'DD/MM/YYYY') AS data 
FROM vol v
WHERE v.origen IN (SELECT codi_aeroport FROM aeroport WHERE pais='Russia')
ORDER BY 1,2

-- Nacionalitat dels passatgers del vols amb destinació París

SELECT UNIQUE p.pais
FROM persona p INNER JOIN bitllet b ON p.nif=b.nif_passatger
INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
WHERE v.destinacio IN (SELECT codi_aeroport FROM aeroport WHERE ciutat='París')
ORDER BY 1

-- (Reserves Vols) Codi de tots els vols amb bitllets amb data d'abans de l'1 de Desembre de 2013.

SELECT DISTINCT codi_vol
FROM bitllet 
WHERE data < TO_DATE('01/12/2013','DD/MM/YYYY')
ORDER BY 1

-- Nom, adreça i pais dels passatgers que han facturat alguna maleta de color taronja

SELECT UNIQUE p.nom, p.adreça, p.pais
FROM persona p INNER JOIN bitllet b ON p.nif = b. nif_passatger
INNER JOIN maleta m ON m.nif_passatger = b.nif_passatger AND m.codi_vol = b.codi_vol AND m.data = b.data
WHERE m.color = 'taronja'
ORDER BY 1

-- (Reserves vols) NIF, nom, població i país de totes les persones existents a la base de dades.
SELECT DISTINCT nif, nom, poblacio, pais
FROM persona
ORDER BY 1,2,3,4

-- El nom i NIF de totes les persones.
SELECT p.nif, p.nom
FROM persona p
ORDER BY 1,2

-- (Reserves Vols) Localitzador(s) i nom de les persones pel vol KLM4303.
SELECT b.localitzador, p.nom
FROM bitllet b INNER JOIN persona p ON b.nif_passatger = p.nif
WHERE b.codi_vol = 'KLM4303'
ORDER BY 1,2

-- Nom i data de naixement dels nascuts abans de 1970 que han viajat al Barcelona International Airport (El Prat)
SELECT UNIQUE p.nom, TO_CHAR(p.data_naixement, 'DD/MM/YYYY') AS data_naixament
FROM bitllet b INNER JOIN persona p ON b.nif_passatger = p.nif
INNER JOIN vol v ON v.codi_vol = b.codi_vol AND v.data = b.data
WHERE p.data_naixement < TO_DATE('01/01/1970','DD/MM/YYYY') AND v.destinacio IN (SELECT codi_aeroport FROM aeroport WHERE nom = 'Barcelona International Airport (El Prat)')
ORDER BY 1,2

-- Tipus de avions que volen a l'aeroport de Orly 
SELECT UNIQUE tipus_avio
FROM vol
WHERE destinacio IN (SELECT codi_aeroport FROM aeroport WHERE ciutat='Orly') OR origen IN (SELECT codi_aeroport FROM aeroport WHERE ciutat='Orly')
ORDER BY 1

-- (Reserves Vols) Nom i Població dels passatgers que han viatjat algun cop en 1a classe.
SELECT DISTINCT p.nom, p.poblacio
FROM persona p INNER JOIN bitllet b ON p.nif = b.nif_passatger
WHERE b.classe = 1
ORDER BY 1,2;

-- (Espectacles) Nombre, apellidos y DNI de todos los espectadores diferentes que han comprado entradas para ver 'El Màgic d'Oz'.
SELECT DISTINCT es.nom, es.cognoms, es.dni 
FROM espectadors es INNER JOIN entrades en ON es.dni = en.dni_client
INNER JOIN espectacles esp ON esp.codi = en.codi_espectacle 
WHERE esp.nom LIKE '%Oz%'
ORDER BY 1,2,3

-- (Espectacles) ¿Cuántos espectáculos de tipo Ópera se hacen en Barcelona?
SELECT COUNT(*) AS espectacles_opera
FROM espectacles es INNER JOIN recintes r ON es.codi_recinte = r.codi
WHERE es.tipus = 'Opera' AND r.ciutat = 'Barcelona'
ORDER BY 1

-- (Espectacles) Nombre, apellidos y número total de representaciones a las que ha asistido cada espectador
SELECT es.nom, es.cognoms, COUNT(*) AS total_representacions
FROM espectadors es INNER JOIN entrades en ON es.dni = en.dni_client
GROUP BY es.nom, es.cognoms
ORDER BY 1,2,3


-- (Espectacles) Nombre de espectáculo, fecha y hora de representaciones de espectáculos del año 2011 con más de 5 espectadores (format hora TO_CHAR(R.HORA,'HH24:MI:SS')).
SELECT es.nom, TO_CHAR(r.data, 'DD/MM/YYYY') AS data, TO_CHAR(r.hora, 'HH24:MI:SS') AS hora
FROM espectacles es INNER JOIN representacions r ON es.codi = r.codi_espectacle
INNER JOIN entrades en ON en.codi_espectacle = r.codi_espectacle AND en.data = r.data AND en.hora = r.hora
WHERE r.data BETWEEN TO_DATE('01/01/2011', 'DD/MM/YYYY') AND TO_DATE('31/12/2011', 'DD/MM/YYYY')
GROUP BY es.nom, r.data, r.hora
HAVING COUNT(*) > 5
ORDER BY 1,2,3


-- (Espectacles) Nom del recinte/s que ha/n venut entrades amb el preu més alt. Ordenat per nom del recinte.
SELECT DISTINCT r.nom
FROM recintes r INNER JOIN preus_espectacles pe ON r.codi = pe.codi_recinte
WHERE pe.preu >= ALL(SELECT MAX(preu) FROM preus_espectacles)
ORDER BY 1

-- (Espectacles) Nombre del espectáculo donde más entradas de Platea se han vendido
SELECT DISTINCT es.nom
FROM espectacles es INNER JOIN entrades en ON es.codi = en.codi_espectacle
WHERE en.zona = 'Platea'
GROUP BY es.nom, en.data, en.hora 
HAVING COUNT(*) >= ALL(
    SELECT COUNT(*)
    FROM espectacles es INNER JOIN entrades en ON es.codi = en.codi_espectacle
    WHERE en.zona = 'Platea'
GROUP BY es.nom, en.data, en.hora )
ORDER BY 1

-- (Espectacles) Nombre y dirección del recinto donde no ha habido ninguna representación en el año 2011.
SELECT r.nom, r.adreça 
FROM recintes r
WHERE r.codi NOT IN (
    SELECT re.codi
    FROM espectacles es INNER JOIN representacions r ON es.codi = r.codi_espectacle
    INNER JOIN recintes re ON re.codi = es.codi_recinte 
    WHERE r.data BETWEEN TO_DATE('01/01/2011','DD/MM/YYYY') AND TO_DATE('31/12/2011','DD/MM/YYYY'))
ORDER BY 1,2

-- (Espectacles) Representacions del Gener del 2012 que no han venut cap entrada. Mostreu nom de l'espectacle i la data de la representació. 
(SELECT DISTINCT es.nom, TO_CHAR(r.data, 'DD/MM/YYYY') AS data FROM espectacles es INNER JOIN representacions r ON es.codi = r.codi_espectacle AND r.data BETWEEN TO_DATE('01/01/2012', 'DD/MM/YYYY') AND TO_DATE('31/01/2012', 'DD/MM/YYYY'))
MINUS
(SELECT DISTINCT es.nom, TO_CHAR(r.data, 'DD/MM/YYYY') AS data FROM espectacles es INNER JOIN representacions r ON es.codi = r.codi_espectacle INNER JOIN entrades en ON en.codi_espectacle = r.codi_espectacle AND en.data = r.data AND en.hora = r.hora WHERE r.data BETWEEN TO_DATE('01/01/2012', 'DD/MM/YYYY') AND TO_DATE('31/01/2012', 'DD/MM/YYYY'))
ORDER BY 1,2

-- 99. Espectador (DNI, nom i cognoms) que ha comprat més entrades del teatre La Faràndula de Sabadell. (ESPECIFICANDO)
select EP.dni, EP.nom, EP, cognoms, COUNT (*) AS Num_entrades
from Espectadors EP, Entrades EN, Recinte R
where EP.dni = EN.dni_client AND
EN.codi_recinte = R.codi AND
EN.nom = "La Farandula" AND
EN.ciutat = "Sabadell"
group by EP.nom, EP.dni, EP.cognoms
having COUNT (*) >= ALL ( select EP2.dni, EP2.nom, EP2.cognoms
from Espectadors EP2, Recinte R2
where EN2.codi_recinte = R2.codi AND
EN2.nom = "La Farandula"
EN2.ciutat = "Sabadell"
group by EN2.dni_client )
order by EP.dni;

--92. Representació (codi d’espectacle, data i hora), intèrpret i codi del recinte amb més entrades venudes. (NO ESPECIFICANDO: VENUDAS NOMES)
select EP.codi_espectale, EP.data, TO_CHAR(EP.hora, "HH24:MI:SS") AS hora,
EN.interpret, EN.codi_recinte
from Entrades EN, Espectacles EP
where EP.codi = EN.codi_espectacles
group by EP.codi_espectale, EP.data, EN.interpret, EN.codi_recinte
having COUNT (*) >= ALL ( select COUNT (*)
from Entrades EN
group by codi_espectacles, data, hora,
codi_recinte);

--91. Codi i nom del recinte on s’han fet més representacions i nombre de representacions
select R.nom, R.codi, COUNT (*) AS Num_representacions
from Recinte R, Representacions RP, Espectacles ES
where R.codi = ES.codi AND
ES.codi = RP.codi_espectacles
group by R.nom, R.codi
having COUNT >= ALL ( select R.nom, R.codi
from Representacions RP2, Espectacles ES2
where ES2.codi = RP2.codi_espectacles
group by ES2.codi_recinte )
order by R.codi

--97. Codi, nom i población dels recintes que tenen més d'una zona.
select R.codi, R.nom, R.ciutat
from Recintes RE, Zones_Recinte ZR
where R.codi = ZR.Codi_Recinte
group by R.codi, R.nom, R.ciutat
having COUNT)(*) > 1
order by R.codi;

--98. Codi, nom i població dels recintes que només tenen una zona.
select R.codi, R.nom, R.ciutat
from Recintes RE, Zones_Recinte ZR
where R.codi = ZR.Codi_Recinte
group by R.codi, R.nom, R.ciutat
having COUNT(*) = 1
order by R.codi;

--89. Nom i codi del recinte del que s’han venut més entrades.
select R.nom, R.codi, COUNT (*) AS Num_entrades
from Recinte R, Entrades EN
where R.Codi = EN.codi_recinte
group by R.nom, R.codi
having COUNT >= ALL ( select COUNT(*)
from Entrades EN2
group by EN2.codi recinte );

--85. Espectadors (DNI, nom i cognoms) que no han vist cap espectacle de El Tricicle.
select EP.dni, EP.nom, EP.cognoms
from Espectadors EP
MINUS ( select EP.dni, EP.nom, EP.cognoms
from Espectadors EP, Espectacles ES, Entrades EN
where EP.dni = EN.dni_client AND
EN.codi_espectacles = ES.codi AND
ES.nom = "El Tricicle");

--83. DNI dels espectadors que han assistit a tots els espectacles de l’intèpret El Tricicle
select EP.dni_client
from Espectacles EP, Entrades R
where EP.codi = EN.codi_espectacles AND
EP.Interpret = "El Tricicle"
group by EP.dni_client
having COUNT (DISTINCT EP.codi) = ( select COUNT (*)
from Espectacles EP
where EP.Interpret = "El Tricicle");

-- 81. Nombre d’espectadors per espectacle en promig.
select T1.Total_Espectadors / T2.Total_Espectacle AS PROMIG
from ( select COUNT (*) AS Total_Espectadors
from Entrades) T1
(select COUNT (*) AS Total_Espectacles
from Espectacles) T2;

--79. Promig d’ocupació dels espectacles de Barcelona.
select T1.Aforo_Venut / T2.Total_Aforo AS PROMIG
from ( select COUNT (*) AS Aforo_Venut
from Recinte R, Entrades EN
where R.ciutat ="Barcelona" AND
R.codi = EN.codi_recinte) T1
( select COUNT (*) AS Total_Aforo
from Recinte R, Seients S
where R.ciutat ="Barcelona" AND
R.codi = S.codi_recinte) T2

--76. Nom de l’espectacle amb el preu de l’entrada més barat.
select DISTINCT EP.nom, PE.Preu AS Preu
from Espectacles EP, Preus_Espectacles PE
where EP.codi = PE.codis_espectacles AND
PE.Preu <= ( select MIN(PE2.Preu) --*/Comparación de PREU con la dela entrada que tenga el precio mínimo*/
from Preus_Espectacles PE2);

--75. Nom de l’espectacle amb el preu de l’espectacle més car.
select EP.nom, PE:Preu AS Preu
from Espectacles EP, Preus_Espectacles PE
where EP.codi = PE.codis_espectacles AND
PE.Preu >= ( 
  select MAX(PE2.Preu) --*/Comparación de PREU con la dela entrada que tenga el precio máximo*/ 
  from Preus_Espectacles PE2);

--74. Zona de recinte amb més capacitat i nom del recinte on es troba aquesta zona.
select R.nom, ZR.zones, ZR.capacitat AS Capacitat
from Recinte R, Zones_Recinte ZR
where R.Codi = ZR.codi_recinte AND
ZR.capacitat >= ( select MAX(ZR2.Capacitat)
from Zones_Recintes ZR2);

--72. Aforament total dels recintes de Barcelona
select SUM(ZR.capacitat) AS Total_Capacitat
from Recintes R, Zones_Recintes ZR
where R.codi = ZR.codi_recinte AND
R.Ciutat = "Barcelona";

--69. Nombre de representacions que es realitzen el 20 d’Octubre del 2011.
select Codi_Espectacle, COUNT(*) AS Num_Rep
from Representacions
where data = TO_DATE ("20/09/2011", "dd/mm/yyyy")
group by Codi_espectacle

--63. Nom de l’espectacle teatral amb més espectadors */Especifica tipo/*
select ES.nom, COUNT (*) Num_Espectadors
from Espectacles ES, Entrades EN
where ES.Codi = EN.codi_espectacles AND
ES.tipus = "Teatre"
group by ES.nom
having COUNT (*) >= ALL ( select count (*)
from Espectacles ES2, Entrades EN2
where ES2.Codi = EN2.codi_espectacles AND
ES2.tipus = "Teatre"
group by ES.nom );

--62. Nom, ciutat i capacitat del recinte amb menys capacitat. */No especifica nada/*
select R.nom, R.ciutat, SUM(ZR.Capacitat) AS Cap_Total
from Recinte R, Zones_Recinte ZR
where R.Codi = ZR.codi_recinte
group by R.nom, R.ciutat, ZR.capacitat
having SUM(ZR.Capacitat) <= ALL ( select SUM(Capacitat)
from Zones_Recintes
group by codi_recinte);

--50. Intèrprets que van realitzar algun espectacle l’any 2011, però que no n’han fet cap l’any 2012.
select DISTINCT ES.Interpret
from Espectacle ES, Representacions RE
where ES.codi = RE.codi_espectacle AND
RE.data BETWEEN TO_DATE ("01/01/2011","dd/mm/yyyy") AND
TO_DATE("31/01/2011","dd/mm/yyyy")
NOT EXIST ( select *
from Espextacles ES2, Representacions RE2
where ES2.codi = RE2.codi_espectacles AND
ES2.Interpret = ES.Interpret AND
RE2.data > TO_DATE ("01/01/2012", "dd/mm/yyyy"));

--38. Seients (zona, fila i número) que no s’han ocupat en cap de les representacions de El país de les cent paraules.
select S.zona, S.fila, S.numero
from Seient S, Espectacles ES
where S.codi_recinte = ES.codi_recinte AND
ES.nom = "El pais de la cent paraules"
NOT EXIST ( select *
from Entrades EN
where EN.codi_recinte = S.codi_recinte AND
EN.zona = S.zona AND
EN.fila = S.fila AND
EN.numero =S.numero AND
ES.codi = EN.codi_espectacle)
order by S.zona, S.fila, S.numero;

-- 39. Representacions d’espectacles (nom de l’espectacle, data i hora) en què s’han venut totes les entrades, ordenat per nom de l’espectacle i data.
select ES.nom, ES.data, TO_CHAR (EN.hora "HH24:Mi:SS") AS hora
from Espectacles ES, Entrades EN
where ES.codi = EN.codi_recinte AND
EXIST ( select *
from Seient S
where EN.zona = S.zona AND
EN.fila = S.fila AND
EN.numero = S.numero)
order by ES.nom, ES.data;

-- 30. Nom dels espectacles que s’han representat a Barcelona durant el mes de gener del 2012 ordenats per nombre d’entrades promig venudes per dia.
select ES.nom, COUNT (*) / COUNT (DISTINCT EN.data) AS MITJANA
from Espectacles ES, Recinte R, Representacions RS
where ES.codi = R.codi AND
R.ciutat = "Barcelona" AND
ES.Data_Inicial BETWEEN TO_CHAR ("01/01/2012","dd/mm/yyyy")
TO_CHAR ("31/01/2012","dd/mm/yyyy")
ES.Data_Final BETWEEN TO_CHAR ("01/01/2012","dd/mm/yyyy")
TO_CHAR ("31/01/2012","dd/mm/yyyy")
group by ES.nom
order by MITJANA

-- 27. Preu màxim i mínim per cadascun dels espectacles representats al Liceu. A més dels preus s’ha de recuperar també el nom de l’espectacle i les dates inicial i final, ordenat per data d’inici.
select ES.nom, ES.data_inicial, ES.data_final, MAX (PE.preu) AS Preu_Max,
MIN(PE.preu) AS Preu_Min,
from Recintes R, Espectacles ES, Preu_Espectacle PE
where R.nom = "Liceu" AND
ES.codi_recinte = R.codi AND
PE.codi_espectacle = ES.codi AND
PE.codi_recinte = ES.codi_recinte
group by ES.nom, ES.Data_inicial, ES. Data_final
order by ES.Data_Inicial;

--26. Data, hora i nombre d’entrades venudes de cadascuna de les representacions
de El Màgic d’Oz, ordenat per nombre d’entrades venudes.
select EN.data, TO_CHAR(EN.hora,'HH24:MI:SS'),COUNT (*) AS Num_Entrades
from Espectacles ES, Entrades EN
where ES.nom = "El Mago d' Oz" AND
EN.codi_espectacles = ES.codi
group by EN.data, EN.hora
order by Num_Entrades

--2. Codi dels espectacles, dia de la representació i DNI dels clients de les entrades reservades pel mes de març del 2012 en algun dels recintes amb codi 
--101, 103 o 105, ordenats per codi d’espectacle i dia de representació. 
--Si un client té reservada més d’una entrada pel mateix expectacle només ha de sortir un cop.
select DISTINCT EN.codi_espectacle, EN.data, EN.dni
from Entrades EN
where EN.data BETWEEN TO_DATE (‘01/03/2012’,’dd/mm/yyyy’) AND
TO_DATE(‘31/03/2012’,’dd/mm/yyyy’) AND
EN.Codi_Recinte IN (101, 103, 105)
order by EN.codi_espectacle, EN.data;


--1.Zonas de recintos de fuera de Barcelona con más de 15 asientos. Mostrar el nombre del recinto, el nombre de la zona y la capacidad.
SELECT R.Nom, ZR.Zona, ZR.Capacitat
FROM Recintes R, Zones_Recinte ZR
WHERE R.Codi = ZR.Codi_Recinte AND
ZR.Capacitat > 15
ORDER BY 1,2,3

--2.Número de entradas vendidas para el espectáculo "La Ventafocs".
SELECT COUNT(*) as Total
FROM Espectacles E, Entrades EN
WHERE E.Nom = "La Ventafocs" AND
EN.Codi_Espectacle = E.Codi
ORDER BY 1

--3.Nombre de los recintos en los que la media (AVG) de los precios de los espectáculos que se representan es superior a 10.
SELECT R.Nom
FROM Recintes R, Preus_Espectacles PE
WHERE R.Codi = PE.Codi_Recinte
GROUP BY R.Nom
HAVING AVG(PE.Preu)>10
ORDER BY 1

--4.Nombre de los espectáculos que han vendido más de 30 entradas.
SELECT E.Nom
FROM Espectacles E, Entrades EN
WHERE E.Codi = EN.Codi_Espectacle
GROUP BY E.Nom
HAVING COUNT (*) > 30
ORDER BY 1

--5.Nombre de los recintos que tienen zonas con las entradas más caras. También la zona y el precio.
SELECT DISTINCT R.Nom, PE.Zona, PE.Preu
FROM Preus_Espectacles PE, Recintes R
WHERE PE.Codi_Recinte = R.Codi AND
PE.Preu = ( SELECT max(PE2.Preu) FROM Preus_Espectacles PE2 )
ORDER BY 1,2,3

--6.Nombre y apellidos del espectador(s) que ha(n) comprado mas entradas para el espectáculo "Mar i Cel". También el total de entradas vendidas para ese espectador
SELECT ES.Nom, ES.Cognoms, COUNT(*)
FROM Espectadors ES, Entrades EN, Espectacles E
WHERE E.Nom = 'Mar i Cel' AND
E.Codi = EN.Codi_Espectacle AND
ES.DNI = EN.DNI_Client
GROUP BY ES.Nom, ES.Cognoms
HAVING COUNT(*) >=
ALL(SELECT COUNT(*)
FROM Espectacles E2, Entrades EN2
WHERE E2.Nom = 'Mar i Cel' AND
E2.Codi = EN2.Codi_Espectacle
GROUP BY EN2.DNI_Client)
ORDER BY 1,2,3

--7.Nombre y fecha de las representaciones de espectáculos de Febrero de 2012 para los que no se ha vendido ninguna entrada.
SELECT E.Nom, R.Data
FROM Espectacles E, Representacions R
WHERE E.Codi = R.Codi_Espectacle AND
R.Data BETWEEN TO_DATE('01/02/2012','dd/mm/yyyy') AND TO_DATE('29/02/2012','dd/mm/yyyy')
MINUS
(SELECT E2.Nom, R2.Data
FROM Espectacles E2, Representacions R2, Entrades EN2
WHERE E2.Codi = R2.Codi_Espectacle AND
E2.Codi = EN2.Codi_Espectacle AND
R2.Data = EN2.Data AND
R2.Hora = EN2.Hora AND
R2.Data BETWEEN TO_DATE('01/02/2012','dd/mm/yyyy') AND TO_DATE('29/02/2012','dd/mm/yyyy')
GROUP BY E2.Nom, R2.Data
HAVING COUNT(*) >= 1)
ORDER BY 1,2

---------------------------------------------------------------------------------------------------------------
--1.Per cada espectacle fet a la ciutat de Barcelona, mostreu el nom d'espectacle i nombre de representacions ordenat per nom d'espectacle.
SELECT E.Nom, TotalEspectacle
FROM Espectacles E, Recintes R, Representacions RP
WHERE E.Codi_Recinte = R.Codi AND
R.Ciutat = 'Barcelona' AND
RP.Codi_Espectacle = E.Codi
GROUP BY E.Nom
HAVING COUNT (*) as TotalEspectacle
ORDER BY 1,2

--2. Nom i direcció dels teatres de Barcelona.
SELECT R.Nom, R.Adreça
FROM Recintes R
WHERE R.Ciutat = 'Barcelona'
ORDER BY 1,2


--3. Representacions en el teatre Romea on s'han venut més de 5 entrades. Mostrar nom dels espectacles data i hora de la representació, així com el número d'entrades venudes. Ordena per nom de l'espectacle i data i hora de la representació
SELECT E.Nom, EN.Data, EN.Hora, COUNT(*)
FROM Recintes R, Espectacles E, Entrades EN
WHERE R.Nom = 'Romea' AND
E.Codi_Recinte = R.Codi AND
EN.Codi_Espectacle = E.Codi
GROUP BY E.Nom, EN.Data, EN.Hora
HAVING COUNT(*) > 5
ORDER BY 1,2,3

--4.Mostra els nom dels recintes que tinguin preus entrades superiors a 20 eur. Mostra també el preu de l'entrada.
SELECT DISTINCT R.Nom, PE.Preu
FROM Preus_Espectacles PE, Recintes R
WHERE PE.Codi_Recinte = R.Codi AND
PE.Preu > 20
ORDER BY 1


--5.Nom del recinte que té la zona amb capacitat més gran.
SELECT DISTINCT R.Nom
FROM Recintes R, Zones_Recinte ZR
WHERE ZR.Codi_Recinte = R.Codi AND
ZR.Capacitat = (SELECT MAX(ZR2.Capacitat) FROM Zones_Recinte ZR2)
ORDER BY 1

--5 MODIFICADA. Nom del recinte amb la capacitat més gran.
SELECT R.Nom
FROM Recintes R, Zones_Recinte ZR
WHERE ZR.Codi_Recinte = R.Codi
GROUP BY R.Nom
HAVING SUM(ZR.Capacitat) >=
(SELECT MAX(SUM(ZR2.Capacitat))
FROM Zones_Recinte ZR2
GROUP BY ZR2.Codi_Recinte)
ORDER BY 1;

--6. Nom de l'espectacle amb més espectadors.
SELECT E.Nom
FROM Espectacles E, Entrades EN
WHERE E.Codi = EN.Codi_Espectacle
GROUP BY E.Nom
HAVING COUNT(*) =
(SELECT MAX(COUNT(*))
FROM Espectacles E2, Entrades EN2
WHERE E2.Codi = EN2.Codi_Espectacle
GROUP BY E2.Nom)
ORDER BY 1


--7. Seients lliures de platea del Liceu per la representació del dia 6 d'abril de 2012 a les 22. Mostrar fila i número.
(SELECT S.Fila, S.Numero
FROM Seients S, Recintes R
WHERE R.Nom = 'Liceu' AND
R.Codi = S.Codi_Recinte AND
S.Zona = 'Platea')
MINUS
(SELECT EN2.Fila, EN2.Numero
FROM Seients S2, Recintes R2, Entrades EN2
WHERE R2.Nom = 'Liceu' AND
S2.Zona = 'Platea' AND
R2.Codi = S2.Codi_Recinte AND
S2.Codi_Recinte = EN2.Codi_Recinte AND
S2.Zona = EN2.Zona AND
EN2.Data = TO_DATE('06/04/2012','dd/mm/yyyy') AND
TO_CHAR(EN2.hora,'HH24:MI:SS') = '22:00:00')
ORDER BY 1,2

--8. Nom, cogonoms i dni dels espectadors que mai han ocupat seient a platea.
(SELECT ES.Nom, ES.Cognoms, ES.DNI
FROM Espectadors ES)
MINUS
(SELECT ES2.Nom, ES2.Cognoms, ES2.DNI
FROM Espectadors ES2, Entrades EN2
WHERE ES2.DNI = EN2.DNI_Client AND
EN2.Zona = 'Platea')
ORDER BY 1,2,3
---------------------------------------------------------------------------------------------------------------

--1.Nombre, apellidos y DNI de todos los espectadores diferentes que han comprado entradas para ver 'El Màgic d'Oz'.
SELECT DISTINCT ES.Nom, ES.Cognoms, ES.DNI
FROM Espectacles E, Entrades EN, Espectadors ES
WHERE E.Nom = 'El Màgic d''Oz' AND
E.Codi = EN.Codi_Espectacle AND
EN.DNI_Client = ES.DNI
ORDER BY 1,2,3

--2.Nombre, apellidos y número total de representaciones a las que ha asistido cada espectador.
SELECT ES.Nom, ES.Cognoms, COUNT(*)
FROM Espectadors ES, Entrades EN
WHERE EN.DNI_Client = ES.DNI
GROUP BY ES.Nom, ES.Cognoms
ORDER BY 1,2,3

--3.Nombre de espectáculo, fecha y hora de representaciones de espectáculos del año 2011 con más de 5 espectadores (format hora TO_CHAR(R.HORA,'HH24:MI:SS')).
SELECT E.Nom, R.Data, TO_CHAR(R.Hora, 'HH24:MI:SS')
FROM Espectacles E, Representacions R, Entrades EN
WHERE R.DATA BETWEEN TO_DATE('01/01/2011','dd/mm/yyyy') AND TO_DATE('31/12/2011','dd/mm/yyyy') AND
E.Codi = EN.Codi_Espectacle AND
E.Codi = R.Codi_Espectacle
GROUP BY E.Nom, R.Data, R.Hora
HAVING COUNT(*) > 5
ORDER BY 1,2,3

--4.Muestra los ciudades en las que reside más de un espectador.
SELECT ES.Ciutat
FROM Espectadors ES
GROUP BY ES.Ciutat
HAVING COUNT (*) > 1
ORDER BY 1

--5.Nombre de los recintos que tienen zonas con el precio de entrada más alto.
SELECT R.Nom
FROM Recintes R, Preus_Espectacles PE
WHERE R.Codi = PE.Codi_Recinte AND
PE.Preu = (SELECT MAX(PE2.Preu) FROM Preus_Espectacles PE2)
ORDER BY 1

--6.Nombre del espectáculo donde más entradas de Platea se han vendido.
SELECT E.Nom
FROM Espectacles E, Entrades EN
WHERE E.Codi = EN.Codi_Espectacle AND
EN.Zona = 'Platea'
GROUP BY E.Nom
HAVING COUNT (*) >=
(SELECT MAX( COUNT(*))
FROM Entrades EN2
WHERE EN2.Zona = 'Platea'
GROUP BY EN2.Codi_Espectacle)
ORDER BY 1

--7.Nombre y dirección del recinto donde no ha habido ninguna representación en el año 2011.
(SELECT R.Nom, R.Adreça
FROM Recintes R)
MINUS (SELECT R2.Nom, R2.Adreça
FROM Recintes R2, Representacions RE2, Espectacles E2
WHERE R2.Codi = E2.Codi_Recinte AND
E2.Codi = RE2.Codi_Espectacle AND
RE2.Data BETWEEN TO_DATE('01/01/2011','dd/mm/yyyy') AND TO_DATE('31/12/2011','dd/mm/yyyy'))
ORDER BY 1,2

--8.Nombre y fecha de las representaciones de Enero de 2012 donde no se ha vendido ninguna entrada.
(SELECT E.Nom, RE.Data
FROM Espectacles E, Representacions RE
WHERE RE.Data BETWEEN TO_DATE('01/01/2012','dd/mm/yyyy') AND TO_DATE('31/01/2012','dd/mm/yyyy') AND
E.Codi = RE.Codi_Espectacle)
MINUS (SELECT E2.Nom, RE2.Data
FROM Espectacles E2, Representacions RE2, Entrades EN2
WHERE E2.Codi = RE2.Codi_Espectacle AND
E2.Codi = EN2.Codi_Espectacle AND
RE2.Data = EN2.Data AND
RE2.Hora = EN2.Hora AND
EN2.Data BETWEEN TO_DATE('01/01/2011','dd/mm/yyyy') AND TO_DATE('31/01/2012','dd/mm/yyyy')
GROUP BY E2.Nom, RE2.Data)
ORDER BY 1,2

Consultes BBDD Pràctiques Examen
--########################### 09:00 - 11:00#####################################
--1. Quantes maletes ha facturat en Aitor Tilla al vol amb destinació Pekin de la data 1/10/2013?
SELECT COUNT(M.CODI_MALETA)
FROM MALETA M, VOL V, PERSONA P, AEROPORT A
WHERE M.NIF_PASSATGER = P.NIF
AND M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND P.NOM = 'Aitor Tilla'
AND A.CIUTAT = 'Pekin'
AND V.DATA = TO_DATE('01/10/2013', 'DD/MM/YYYY')
ORDER BY 1;
--2. Codi_Vol i data dels vols amb origen a algún aeroport rus
SELECT V.CODI_VOL, TO_DATE(V.DATA, 'DD/MM/YYYY') AS DATA
FROM VOL V
WHERE ORIGEN IN (SELECT CODI_AEROPORT
FROM AEROPORT A
WHERE PAIS = 'Russia')
ORDER BY 1;
--3. Portes d'embarcament (terminal, area, porta) de l'aeroport JFK on no han sortit
--mai cap vol amb destinació França
(SELECT PE.TERMINAL, PE.AREA, PE.PORTA
FROM PORTA_EMBARCAMENT PE
WHERE PE.CODI_AEROPORT = 'JFK')
MINUS
(SELECT PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, AEROPORT A, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
AND V.TERMINAL = PE.TERMINAL
AND V.AREA = PE.AREA
AND V.PORTA = PE.PORTA
AND PE.CODI_AEROPORT ='JFK'
AND A.PAIS = 'França'
AND V.DESTINACIO = A.CODI_AEROPORT)
ORDER BY 1,2,3;
--4. Per les reserves de vols amb destinació el Marroc, aquelles on només viatja un únic passatger. Mostreu localitzador i nom del passatger que hi viatja.
SELECT B. LOCALITZADOR, P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL IN (SELECT CODI_VOL
FROM BITLLET B
WHERE CODI_VOL IN (SELECT V.CODI_VOL
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'Marroc')
GROUP BY CODI_VOL
HAVING COUNT(*) = 1)
ORDER BY 1;
--5. Companyies que tenen, almenys, els mateixos tipus d'avions que Vueling airlines (El resultat també ha de tornar Vueling Airlines)
SELECT DISTINCT COMPANYIA
FROM VOL
WHERE TIPUS_AVIO IN (SELECT DISTINCT TIPUS_AVIO
FROM VOL
WHERE COMPANYIA = 'VUELING AIRLINES')
ORDER BY 1;
--6. Nom de la companyia que fa més vols amb origen Italia
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(*) = (SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1;
--7. Per a cada aeroport i terminal, nom de l'aeroport, terminal i nombre d'arees.
SELECT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA) AS AREAS
FROM PORTA_EMBARCAMENT PE, AEROPORT A
WHERE PE.CODI_AEROPORT = A.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1,2;
--8. Promitg de vols per a cada companyia. Aclariment: el numerador es el nombre de vols totals i el denominador el nombre total de companyies
SELECT N.VOLS/D.COMPANYIES AS PROMITG
FROM (SELECT COUNT(*) AS VOLS FROM VOL) N,
(SELECT COUNT(DISTINCT COMPANYIA) AS COMPANYIES FROM VOL) D
ORDER BY 1;
--########################### 09:00 - 11:00#####################################
--########################### 11:00 - 13:00#####################################
--1 (Aerolinies) Promig d'espanyols als vols amb destinació Munich.
SELECT E.ESPANYOLS/T.TOTAL
FROM (SELECT COUNT(*) AS ESPANYOLS
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Munich'
AND P.PAIS = 'Espanya') E, (SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V, AEROPORT A
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Munich') T
ORDER BY 1;
--2 (Aerolinies) Nom, correu i companyia aeria dels clients (persones)
--que han fet reserves de 2 bitllets d’anada i 2 bitlles de tornada
(SELECT P.NOM, P.MAIL, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.ANADA_TORNADA = 'ANADA'
GROUP BY P.NOM, P.MAIL, V.COMPANYIA
HAVING COUNT(B.ANADA_TORNADA) = 2)
INTERSECT
(SELECT P.NOM, P.MAIL, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.ANADA_TORNADA = 'TORNADA'
GROUP BY P.NOM, P.MAIL, V.COMPANYIA
HAVING COUNT(B.ANADA_TORNADA) = 2)
ORDER BY 1,2,3;
--3 (Aerolinies) Pes facturat als vols de 'AIR FRANCE' del 2013.
SELECT SUM(PES) AS PES
FROM MALETA M, VOL V
WHERE M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.COMPANYIA = 'AIR FRANCE'
AND TO_CHAR(V.DATA, 'YYYY') = '2013'
ORDER BY 1;
--4 (Aerolinies) Nacionalitat dels passatgers del vols amb destinació Paris Orly Airport.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.NOM = 'Paris Orly Airport'
ORDER BY 1;
--5 (Aerolinies) En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a ús dels passatgers?
SELECT FILA
FROM SEIENT
WHERE CODI_VOL = 'AEA2159'
AND TO_CHAR(DATA, 'DD/MM/YYYY') = '01/08/2013'
AND LLETRA = 'A'
ORDER BY 1;
--6 (Aerolinies) Número de terminals dels aeroports francesos (retorneu el nom de l’aeroport i la quantitat de terminals).
SELECT A.NOM, COUNT(DISTINCT PE.TERMINAL) AS TERMINALS
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY A.NOM
ORDER BY 1;
--7 (Aerolinies) Nom de la persona que més cops ha pagat amb Paypal. Retorneu també el
--número de cops que ho ha fet.
SELECT P.NOM, COUNT(*) AS COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
and R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) > ANY (SELECT COUNT(*)
FROM RESERVA
WHERE MODE_PAGAMENT = 'Paypal'
GROUP BY NIF_CLIENT)
ORDER BY 1;
--8 (Aerolinies) Destinació dels vols que tenen entre els seus passatgers persones de
--totes les ciutats de Turquia.
SELECT DISTINCT V.DESTINACIO
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.NIF_PASSATGER IN (SELECT NIF
FROM PERSONA
WHERE PAIS = 'Turquia')
ORDER BY 1;
--########################### 11:00 - 13:00#####################################
--########################### 15:00 - 17:00#####################################
--1 Pais, codi_aeroport, terminal i àrea dels aeroport amb 3 portes d'embarcament i el nom del pais comenci per 'E'
SELECT DISTINCT A.PAIS, PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
AND A.PAIS LIKE 'E%'
AND PE.CODI_AEROPORT IN (SELECT PE.CODI_AEROPORT
FROM PORTA_EMBARCAMENT PE
GROUP BY PE.CODI_AEROPORT
HAVING COUNT(DISTINCT PORTA) = 3)
ORDER BY 1, 2 ,3, 4;
--2 Mostreu el numero de passatgers que no han escollit els seus seients
SELECT COUNT(*)
FROM BITLLET
WHERE FILA IS NULL
AND LLETRA IS NULL
ORDER BY 1;
--3 Nombre de passatgers agrupats segons pais de destinació (en el vol). Mostreu pais i nombre de passatgers
SELECT A.PAIS, COUNT(B.NIF_PASSATGER) AS NOMBRE_PASSATGERS
FROM VOL V, AEROPORT A, BITLLET B
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY A.PAIS
ORDER BY 1;
--4 Nom dels clients que sempre paguem amb Visa Electron
SELECT P.NOM
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Visa Electron'
GROUP BY P.NOM
HAVING (P.NOM, COUNT(*)) IN (SELECT P.NOM, COUNT(*)
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
GROUP BY P.NOM)
ORDER BY 1;
--5 Ciutats espanyoles amb aeroports
SELECT CIUTAT
FROM AEROPORT
WHERE PAIS = 'Espanya';
--6 Noms de les companyies que volen als mateixos aeroports on ha volat Estela Gartija.
SELECT DISTINCT COMPANYIA
FROM VOL V
WHERE DESTINACIO IN (SELECT V.DESTINACIO
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.NOM = 'Estela Gartija')
ORDER BY 1;
--7 Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers dels vols que tenen el nombre de passatges més petit.
SELECT V.ORIGEN, V.CODI_VOL, V.DATA, COUNT(B.NIF_PASSATGER)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(B.NIF_PASSATGER) = (SELECT MIN(COUNT (*))
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1,2,3,4;
--8 Percentatge d'ús de l'aeroport de Barcelona. Indicació: el numerador és el nombre de vols ambos origen, o destinació, Barcelona; el denominador el nombre total de vols.
SELECT T1.BARCELONA/T2.TOTAL
FROM (SELECT COUNT(*) AS BARCELONA
FROM VOL
WHERE ORIGEN = (SELECT CODI_AEROPORT
FROM AEROPORT
WHERE CIUTAT = 'Barcelona')
OR DESTINACIO = (SELECT CODI_AEROPORT
FROM AEROPORT
WHERE CIUTAT = 'Barcelona')) T1, (SELECT COUNT(*) AS TOTAL
FROM VOL) T2
ORDER BY 1;
--########################### 15:00 - 17:00#####################################

--Nom i Població dels passatgers que han viatjat algun cop en 1a classe.
SELECT P.NOM, P.POBLACIO
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = 1
GROUP BY P.NOM, P.POBLACIO
ORDER BY 1

--Per cada aeroport llisti el nom de l'aeroport, la terminal, l'àrea i el nombre de portes d'aquesta àrea.
SELECT A.NOM, PE.TERMINAL, PE.AREA, COUNT(DISTINCT PE.PORTA) AS PORTES
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL, PE.AREA
ORDER BY 1,2,3,4

--Nombre màxim de maletes (com max_maletes) que han estat despatxades en el mateix dia per un mateix passatger.
SELECT MAX(COUNT(*)) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF, B.DATA
ORDER BY 1;
SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M, BITLLET B
WHERE M.CODI_VOL = B.CODI_VOL
AND M.DATA = B.DATA
AND M.NIF_PASSATGER = P.NIF
AND B.NIF_PASSATGER = M.NIF_PASSATGER
GROUP BY P.NIF)
ORDER BY 1;

--Del total de maletes facturades, quin percentatge són de color vermell. (NOTA: el percentatge és sempre un numero entre 0 i 100)
SELECT T1.M_VERMELLES/T2.M_TOTAL AS PERCENTATGE
FROM(SELECT COUNT(*) AS M_VERMELLES
FROM MALETA M
WHERE M.COLOR = 'vermell') T1, (SELECT COUNT(*) AS M_TOTAL
FROM MALETA M) T2
ORDER BY 1;

-- Nombre de vegades que apareix la nacionalitat més freqüent entre les persones de la BD. Atributs: nacionalitat i nombre de vegades com a n_vegades.
SELECT P.PAIS, COUNT(*) AS N_NACIONALITATS
FROM PERSONA P
GROUP BY P.PAIS
HAVING COUNT(*)>= ALL(SELECT COUNT(*)
FROM PERSONA P
GROUP BY P.PAIS)
ORDER BY 1,2;

--Quan pes ha facturat l'Alberto Carlos Huevos al vol amb destinació Rotterdam de la data 02 de juny del 2013?
SELECT M.PES
FROM MALETA M, PERSONA P, VOL V, AEROPORT A
WHERE P.NOM = 'Alberto Carlos Huevos'
AND P.NIF = M.NIF_PASSATGER
AND M.CODI_VOL = V.CODI_VOL
AND V.DATA = TO_DATE('02/06/2013', 'DD/MM/YYYY')
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Rotterdam'
ORDER BY 1;
--Seients disponibles (fila i lletra) pel vol AEA2195 amb data 31/07/2013
SELECT S.FILA, S.LLETRA
FROM SEIENT S
WHERE S.CODI_VOL = 'AEA2195'
AND S.EMBARCAT != 'SI'
AND S.DATA = TO_DATE('31/07/2013','DD/MM/YYYY')
GROUP BY S.FILA, S.LLETRA
ORDER BY 1;

--Nom dels passatgers que tenen alguna maleta que pesa més de 10 kilos. Especifica també en quin vol.
SELECT P.NOM, M.CODI_VOL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND M.PES > 10
GROUP BY P.NOM, M.CODI_VOL
ORDER BY 1,2;

--Nombre de Bitllets per compradors. Es demana nom del comprador i nombre de bitllets comprats.
SELECT P.NOM, COUNT(B.NIF_PASSATGER) AS N_BITLLETS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
ORDER BY 1;

--Nom(es) de la(es) persona(es) que més cops ha(n) pagat amb Paypal. Retorneu també el número de cops que ho ha(n) fet.
SELECT P.NOM, COUNT(*) AS COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM RESERVA
WHERE MODE_PAGAMENT = 'Paypal'
GROUP BY NIF_CLIENT)
ORDER BY 1;

--Tots els models diferents d'avió que existeixen

SELECT DISTINCT TIPUS_AVIO
FROM VOL
ORDER BY 1;
--Nom i telèfon de tots les persones portugueses.
SELECT P.NOM, P.TELEFON
FROM PERSONA P
WHERE P.PAIS = 'Portugal'
ORDER BY 1,2;
--Companyies que tenen avions 'Airbus A380' de tots els tipus. Nota: per restringir
--els tipus d'avio a 'Airbus A380' useu l'operador "like": tipus_avio like 'Airbus A380%'.
SELECT DISTINCT V.COMPANYIA
FROM VOL V
WHERE V.TIPUS_AVIO LIKE 'Airbus A380%'
GROUP BY V.COMPANYIA
ORDER BY 1;
--NIF i nom del(s) passatger(s) que ha(n) facturat menys pes en el total dels seus vols.
SELECT P.NIF, P.NOM
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND M.PES <= ALL (SELECT PES
FROM MALETA)
GROUP BY P.NIF, P.NOM
ORDER BY 1,2;
--Nombre de vegades que apareix la nacionalitat més freqüent entre les persones de la BD.
--Es demana la nacionalitat i nombre de vegades que apareix.
SELECT P.PAIS, COUNT(*) AS N_VEGADES
FROM PERSONA P
GROUP BY P.PAIS
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_VEGADES
FROM PERSONA P
GROUP BY P.PAIS)
ORDER BY 1,2;
--Total de bitllets venuts per la companyia IBERIA.
SELECT COUNT(*) AS N_BITLLETS
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA = 'IBERIA'
ORDER BY 1;
--Nombre de bitllets per compradors. Es demana nom del comprador i nombre de bitllets comprats.
SELECT P.NOM, COUNT(*) AS N_BITLLETS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
GROUP BY P.NOM
ORDER BY 1,2;
--Nom dels clients que sempre paguem amb Visa Electron i dels que sempre paguem amb Paypal.
SELECT P.NOM
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND (R.MODE_PAGAMENT = 'Visa Electron' OR R.MODE_PAGAMENT = 'Paypal')
GROUP BY P.NOM
ORDER BY 1;
--Data i codi de vol dels vols amb més de 3 passatgers. ESTA MALAMENT
SELECT V.DATA, V.CODI_VOL
FROM VOL V
WHERE N_PASSATGERS = (SELECT COUNT(*) AS N
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA)
GROUP BY V.DATA, V.CODI_VOL
HAVING N_PASSATGERS > 3
ORDER BY 1,2;
--El pes de les maletes (com a pes_total) dels passatgers italians.
SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M, PERSONA P
WHERE M.NIF_PASSATGER = P.NIF
AND P.PAIS = 'Italia'
ORDER BY 1;
--Nombre de passatgers (com a N_Passatgers) amb cognom Blanco té el vol
--amb codi IBE2119 i data 30/10/13.
SELECT COUNT(*) AS N_PASSATGERS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = 'IBE2119'
AND P.NOM LIKE '%Blanco%'
AND B.DATA = TO_DATE('30/10/2013','DD/MM/YYYY')
ORDER BY 1;
--Codi, data i pes total dels vols que han facturat, en total, un pes igual o superior a 26 Kgs.
SELECT M.CODI_VOL, M.DATA, SUM(M.PES) AS PES_TOTAL
FROM MALETA M
GROUP BY M.CODI_VOL, M.DATA
HAVING SUM(M.PES) > 26
ORDER BY 1,2,3;
--Quantes maletes ha facturat en Aitor Tilla al vol amb destinació Pekin de la data 1/10/2013
SELECT COUNT(*) AS N_MALETES
FROM MALETA M, PERSONA P, VOL V, AEROPORT A
WHERE M.NIF_PASSATGER = P.NIF
AND P.NOM = 'Aitor Tilla'
AND M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Pekin'
AND V.DATA = TO_DATE('01/10/2013','DD/MM/YYYY')
ORDER BY 1;
--Nom dels passatgers que han facturat més d'una maleta vermella.
SELECT P.NOM
FROM BITLLET B, PERSONA P, MALETA M
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell'
GROUP BY P.NOM
HAVING COUNT(*) > 1
ORDER BY 1;
--Número de porta amb més d'un vol assignat. Mostrar codi de l'aeroport al que pertany,
--terminal, area i porta.
SELECT P.CODI_AEROPORT, P.TERMINAL, P.AREA, P.PORTA
FROM PORTA_EMBARCAMENT P, VOL V
WHERE V.ORIGEN = P.CODI_AEROPORT
AND V.TERMINAL = P.TERMINAL
AND V.AREA = P.AREA
AND V.PORTA = P.PORTA
GROUP BY P.CODI_AEROPORT, P.TERMINAL, P.AREA, P.PORTA
HAVING COUNT(V.CODI_VOL) > 1
ORDER BY 1,2,3,4;
--Nom de la companyia i nombre de vols que ha fet as N_VOLS.
SELECT V.COMPANYIA, COUNT(*) AS N_VOLS
FROM VOL V
GROUP BY V.COMPANYIA
ORDER BY 1,2;
--Nombre de seients reservades (com a N_Seients_Reservades) al vol de VUELING AIRLINES VLG2117
--del 27 de Agosto de 2013.
SELECT COUNT(R.LOCALITZADOR) AS N_SEIENTS_RESERVADES
FROM RESERVA R, BITLLET B, VOL V
WHERE R.LOCALITZADOR = B.LOCALITZADOR
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA = 'VUELING AIRLINES'
AND V.CODI_VOL = 'VLG2117'
AND V.DATA = TO_DATE('27/08/2013','DD/MM/YYYY')
ORDER BY 1;
--Tipus d'avió(ns) que té(nen) més passatgers no espanyols.
SI FUNCIONA:
SELECT DISTINCT V.TIPUS_AVIO
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS != 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL(SELECT COUNT(*) AS N_PASS
FROM BITLLET B, VOL V, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS != 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;
SI FUNCIONA:
SELECT DISTINCT V.TIPUS_AVIO
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL((SELECT COUNT(*) AS N_PASS
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.TIPUS_AVIO)
MINUS
(SELECT COUNT(*) AS N_ESP
FROM VOL V, BITLLET B, PERSONA P
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO))
ORDER BY 1;
--NIF i nom de la(/es) persona(/es) que ha(n) reservat més bitllets.
SELECT P.NIF, P.NOM
FROM PERSONA P, RESERVA R, BITLLET B
WHERE P.NIF = R.NIF_CLIENT
AND B.LOCALITZADOR = R.LOCALITZADOR
GROUP BY P.NIF, P.NOM
HAVING COUNT(*) >= ALL (SELECT COUNT (*) AS N_RESERVES
FROM PERSONA P, RESERVA R, BITLLET B
WHERE P.NIF = R.NIF_CLIENT
AND B.LOCALITZADOR = R.LOCALITZADOR
GROUP BY P.NIF)
ORDER BY 1,2;
--Nom(es) de la(es) persona(es) que més cops ha(n) pagat amb Paypal. PER QUE NO CAL BITLLET B?
--Retorneu també el número de cops que ho ha(n) fet.
SELECT P.NOM, COUNT(*) AS N_COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM)
ORDER BY 1;
--Tipus d'avió(ns) que fa(n) mes vols amb origen Espanya.
SELECT V.TIPUS_AVIO
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Espanya'
GROUP BY V.TIPUS_AVIO)
ORDER BY 1;
--Tipus d'avió(ns) amb més files. PER QUE DISTINCT?
SELECT V.TIPUS_AVIO
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.TIPUS_AVIO
HAVING COUNT(DISTINCT S.FILA) >= ALL(SELECT COUNT(DISTINCT S.FILA)
FROM SEIENT S
GROUP BY S.CODI_VOL)
ORDER BY 1;
--Nom(es) de la companyia(es) que fa(n) més vols amb origen Italia.
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(V.CODI_VOL) >= ALL(SELECT COUNT(V.CODI_VOL)
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1;
--ciutat de França amb més arribades
SELECT A.CIUTAT
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY A.CIUTAT
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY V.CODI_VOL)
ORDER BY 1;
--NIF, nom del(s) passatger(s) i pes total que ha(n) facturat
--menys pes en el conjunt de tots els seus bitllets.
SELECT P.NIF, P.NOM, SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM
HAVING SUM(M.PES) <= ALL(SELECT SUM(M.PES) AS PES_TOTAL
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY M.NIF_PASSATGER)
ORDER BY 1,2;
--Nom de totes les persones que son del mateix país que "Domingo Diaz Festivo" (ell inclòs).
SELECT P.NOM
FROM PERSONA P
WHERE P.PAIS = (SELECT P.PAIS
FROM PERSONA P
WHERE P.NOM = 'Domingo Diaz Festivo')
GROUP BY P.NOM
ORDER BY 1;
--Color(s) de la(/es) maleta(/es) facturada(/es) més pesada(/es).
SELECT M.COLOR
FROM MALETA M
WHERE M.PES = (SELECT MAX(M.PES)
FROM MALETA M)
ORDER BY 1;
--Dates de vol (DD/MM/YYYY) del(s) avió(ns) amb mes capacitat (major nombre de seients)
NO FUNCIONA:
SELECT V.CODI_VOL
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL
HAVING COUNT(S.FILA) >= ALL(SELECT MAX(COUNT(S.FILA)) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1;
NO FUNCIONA:
SELECT V.CODI_VOL, TO_CHAR(V.DATA, 'DD/MM/YYYY') AS DATA
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL, TO_CHAR(V.DATA, 'DD/MM/YYYY')
HAVING COUNT(S.FILA) >= ALL(SELECT MAX(COUNT(S.FILA)) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL= S.CODI_VOL AND V.DATA=S.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1;
--Nombre de seients reservats al vol ROYAL AIR MAROC RAM964 02/06/2014
SELECT COUNT(*) AS N_RESERVES
FROM SEIENT S, BITLLET B
WHERE S.CODI_VOL = 'RAM964'
AND S.DATA = TO_DATE('02/07/2014','DD/MM/YYYY')
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
ORDER BY 1;
--Numero de places lliures pel vol AEA2195 amb data 31/07/13
SELECT COUNT(*) AS N_LLIURES
FROM SEIENT S
WHERE S.CODI_VOL = 'AEA2195'
AND S.DATA = TO_DATE('31/07/2013', 'DD/MM/YYYY')
AND (S.FILA, S.LLETRA) NOT IN (SELECT S.FILA, S.LLETRA
FROM SEIENT S, BITLLET B
WHERE S.CODI_VOL = 'AEA2195'
AND S.DATA = TO_DATE('31/07/2013', 'DD/MM/YYYY')
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
AND B.FILA = S.FILA
AND B.LLETRA = S.LLETRA)
ORDER BY 1;
--Aeroports que no tenen vol en los tres primeros meses de 2014
SELECT A.CODI_AEROPORT
FROM AEROPORT A, VOL V
WHERE V.ORIGEN = A.CODI_AEROPORT
AND V.CODI_VOL NOT IN (SELECT V.CODI_VOL
FROM VOL V
WHERE V.DATA BETWEEN TO_DATE('01/01/2014','DD/MM/YYYY') AND TO_DATE('31/3/2014','DD/MM/YYYY'))
GROUP BY A.CODI_AEROPORT
ORDER BY 1;
--Noms dels aeroports i nom de les ciutats de destinació d'aquells
--vols a les que els falta un passatger per estar plens
SELECT A.NOM, A.CIUTAT
FROM AEROPORT A, VOL V, (SELECT COUNT(*) AS TOTAL_1, B.CODI_VOL, B.DATA
FROM BITLLET B
GROUP BY B.CODI_VOL, B.DATA) T1, --SUBCONSULTA Q ENS RETORNA EL NUMERO DE BITLLETS AMB CODI I DATA
(SELECT COUNT(*) AS TOTAL_2, S.CODI_VOL, S.DATA
FROM SEIENT S
GROUP BY S.CODI_VOL, S.DATA) T2 --SUBCONSULTA Q ENS RETORNA EL NUMERO DE SEIENTS AMB CODI I DATA
WHERE T1.DATA = T2.DATA --JOINT DELS BITLLETS AMB ELS SEIENTS
AND T1.CODI_VOL = T2.CODI_VOL --JOINT DELS BITLLETS AMB ELS SEIENTS
AND T1.DATA = V.DATA --JOINT DEL BITLLET AMB EL VOL
AND T1.CODI_VOL = V.CODI_VOL --JOINT DEL BITLLET AMB EL VOL
AND V.DESTINACIO = A.CODI_AEROPORT --JOINT DEL VOL AMB EL AEROPORT
AND (T1.TOTAL_1 - T2.TOTAL_2) = 1 --CONDICIO NUM BITLLETS - NUM SEIENTS = 1 (1 LLOC LLIURE)
GROUP BY A.NOM, A.CIUTAT
ORDER BY 1, 2;
--Codi de vol i data de tots els bitllets d'anada.
SELECT CODI_VOL, TO_CHAR(DATA, 'DD/MM/YYYY') AS DATA
FROM BITLLET
WHERE ANADA_TORNADA = 'ANADA'
ORDER BY 1,2;
--Nombre de seients reservats al vol de ROYAL AIR MAROC codi RAM964 del 2 de Juny de 2014.
SELECT COUNT(*) AS N_BITLLETS
FROM RESERVA R, SEIENT S, BITLLET B
WHERE R.LOCALITZADOR = B.LOCALITZADOR
AND B.CODI_VOL = S.CODI_VOL
AND B.DATA = S.DATA
AND B.CODI_VOL = 'RAM964'
AND B.DATA = TO_DATE('02/06/2014','DD/MM/YYYY')
ORDER BY 1;
--Nom, email (mail) i adreça dels passatgers que han volat amb les mateixes companyies que
--Andres Trozado (poden haver volat amb més companyies).
SELECT P.NOM, P.MAIL, P.ADREÇA, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.COMPANYIA IN (SELECT V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.NOM = 'Andres Trozado'
GROUP BY V.COMPANYIA)
GROUP BY P.NOM, P.MAIL, P.ADREÇA, V.COMPANYIA
ORDER BY 1,2,3;
--Nom i NIF de les persones que volen a França i que no han facturat mai maletes de color vermell.
--NOTA: Ha d'incloure també les persones que volen a França però que no han facturat mai.
SELECT P.NOM, P.NIF
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
AND P.NIF NOT IN (SELECT DISTINCT P.NIF
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell')
GROUP BY P.NOM, P.NIF;
SELECT DISTINCT P.NIF
FROM PERSONA P, MALETA M, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.NIF_PASSATGER = M.NIF_PASSATGER
AND B.CODI_VOL = M.CODI_VOL
AND B.DATA = M.DATA
AND M.COLOR = 'vermell'
ORDER BY 1;
SELECT P.NOM, P.NIF
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY P.NOM, P.NIF;
--Per a cada vol, calculi el nombre de maletes facturades. Retorneu el codi de vol, la data,
-- i el nombre de maletes. (NOTA: les dates s'han de presentar al format 'DD/MM/YYYY')
SELECT V.CODI_VOL, TO_CHAR(V.DATA,'DD/MM/YYYY') AS DATA, COUNT(*) AS N_MALETES
FROM VOL V, MALETA M
WHERE M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
GROUP BY V.CODI_VOL, V.DATA
ORDER BY 1,2,3;
--Preu promig de les reserves pagats amb Visa. El numerador es
--el preu total de les reserves pagades amb Visa i el denominador és el numero de
--reserves pagats amb Visa.
SELECT T1.PREU_TOTAL/T2.N_RESERVES AS PROMIG_RESERVES
FROM (SELECT SUM(PREU) AS PREU_TOTAL
FROM RESERVA
WHERE MODE_PAGAMENT = 'Visa') T1,
(SELECT COUNT(*) AS N_RESERVES
FROM RESERVA
WHERE MODE_PAGAMENT = 'Visa') T2
ORDER BY 1;
--3. Pais dels passatgers que han viatjat en primera classe més de 5 vegades.
SELECT P.PAIS
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND P.NIF IN (SELECT P.NIF
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = '1'
GROUP BY P.NIF
HAVING COUNT(*) > 5)
GROUP BY P.PAIS
ORDER BY 1;
--passatgers que han viatjat en primeraclasse mes de 5 cops:
SELECT P.NIF
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CLASSE = '1'
GROUP BY P.NIF
HAVING COUNT(*) > 5;
--4. Codi(s) de vol del(s) avion(s) més petit(s) (menor nombre de seients)
SELECT DISTINCT V.CODI_VOL
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA
HAVING COUNT(*) <= ALL(SELECT COUNT(*) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA);
--nombre seients de cada avio:
SELECT COUNT(*) AS N_SEIENTS
FROM VOL V, SEIENT S
WHERE V.CODI_VOL = S.CODI_VOL
AND V.DATA = S.DATA
GROUP BY V.CODI_VOL, V.DATA
ORDER BY 1;
--8. Nombre de maletes de cada passatger. Retorna el nom del passatger, el seu NIF i el número
--de maletes que posseeix
SELECT P.NOM, P.NIF, COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF, P.NOM
ORDER BY 1,2,3;
--nombre maletes de cada passatger amb maleta
SELECT COUNT(*) AS N_MALETES
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
GROUP BY P.NIF;
--5. Noms de les persones que han facturat tots els mateixos colors de maleta que
--Jose Luis Lamata Feliz (també poden haver facturat maletes d'altres colors)
SELECT DISTINCT R1.NOM
FROM (SELECT DISTINCT P.NOM, M.COLOR
FROM PERSONA P, MALETA M
WHERE P.NIF=M.NIF_PASSATGER) R1
WHERE (R1.NOM) NOT IN (SELECT R2.NOM
FROM (SELECT DISTINCT M.COLOR
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND P.NOM = 'Jose Luis Lamata Feliz') S,
(SELECT DISTINCT P.NOM, M.COLOR
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER) R2
WHERE (S.COLOR, R2.NOM) NOT IN (SELECT R3.COLOR, R3.NOM
FROM (SELECT DISTINCT P.NOM, M.COLOR
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER) R3))
ORDER BY 1;
--colors de les maletes que ha facturat Jose Luis Lamata Feliz:
SELECT M.COLOR
FROM MALETA M, PERSONA P
WHERE P.NIF = M.NIF_PASSATGER
AND P.NOM = 'Jose Luis Lamata Feliz'
GROUP BY M.COLOR
ORDER BY 1;
--2. Nombre de vols promig de totes les companyies.
--NOTA: el numerador és el nombre de vols totals i el denominador el nombre total de companyies.
SELECT T1.VOLS_TOTALS/T2.N_COMPANYIES AS PROMIG
FROM (SELECT COUNT(*) AS VOLS_TOTALS
FROM VOL V) T1,
(SELECT COUNT(DISTINCT V.COMPANYIA) AS N_COMPANYIES
FROM VOL V) T2
ORDER BY 1;
SELECT DISTINCT TV.TOTAL_VOLS/TC.TOTAL_COMPANYIES
FROM (SELECT SUM(COUNT(*)) AS TOTAL_VOLS
FROM VOL V
GROUP BY V.COMPANYIA) TV, (SELECT SUM(COUNT(DISTINCT V.COMPANYIA)) AS TOTAL_COMPANYIES
FROM VOL V
GROUP BY V.COMPANYIA) TC
ORDER BY 1;
--7. Percentage del pes total de de maletes que correspon als passatgers francesos.
--NOTA: un percentatge s'expressa amb nombres entre 0 i 100.
SELECT (T1.PES_FRANCESOS*100)/T2.PES_TOTAL AS PERCENTATGE
FROM (SELECT SUM(M.PES) AS PES_FRANCESOS
FROM PERSONA P, MALETA M
WHERE P.NIF = M.NIF_PASSATGER
AND P.PAIS = 'França') T1,
(SELECT SUM(M.PES) AS PES_TOTAL
FROM MALETA M) T2
ORDER BY 1;
--6. Per a tots els vols de l'any 2014 mostra el NIF dels que volen i dels que han comprat
--cada bitllet.
SELECT NIF_PASSATGER, NIF_CLIENT
FROM BITLLET B, RESERVA R
WHERE B.LOCALITZADOR = R.LOCALITZADOR
AND B.DATA BETWEEN TO_DATE('01/01/2014','DD/MM/YYYY') AND TO_DATE('31/12/2014','DD/MM/YYYY')
GROUP BY NIF_CLIENT, NIF_PASSATGER
ORDER BY 1,2;

--Quantes maletes ha facturat en Aitor Tilla al vol amb destinació Pekin de la data 1/10/2013?
SELECT COUNT(M.CODI_MALETA)
FROM MALETA M, VOL V, PERSONA P, AEROPORT A
WHERE M.NIF_PASSATGER = P.NIF
AND M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND P.NOM = 'Aitor Tilla'
AND A.CIUTAT = 'Pekin'
AND V.DATA = TO_DATE('01/10/2013', 'DD/MM/YYYY')
ORDER BY 1;
Codi_Vol i data dels vols amb origen a algún aeroport rus
SELECT V.CODI_VOL, TO_DATE(V.DATA, 'DD/MM/YYYY') AS DATA
FROM VOL V
WHERE ORIGEN IN (SELECT CODI_AEROPORT
FROM AEROPORT A
WHERE PAIS = 'Russia')
ORDER BY 1;

--Portes d'embarcament (terminal, area, porta) de l'aeroport JFK on no han sortit mai cap vol amb destinació França
(SELECT PE.TERMINAL, PE.AREA, PE.PORTA
FROM PORTA_EMBARCAMENT PE
WHERE PE.CODI_AEROPORT = 'JFK')
MINUS
(SELECT PE.TERMINAL, PE.AREA, PE.PORTA
FROM VOL V, AEROPORT A, PORTA_EMBARCAMENT PE
WHERE V.ORIGEN = PE.CODI_AEROPORT
AND V.TERMINAL = PE.TERMINAL
AND V.AREA = PE.AREA
AND V.PORTA = PE.PORTA
AND PE.CODI_AEROPORT ='JFK'
AND A.PAIS = 'França'
AND V.DESTINACIO = A.CODI_AEROPORT)
ORDER BY 1,2,3;

--Per les reserves de vols amb destinació el Marroc, aquelles on només viatja un únic passatger. Mostreu localitzador i nom del passatger que hi viatja.
SELECT B. LOCALITZADOR, P.NOM
FROM PERSONA P, BITLLET B
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL IN (SELECT CODI_VOL
FROM BITLLET B
WHERE CODI_VOL IN (SELECT V.CODI_VOL
FROM AEROPORT A, VOL V
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND A.PAIS = 'Marroc')
GROUP BY CODI_VOL
HAVING COUNT(*) = 1)
ORDER BY 1;

--Companyies que tenen, almenys, els mateixos tipus d'avions que Vueling airlines (El resultat també ha de tornar Vueling Airlines)
SELECT DISTINCT COMPANYIA
FROM VOL
WHERE TIPUS_AVIO IN (SELECT DISTINCT TIPUS_AVIO
FROM VOL
WHERE COMPANYIA = 'VUELING AIRLINES')
ORDER BY 1;

--Nom de la companyia que fa més vols amb origen Italia
SELECT V.COMPANYIA
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA
HAVING COUNT(*) = (SELECT MAX(COUNT(*))
FROM VOL V, AEROPORT A
WHERE V.ORIGEN = A.CODI_AEROPORT
AND A.PAIS = 'Italia'
GROUP BY V.COMPANYIA)
ORDER BY 1;


--Per a cada aeroport i terminal, nom de l'aeroport, terminal i nombre d'arees.
SELECT A.NOM, PE.TERMINAL, COUNT(DISTINCT PE.AREA) AS AREAS
FROM PORTA_EMBARCAMENT PE, AEROPORT A
WHERE PE.CODI_AEROPORT = A.CODI_AEROPORT
GROUP BY A.NOM, PE.TERMINAL
ORDER BY 1,2;

--Promitg de vols per a cada companyia. Aclariment: el numerador es el nombre de vols totals i el denominador el nombre total de companyies
SELECT N.VOLS/D.COMPANYIES AS PROMITG
FROM (SELECT COUNT(*) AS VOLS FROM VOL) N,
(SELECT COUNT(DISTINCT COMPANYIA) AS COMPANYIES FROM VOL) D
ORDER BY 1;

--Promig d'espanyols als vols amb destinació Munich.
SELECT E.ESPANYOLS/T.TOTAL
FROM (SELECT COUNT(*) AS ESPANYOLS
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Munich'
AND P.PAIS = 'Espanya') E, (SELECT COUNT(*) AS TOTAL
FROM BITLLET B, VOL V, AEROPORT A
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.CIUTAT = 'Munich') T
ORDER BY 1;

--Nom, correu i companyia aeria dels clients (persones) que han fet reserves de 2 bitllets d’anada i 2 bitlles de tornada
(SELECT P.NOM, P.MAIL, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.ANADA_TORNADA = 'ANADA'
GROUP BY P.NOM, P.MAIL, V.COMPANYIA
HAVING COUNT(B.ANADA_TORNADA) = 2)
INTERSECT
(SELECT P.NOM, P.MAIL, V.COMPANYIA
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.ANADA_TORNADA = 'TORNADA'
GROUP BY P.NOM, P.MAIL, V.COMPANYIA
HAVING COUNT(B.ANADA_TORNADA) = 2)
ORDER BY 1,2,3;

--Pes facturat als vols de 'AIR FRANCE' del 2013.
SELECT SUM(PES) AS PES
FROM MALETA M, VOL V
WHERE M.CODI_VOL = V.CODI_VOL
AND M.DATA = V.DATA
AND V.COMPANYIA = 'AIR FRANCE'
AND TO_CHAR(V.DATA, 'YYYY') = '2013'
ORDER BY 1;

--Nacionalitat dels passatgers del vols amb destinació Paris Orly Airport.
SELECT DISTINCT P.PAIS
FROM PERSONA P, BITLLET B, VOL V, AEROPORT A
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND V.DESTINACIO = A.CODI_AEROPORT
AND A.NOM = 'Paris Orly Airport'
ORDER BY 1;

--En quines files del vol AEA2159 del 01/08/13 hi ha seients de finestra (lletra A) disponibles per a ús dels passatgers?
SELECT FILA
FROM SEIENT
WHERE CODI_VOL = 'AEA2159'
AND TO_CHAR(DATA, 'DD/MM/YYYY') = '01/08/2013'
AND LLETRA = 'A'
ORDER BY 1;

--Número de terminals dels aeroports francesos (retorneu el nom de l’aeroport i la quantitat de terminals).
SELECT A.NOM, COUNT(DISTINCT PE.TERMINAL) AS TERMINALS
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
AND A.PAIS = 'França'
GROUP BY A.NOM
ORDER BY 1;

--Nom de la persona que més cops ha pagat amb Paypal. Retorneu també el número de cops que ho ha fet.
SELECT P.NOM, COUNT(*) AS COPS
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
and R.MODE_PAGAMENT = 'Paypal'
GROUP BY P.NOM
HAVING COUNT(*) > ANY (SELECT COUNT(*)
FROM RESERVA
WHERE MODE_PAGAMENT = 'Paypal'
GROUP BY NIF_CLIENT)
ORDER BY 1;

--Destinació dels vols que tenen entre els seus passatgers persones de totes les ciutats de Turquia.
SELECT DISTINCT V.DESTINACIO
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND B.NIF_PASSATGER IN (SELECT NIF
FROM PERSONA
WHERE PAIS = 'Turquia')
ORDER BY 1;

--Pais, codi_aeroport, terminal i àrea dels aeroport amb 3 portes d'embarcament i el nom del pais comenci per 'E'
SELECT DISTINCT A.PAIS, PE.CODI_AEROPORT, PE.TERMINAL, PE.AREA
FROM AEROPORT A, PORTA_EMBARCAMENT PE
WHERE A.CODI_AEROPORT = PE.CODI_AEROPORT
AND A.PAIS LIKE 'E%'
AND PE.CODI_AEROPORT IN (SELECT PE.CODI_AEROPORT
FROM PORTA_EMBARCAMENT PE
GROUP BY PE.CODI_AEROPORT
HAVING COUNT(DISTINCT PORTA) = 3)
ORDER BY 1, 2 ,3, 4;

--Mostreu el numero de passatgers que no han escollit els seus seients
SELECT COUNT(*)
FROM BITLLET
WHERE FILA IS NULL
AND LLETRA IS NULL
ORDER BY 1;

--Nombre de passatgers agrupats segons pais de destinació (en el vol). Mostreu pais i nombre de passatgers
SELECT A.PAIS, COUNT(B.NIF_PASSATGER) AS NOMBRE_PASSATGERS
FROM VOL V, AEROPORT A, BITLLET B
WHERE V.DESTINACIO = A.CODI_AEROPORT
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY A.PAIS
ORDER BY 1;

--Nom dels clients que sempre paguem amb Visa Electron
SELECT P.NOM
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
AND R.MODE_PAGAMENT = 'Visa Electron'
GROUP BY P.NOM
HAVING (P.NOM, COUNT(*)) IN (SELECT P.NOM, COUNT(*)
FROM PERSONA P, RESERVA R
WHERE P.NIF = R.NIF_CLIENT
GROUP BY P.NOM)
ORDER BY 1;
--Ciutats espanyoles amb aeroports
SELECT CIUTAT
FROM AEROPORT
WHERE PAIS = 'Espanya';

--Noms de les companyies que volen als mateixos aeroports on ha volat Estela Gartija.
SELECT DISTINCT COMPANYIA
FROM VOL V
WHERE DESTINACIO IN (SELECT V.DESTINACIO
FROM PERSONA P, BITLLET B, VOL V
WHERE P.NIF = B.NIF_PASSATGER
AND B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
AND P.NOM = 'Estela Gartija')
ORDER BY 1;

--Codi de l'aeroport d'origen, codi de vol, data de vol i nombre de passatgers dels vols que tenen el nombre de passatges més petit.
SELECT V.ORIGEN, V.CODI_VOL, V.DATA, COUNT(B.NIF_PASSATGER)
FROM VOL V, BITLLET B
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.ORIGEN, V.CODI_VOL, V.DATA
HAVING COUNT(B.NIF_PASSATGER) = (SELECT MIN(COUNT (*))
FROM BITLLET B, VOL V
WHERE B.CODI_VOL = V.CODI_VOL
AND B.DATA = V.DATA
GROUP BY V.CODI_VOL)
ORDER BY 1,2,3,4;

--Percentatge d'ús de l'aeroport de Barcelona. Indicació: el numerador és el nombre de vols ambos origen, o destinació, Barcelona; el denominador el nombre total de vols.
SELECT T1.BARCELONA/T2.TOTAL
FROM (SELECT COUNT(*) AS BARCELONA
FROM VOL
WHERE ORIGEN = (SELECT CODI_AEROPORT
FROM AEROPORT
WHERE CIUTAT = 'Barcelona')
OR DESTINACIO = (SELECT CODI_AEROPORT
FROM AEROPORT
WHERE CIUTAT = 'Barcelona')) T1, (SELECT COUNT(*) AS TOTAL
FROM VOL) T2
ORDER BY 1;

--1-Nom i intèrpret dels espectacles musicals amb un únic dia de representació, ordenats per nom i intèrpret. 
SELECT Esp.Nom, Esp.Interpret 
FROM Espectacles Esp 
WHERE Esp.Tipus='Musical' AND Esp.Data_Inicial=Esp.Data_Final

--2-Codi dels espectacles, dia de la representació i DNI dels clients de les entrades reservades pel mes de març del 2012 en algun dels recintes amb codi 101, 103 o 105, ordenats per codi d’espectacle i dia de representació. Si un client té reservada més d’una entrada pel mateix expectacle només ha de sortir un cop. 
SELECT DISTINCT Codi_Espectacle, Data,DNI_Client 
FROM Entrades 
WHERE Data BETWEEN TO_DATE('01/03/2012','dd/mm/yyyy') AND TO_DATE('31/03/2012','dd/mm/yyyy') AND Codi_Recinte IN (101, 103, 105) 

--3-Codi, nom i nombre de dies que han estat en cartellera tots els espectacles teatrals representats l’any 2012, ordenats de més a menys dies de representació. 
SELECT ES.CODI, ES.NOM, ES.DATA_FINAL - ES.DATA_INICIAL + 1 AS NUM_DIES 
FROM ESPECTACLES ES 
WHERE ES.TIPUS = 'Teatre' AND ES.DATA_INICIAL <= TO_DATE ('31/12/11', 'DD/MM/YY') AND ES.DATA_FINAL >= TO_DATE ('01/01/11', 'DD/MM/YY') 

--4-Data i hora de totes les representacions de l’espectacle La extraña pareja ordenades per data. 
SELECT DISTINCT RP.Data, TO_CHAR(RP.Hora,'HH24:MI:SS') AS Hora 
FROM Espectacles EC, Representacions RP 
WHERE EC.Nom = 'La extraña pareja' AND EC.Codi = RP.Codi_Espectacle

--5-Espectacles que es realitzen fora de Barcelona. De cada espectacle, s’ha de recuperar el nom, tipus, intèrpret, nom i ciutat del recinte. 
SELECT ES.NOM, ES.TIPUS, ES.INTERPRET, RC.NOM, RC.CIUTAT 
FROM ESPECTACLES ES, RECINTES RC 
WHERE RC.CIUTAT != 'Barcelona' AND RC.CODI = ES.CODI_RECINTE 

--6-Per cadascuna de les zones del teatre on es representa La extraña pareja, nom de la zona, capacitat i preu de les entrades. 
SELECT ZR.Zona, ZR.Capacitat, PE.Preu FROM Espectacles EC, Preus_Espectacles PE, Zones_Recinte ZR 
WHERE EC.Nom = 'La extraña pareja' AND EC.Codi = PE.Codi_Espectacle AND PE.Codi_Recinte = ZR.Codi_Recinte AND PE.Zona = ZR.Zona 

--7-Ciutats on ha actuat algun cop El Tricicle. S’ha de recuperar nom de la ciutat, nom de l’espectacle que s’hi ha representat i dates inicial i final de les representacions. 
SELECT RC.Ciutat, EC.Nom, EC.Data_Inicial, EC.Data_Final 
FROM Espectacles EC, Recintes RC WHERE EC.Interpret = 'El Tricicle' AND EC.Codi_Recinte = RC.Codi 

--8-Nom, cognoms i adreça dels espectadors que han adquirit alguna entrada per la representació del dia 20 de novembre del 2011 a les 21h deWest Side Story, ordenats per cognoms i adreça. 
SELECT DISTINCT EP.NOM, EP.COGNOMS, EP.ADREÇA FROM ESPECTADORS EP, ENTRADES EN, ESPECTACLES ES 
WHERE ES.NOM = 'West Side Story' AND ES.CODI = EN.CODI_ESPECTACLE AND EP.DNI=EN.DNI_CLIENT AND EN.DATA = TO_DATE('20/11/2011','DD/MM/YYYY') AND EN.HORA = TO_DATE('30/12/1899 21:00:00', 'DD/MM/YYYY HH24:MI:SS') 

--9-Nom de tots els espectacles pels quals tenen reservada alguna entrada el dia 23 de febrer del 2012 els espectadors que viuen a Cerdanyola. 
SELECT DISTINCT ES.NOM FROM ESPECTACLES ES, ESPECTADORS EP, ENTRADES EN 
WHERE EN.DATA = TO_DATE('23/02/2012', 'DD/MM/YYYY') AND EP.CIUTAT = 'Cerdanyola' AND EN.CODI_ESPECTACLE = ES.CODI AND EN.DNI_CLIENT = EP.DNI 

--10-Seients (zona, fila i número) del recinte on es representa Hamlet, ordenats per zona, fila i número. 
SELECT EN.ZONA, EN.FILA, EN.NUMERO 
FROM SEIENTS EN, ESPECTACLES ES, RECINTES RC 
WHERE ES.NOM = 'Hamlet' AND RC.CODI = EN.CODI_RECINTE AND ES.CODI_RECINTE = RC.CODI 

-- 11-Entrades (nom de l'espectacle, data, hora, zona, fila i número) adquirides per clients de Barcelona per les representacions del mes de febrer del 2012 en el teatre Romea. 
SELECT ES.NOM, EN.DATA, TO_CHAR(EN.HORA, 'HH24:MI:SS'), EN.ZONA, EN.FILA, EN.NUMERO 
FROM ESPECTACLES ES, ENTRADES EN, ESPECTADORS EP, RECINTES RC 
WHERE RC.NOM = 'Romea' AND EP.CIUTAT = 'Barcelona' AND ES.CODI = EN.CODI_ESPECTACLE AND EN.DNI_CLIENT = EP.DNI AND EN.DATA > TO_DATE ('01/02/2012', 'DD/MM/YYYY') AND EN.DATA < TO_DATE('28/02/2012', 'DD/MM/YYYY') 

--12-Nom dels espectacles que s'han representat al mateix recinte que L'auca del senyor Esteve. 
SELECT DISTINCT ES.NOM 
FROM ESPECTACLES ES, ESPECTACLES ES1, RECINTES RC 
WHERE ES1.NOM = 'L''auca del senyor Esteve' AND ES.CODI_RECINTE = ES1.CODI_RECINTE AND ES.NOM != 'L''auca del senyor Esteve' 

--13-Nom dels espectacles que es representaran en el mateix recinte que Cianur i puntes de coixí un cop s'acabin les representacions d'aquest espectacle. 

SELECT ES1.NOM 
FROM ESPECTACLES ES, ESPECTACLES ES1 
WHERE ES.NOM = 'Cianur i puntes de coixí' AND ES.CODI_RECINTE = ES1.CODI_RECINTE AND ES.DATA_INICIAL < ES1.DATA_INICIAL 

--14- Per totes les entrades venudes a espectadors que no són de Barcelona per espectacles representats a Barcelona, recuperar el nom de l'espectacle, el nom del recinte, la data i hora de la representació, la zona, fila i número del seient, el preu de l'entrada i el nom i cognoms de l'espectador. 
SELECT DISTINCT ES.NOM, RC.NOM, EN.DATA, TO_CHAR(EN.HORA, 'HH24:MI:SS'), EN.ZONA, EN.FILA, EN.NUMERO, PE.PREU, EP.NOM, EP.COGNOMS 
FROM ESPECTACLES ES, ENTRADES EN, RECINTES RC, PREUS_ESPECTACLES PE, ESPECTADORS EP
WHERE EP.CIUTAT != 'Barcelona' AND RC.CIUTAT = 'Barcelona' AND PE.CODI_ESPECTACLE = ES.CODI AND ES.CODI_RECINTE = RC.CODI AND PE.CODI_RECINTE = RC.CODI AND EP.DNI = EN.DNI_CLIENT AND EN.ZONA = PE.ZONA AND EN.CODI_RECINTE = RC.CODI AND EN.CODI_ESPECTACLE = ES.CODI 

--15-Nombre d'espectacles diferents representats a Girona durant l'any 2011. 
SELECT DISTINCT COUNT(*) AS NUM_ESPECTACLES FROM ESPECTACLES ES, RECINTES RC WHERE RC.CIUTAT = 'Girona' AND ES.CODI_RECINTE = RC.CODI
AND ES.DATA_INICIAL > TO_DATE('01/01/2011', 'DD/MM/YYYY') AND ES.DATA_FINAL < TO_DATE('31/12/2011','DD/MM/YYYY') 

--16-Capacitat total del teatre Romea. 
SELECT SUM(ZR.CAPACITAT) AS CAPACITAT_TOTAL 
FROM ZONES_RECINTE ZR, RECINTES RC 
WHERE RC.NOM = 'Romea' AND RC.CODI = ZR.CODI_RECINTE 

--17-Preu màxim i mínim de les entrades per l'espectacle Jazz a la tardor. 
SELECT MAX(PE.PREU) AS PREU_MAX, MIN(PE.PREU) AS PREU_MIN 
FROM PREUS_ESPECTACLES PE, ESPECTACLES ES WHERE PE.CODI_ESPECTACLE = ES.CODI AND ES.NOM = 'Jazz a la tardor' 

--18-Preu total que ha de pagar per totes les entrades que té reservades l'espectador amb compte corrent 1111-111-11-1234567890 per l'espectacle Entre Tres. 

SELECT SUM(PE.PREU) AS PREU_TOTAL FROM PREUS_ESPECTACLES PE, ESPECTACLES ES, ESPECTADORS ED, ENTRADES EN 
WHERE PE.CODI_ESPECTACLE = ES.CODI AND ES.NOM = 'Entre Tres' AND ED.COMPTE_CORRENT = '1111-111-11-1234567890' AND EN.DNI_CLIENT = ED.DNI AND EN.CODI_ESPECTACLE = ES.CODI 

--19-Recaptació total que s'obtindria en una representació d'Entre Tres en cas de vendre totes les entrades. SELECT SUM(PE.PREU) AS RECAPTACIO 
FROM ESPECTACLES ES, PREUS_ESPECTACLES PE, SEIENTS SE 
WHERE ES.NOM = 'Entre Tres' AND PE.CODI_ESPECTACLE = ES.CODI AND SE.ZONA = PE.ZONA AND SE.CODI_RECINTE = PE.CODI_RECINTE AND SE.CODI_RECINTE = ES.CODI_RECINTE 

--20-Recaptació obtinguda a la representació de Hamlet del dia 31 de març del 2012. 
SELECT SUM(PE.PREU) AS RECAPTACIO 
FROM PREUS_ESPECTACLES PE, ENTRADES EN, ESPECTACLES ES 
WHERE ES.NOM = 'Hamlet' AND EN.DATA = TO_DATE('31/03/2012','DD/MM/YYYY') AND PE.CODI_ESPECTACLE = ES.CODI AND ES.CODI = EN.CODI_ESPECTACLE AND EN.CODI_RECINTE = ES.CODI_RECINTE AND PE.ZONA = EN.ZONA

--21-Preu mig per entrada de La extraña pareja. Suposem que el preu mig el calculem sumant el pteu de les entrades de totes les zones del recinte dividit pel nombre de zones. 
SELECT SUM(P.preu)/count(*) as PreuMig FROM Espectacles E, Preus_Espectacles P WHERE E.Codi=P.Codi_Espectacle AND E.nom='La extraña pareja' 

--22-Nombre total d'espectadors que han assistit a les representacions de Mar i Cel. 
SELECT COUNT(*) AS NUM_ESPECTADORS FROM ESPECTADORS EP, ENTRADES EN, ESPECTACLES ES WHERE EP.DNI = EN.DNI_CLIENT AND ES.NOM = 'Mar i Cel' AND EN.CODI_ESPECTACLE = ES.CODI 

--23-Mitjana d'assistència a les representacions de West Side Story. 
SELECT COUNT(*)/COUNT(DISTINCT EN.DATA) AS MITJANA FROM ENTRADES EN, ESPECTACLES ES WHERE ES.NOM = 'West Side Story' AND ES.CODI = EN.CODI_ESPECTACLE

--24- Nombre total de representacions, nombre d'espectacles diferents i promig de representacions per espectacle realitzades al Teatre Municipal de Girona durant 2011. 
SELECT COUNT(*) AS NUM_REP, COUNT(DISTINCT ES.CODI) AS NUM_ESP, COUNT(*)/COUNT(DISTINCT ES.CODI) AS PROMIG_REP FROM REPRESENTACIONS RP, ESPECTACLES ES, RECINTES RC 
WHERE RC.NOM = 'Municipal' AND RC.CIUTAT = 'Girona' AND RC.CODI = ES.CODI_RECINTE AND RP.CODI_ESPECTACLE = ES.CODI AND RP.DATA >= TO_DATE('01/01/2011','DD/MM/YYYY') AND RP.DATA <= TO_DATE('31/12/2011', 'DD/MM/YYYY') 

--25-Nombre de recintes diferents on han actuat La Joventut de la Faràndula. 
SELECT COUNT (DISTINCT RC.Codi) FROM Espectacles EC, Recintes RC WHERE EC.Interpret = 'La Joventut de la Faràndula' AND RC.Codi = EC.Codi_Recinte

-- 26- Data, hora i nombre d’entrades venudes de cadascuna de les representacions de El Màgic d’Oz.
SELECT Ent.Data, to_char(r.hora, 'HH24:MI:SS'), COUNT (*)
FROM Entrades Ent, Representacions R, Espectacles E
WHERE Ent.Codi_Espectacle=R.Codi_Espectacle AND R.Codi_Espectacle=E.Codi
and R.Data=Ent.Data AND R.Hora=Ent.Hora AND E.Nom='El Màgic d''Oz'
GROUP BY Ent.Data, to_char(r.hora, 'HH24:MI:SS')

--27-Preu màxim i mínim per cadascun dels espectacles representats al Liceu. A més dels preus s’ha de recuperar també el nom de l’espectacle i les dates inicial i final.
SELECT MAX(P.PREU), MIN(P.PREU), E.NOM, E.DATA_INICIAL, E.DATA_FINAL
FROM PREUS_ESPECTACLES P, ESPECTACLES E, RECINTES R
WHERE E.CODI_RECINTE=R.CODI and P.CODI_ESPECTACLE=E.CODI
and P.CODI_RECINTE=R.CODI and R.NOM='Liceu'
GROUP BY E.NOM, E.DATA_INICIAL, E.DATA_FINAL

--32-Nom dels recintes de Barcelona que tenen una capacitat total més gran que el Teatre Victòria. 
SELECT r.nom FROM ZONES_RECINTE Z, RECINTES R WHERE R.codi= Z.codi_recinte and R.CIUTAT='Barcelona' GROUP BY R.NOM HAVING SUM(z.capacitat) > (SELECT SUM(Z2.CAPACITAT) FROM ZONES_RECINTE Z2, RECINTES R2 WHERE R2.codi= Z2.codi_recinte and R2.nom='Victòria')

--33-Nom dels espectacles amb un preu màxim de les entrades superior a 18€. 
SELECT DISTINCT E.NOM FROM ESPECTACLES E, PREUS_ESPECTACLES P WHERE E.CODI=P.CODI_ESPECTACLE AND P.PREU> 18

--38-Seients (zona, fila i número) que no s'han ocupat en cap de les representacions de El país de les cent paraules. 
SELECT S.Zona, S.Fila, S.Numero FROM Espectacles EC, Seients S WHERE EC.Nom = 'El país de les Cent Paraules' AND EC.Codi_Recinte = S.Codi_Recinte AND
NOT EXISTS ( SELECT * FROM Entrades EN WHERE EN.Codi_Espectacle = EC.Codi AND EN.Codi_Recinte = S.Codi_Recinte AND EN.Zona = S.Zona AND EN.Fila = S.Fila AND S.Numero = EN.Numero) 

--40-Seients lliures (zona, fila i número) per anar a veure Hamlet el dia 6 d'abril del 2012, ordenat per zona, fila i número. 
SELECT SE.ZONA, SE.FILA, SE.NUMERO FROM ESPECTACLES ES, SEIENTS SE 
WHERE ES.CODI_RECINTE = SE.CODI_RECINTE AND ES.NOM = 'Hamlet'
AND NOT EXISTS (SELECT * FROM ENTRADES EN WHERE EN.DATA = TO_DATE('06/04/2012','DD/MM/YYYY') AND SE.CODI_RECINTE = EN.CODI_RECINTE AND EN.CODI_ESPECTACLE = ES.CODI AND EN.FILA = SE.FILA AND EN.ZONA = SE.ZONA AND EN.NUMERO = SE.NUMERO)

--41-Nom del recinte amb una capacitat total més gran. 
SELECT R.NOM FROM RECINTES R, ZONES_RECINTE Z WHERE R.CODI = Z.CODI_RECINTE 
GROUP BY R.NOM HAVING SUM(Z.CAPACITAT)>= (SELECT MAX(SUM(Z1.CAPACITAT)) 
FROM ZONES_RECINTE Z1 
GROUP BY Z1.CODI_RECINTE)

--42-Codi i nom de l'espectacle i zona de recinte amb el preu més alt de tots.
 SELECT E.CODI, E.NOM, Z.ZONA FROM ESPECTACLES E, PREUS_ESPECTACLES PE, ZONES_RECINTE Z 
 WHERE e.codi=pe.codi_espectacle AND pe.codi_recinte=z.codi_recinte AND pe.ZONA = Z.ZONA AND PE.PREU>=(SELECT MAX(PE2.PREU) 
 FROM ESPECTACLES E2, PREUS_ESPECTACLES PE2, ZONES_RECINTE Z2 WHERE e2.codi=pe2.codi_espectacle AND pe2.codi_recinte=z2.codi_recinte AND pe2.ZONA = Z2.ZONA)

--45-Zona, fila i número dels seients que s'han ocupat sempre en totes les representacions de l'espectacle Els Pastorets. 
SELECT S.Zona, S.Fila, S.Numero FROM Espectacles EC, Seients S 
WHERE EC.Nom = 'Els Pastorets' AND S.Codi_Recinte = EC.Codi_Recinte 
AND NOT EXISTS ( SELECT * FROM Representacions RP WHERE RP.Codi_Espectacle = EC.Codi AND NOT EXISTS (SELECT * FROM Entrades EN WHERE EN.Codi_Espectacle = EC.Codi AND EN.Data = RP.Data AND EN.Hora = RP.Hora AND EN.Codi_Recinte = S.Codi_Recinte AND EN.Zona = S.Zona AND EN.Fila = S.Fila AND EN.Numero = S.Numero ) )

--46-Nom de l'espectacle del que s'han fet més representacions. 
SELECT E.NOM FROM ESPECTACLES E, REPRESENTACIONS R WHERE E.CODI=R.CODI_ESPECTACLE 
GROUP BY E.NOM HAVING COUNT(*)>=(SELECT MAX(COUNT(*)) FROM REPRESENTACIONS R1 GROUP BY R1.CODI_ESPECTACLE)

--47-Zones del recinte on es representa l'espectacle Mar i Cel amb tots els seients ocupats per la representació del dia 2 de març del 2012. 
SELECT ZR.Zona 
FROM Espectacles EC, Zones_Recinte ZR 
WHERE EC.Nom = 'Mar i Cel' AND ZR.Codi_Recinte = EC.Codi_Recinte 
AND NOT EXISTS ( SELECT * FROM Seients S WHERE S.Codi_Recinte = ZR.Codi_Recinte AND S.Zona = ZR.Zona 
AND NOT EXISTS ( SELECT * FROM Entrades EN WHERE EN.Codi_Espectacle = EC.Codi AND EN.Data = TO_DATE('02/03/2012','dd/mm/yyyy') 
AND EN.Zona = S.Zona AND EN.Fila = S.Fila AND EN.Numero = S.Numero )) 

--48- Nom dels espectacles representats a Barcelona pels quals han comprat entrades espectadors que viuen fora de la ciutat. 
SELECT ES.NOM 
FROM ESPECTACLES ES, ESPECTADORS EP, RECINTES RC, ENTRADES EN 
WHERE ES.CODI = EN.CODI_ESPECTACLE AND ES.CODI_RECINTE = RC.CODI AND EN.CODI_RECINTE = RC.CODI AND EP.DNI = EN.DNI_CLIENT AND EP.CIUTAT != 'Barcelona' and RC.CIUTAT = 'Barcelona' GROUP BY ES.NOM

--49-Ciutat que no sigui Barcelona on es realitzen més espectacles. 
SELECT R.CIUTAT FROM ESPECTACLES E, RECINTES R WHERE E.CODI_RECINTE= R.CODI AND R.CIUTAT<>'Barcelona' GROUP BY R.CIUTAT HAVING COUNT(*)>=(SELECT MAX(COUNT(*)) FROM ESPECTACLES E1, RECINTES R1 WHERE E1.CODI_RECINTE=R1.CODI AND R1.CIUTAT<> 'Barcelona' GROUP BY R1.CIUTAT)

--50-Intèrprets que van realitzar algun espectacle l'any 2012, però que no n'han fet cap l'any 2012. 
SELECT DISTINCT EC.Interpret FROM Espectacles EC, Representacions RP 
WHERE EC.Codi = RP.Codi_Espectacle AND RP.Data BETWEEN TO_DATE('01/01/2011', 'dd/mm/yyyy') AND TO_DATE('31/12/2011','dd/mm/yyyy') AND NOT EXISTS (SELECT * FROM Espectacles EC2, Representacions RP2
WHERE EC2.Codi = RP2.Codi_Espectacle AND EC2.Interpret = EC.Interpret AND RP2.Data > TO_DATE('01/01/2012','dd/mm/yyyy'))

--51-DNI, nom i cognoms dels espectadors que s'han gastat més de 500€ en espectacles. 
SELECT E.DNI, E.NOM, E.COGNOMS FROM ESPECTADORS E, ENTRADES EN, PREUS_ESPECTACLES P WHERE E.DNI=EN.DNI_CLIENT AND EN.CODI_ESPECTACLE=P.CODI_ESPECTACLE AND EN.ZONA=P.ZONA GROUP BY E.DNI, E.NOM, E.COGNOMS HAVING SUM(P.PREU)>500

--52-Nom, adreça i ciutat dels recintes amb capacitat de més de 60 persones. 
SELECT R.NOM, R.ADREÇA, R.CIUTAT FROM RECINTES R, ZONES_RECINTE Z WHERE R.CODI=Z.CODI_RECINTE GROUP BY R.NOM, R.ADREÇA, R.CIUTAT HAVING SUM(Z.CAPACITAT)>60

--53-Nom dels intérprets que han realitzat un únic espectacle. 
SELECT E.INTERPRET FROM ESPECTACLES E GROUP BY E.INTERPRET HAVING COUNT(*)=1

--54-Nom i tipus d'espectacles que han fet una única representació. 
SELECT E.NOM, E.TIPUS FROM ESPECTACLES E, REPRESENTACIONS R 
WHERE E.CODI= R.CODI_ESPECTACLE 
GROUP BY E.NOM, E.TIPUS -- (me dara solo una entrada por representacion) //puedo probar código sin el group by ni having by, después pongo el group by i finalmente los 2 y observo como se desenvolupa 
HAVING COUNT(*)=1 --(NOS SIRVE PARA SABER SI SOLO HAY 1 REPRESENTACION)

-- 55-Nom, adreça i ciutat dels recintes que únicament han realitzat un únic espectacle. 
SELECT R.NOM, R.ADREÇA, R.CIUTAT 
FROM ESPECTACLES E, RECINTES R 
WHERE E.CODI_RECINTE= R.CODI 
GROUP BY R.NOM, R.ADREÇA, R.CIUTAT 
HAVING COUNT(*)=1

-- 56-Nombre d'espectadors que ha tingut el teatre La Faràndula per a cada espectacle, ordenats de més a menys espectadors. S'ha de recuperar el nom de l'espectacle i el nombre d'espectadors. 
select e.nom, count(*) as NUm_espect 
from espectacles e, entrades en, recintes r 
where r.nom='La Faràndula' and e.codi_recinte=r.codi and en.codi_espectacle=e.codi group by e.nom
-- 57-DNI, nom i cognoms dels espectadors i codi d'espectacle dels espectadors que han adquirit una sola entrada d'un espectacle. 
SELECT ES.DNI, ES.NOM, ES.COGNOMS, EN.CODI_ESPECTACLE 
FROM ESPECTADORS ES, ENTRADES EN 
WHERE ES.DNI = EN.DNI_CLIENT 
GROUP BY ES.DNI, ES.NOM, ES.COGNOMS, EN.CODI_ESPECTACLE HAVING COUNT(*)=1

-- 58-DNI, nom i cognoms de l'espectador que ha adquirit més entrades per a qualsevol espectacle. 
SELECT ED.DNI, ED.Nom, ED.Cognoms FROM Espectadors ED, Entrades EN 
WHERE ED.DNI = EN.DNI_Client GROUP BY ED.DNI, ED.Nom, ED.Cognoms 
HAVING COUNT(*) >= ALL ( SELECT COUNT(*) FROM Espectadors ED2, Entrades EN2 WHERE ED2.DNI = EN2.DNI_Client GROUP BY ED2.DNI)

-- 59-Nom, tipus d'espectacle, data i hora de la representació amb més espectadors. 
SELECT ES.Nom, ES.Tipus, R.Data, TO_CHAR(R.Hora,'HH24:MI:SS') 
FROM Espectacles ES, Representacions R, Entrades EN 
WHERE R.Codi_Espectacle = ES.Codi AND EN.Codi_Espectacle = R.Codi_Espectacle AND EN.Data = R.Data AND EN.Hora=R.Hora 
GROUP BY ES.Nom, ES.Tipus, R.Data, R.Hora 
HAVING SUM(EN.Numero) >= ALL( SELECT SUM(EN2.Numero) 
FROM Representacions R2, Entrades EN2 WHERE R2.Codi_Espectacle=EN2.Codi_Espectacle AND R2.Data=EN2.Data AND R2.Hora=EN2.Hora 
GROUP BY R2.Codi_Espectacle, R2.Data, R2.Hora)

-- 60-Nom i tipus d'espectacle amb més espectadors. 
SELECT ES.Nom, ES.Tipus 
FROM Espectacles ES, Entrades EN 
WHERE EN.Codi_Espectacle = ES.Codi GROUP BY ES.Nom, ES.Tipus 
HAVING SUM(EN.Numero) >= ALL( SELECT SUM(EN2.Numero) FROM Representacions R2, Entrades EN2 WHERE R2.Codi_Espectacle= EN2.Codi_Espectacle AND R2.Data = EN2.Data AND R2.Hora=EN2.Hora GROUP BY R2.Codi_Espectacle)

-- 61-Nom, ciutat i capacitat del recinte amb més capacitat. 
SELECT RC.Nom, RC.Ciutat, SUM(ZR.Capacitat) AS Cap_Total 
FROM Recintes RC, Zones_Recinte ZR 
WHERE ZR.Codi_Recinte = RC.Codi 
GROUP BY RC.Nom, RC.Ciutat 
HAVING SUM(ZR.Capacitat) >= ALL ( SELECT SUM(Capacitat) FROM Zones_Recinte GROUP BY Codi_Recinte)

-- 62-Nom, ciutat i capacitat del recinte amb menys capacitat. 
SELECT RC.Nom, RC.Ciutat, SUM(ZR.Capacitat) AS Cap_Total 
FROM Recintes RC, Zones_Recinte ZR 
WHERE ZR.Codi_Recinte = RC.Codi GROUP BY RC.Nom, RC.Ciutat HAVING SUM(ZR.Capacitat) <= ALL ( SELECT SUM(Capacitat) FROM Zones_Recinte GROUP BY Codi_Recinte)

--63-Nom de l'espectacle teatre amb més espectadors. 
SELECT ES.Nom FROM Espectacles ES, Entrades EN 
WHERE ES.Tipus = 'Teatre' AND EN.Codi_Espectacle = ES.Codi 
GROUP BY ES.Nom HAVING COUNT(*) >= ALL ( SELECT COUNT(*) FROM Espectacles ES2, Entrades EN2 WHERE ES2.Tipus = 'Teatre' AND EN2.Codi_Espectacle=ES2.Codi GROUP BY ES2.Codi)

--64-Nom de l'espectacle musical amb menys espectadors. 
SELECT ES.NOM FROM ESPECTACLES ES, REPRESENTACIONS R, ENTRADES EN WHERE ES.CODI=R.CODI_ESPECTACLE AND R.CODI_ESPECTACLE=EN.CODI_ESPECTACLE AND R.DATA=EN.DATA AND R.HORA=EN.HORA and es.tipus='Musical' GROUP BY ES.NOM HAVING COUNT(*)<=(SELECT MIN(COUNT(*)) FROM ESPECTACLES ES1, REPRESENTACIONS R1, ENTRADES EN1 WHERE ES1.CODI=R1.CODI_ESPECTACLE AND R1.CODI_ESPECTACLE=EN1.CODI_ESPECTACLE AND R1.DATA=EN1.DATA AND R1.HORA=EN1.HORA and es1.tipus='Musical' GROUP BY ES1.NOM)

--65-Codi, nom i ciutat del recinte que ha tingut més representacions. 
SELECT R.Codi, R.Nom, R.Ciutat FROM Recintes R, Espectacles ES, Representacions RE WHERE ES.Codi_Recinte = R.Codi AND RE.Codi_Espectacle=ES.Codi GROUP BY R.Codi, R.Nom, R.Ciutat HAVING COUNT(*) >= ALL ( SELECT COUNT(*) FROM Representacions RE2, Espectacles ES2 WHERE RE2.Codi_Espectacle=ES2.Codi GROUP BY ES2.Codi_Recinte)

--66-Codi, nom i ciutat del recinte amb més espectadors. 
SELECT R.Codi, R.Nom, R.Ciutat FROM Recintes R, Entrades EN WHERE EN.Codi_Recinte = R.Codi GROUP BY R.Codi, R.Nom, R.Ciutat HAVING COUNT(*) >= ALL ( SELECT COUNT(*) FROM Entrades EN2 GROUP BY EN2.Codi_Recinte)

--68-Nom i ciutat dels espectadors que no han adquirit cap entrada de cap espectacle. 
SELECT ED.Nom, ED.Ciutat FROM Espectadors ED WHERE ED.DNI NOT IN ( SELECT DISTINCT EN.DNI_Client FROM Entrades EN)

--69-Nombre de representacions que es realitzen el 20 d'Octubre del 2011. SELECT SUM (COUNT(R.Codi_Espectacle))
FROM Representacions R WHERE R.Data = TO_DATE('20/10/2011','dd/mm/yyyy')

--70-Nom dels espectacles realitzats per La Joventut de la Farándula. 
SELECT DISTINCT ES.Nom FROM Espectacles ES, Representacions RE WHERE Interpret = 'La Joventut de la Faràndula' AND RE.Codi_Espectacle=ES.Codi

-- 71-Nom i aforament dels recintes de Barcelona. 
SELECT R.Nom, SUM(ZR.Capacitat) AS Num_Places FROM Recintes R, Zones_Recinte ZR WHERE Ciutat = 'Barcelona' AND ZR.Codi_Recinte=R.Codi GROUP BY R.Nom

-- 72-Aforament total dels recintes de Barcelona. 
SELECT SUM(ZR.Capacitat) AS Aforo_Total FROM Recintes R, Zones_Recinte ZR WHERE R.Ciutat='Barcelona' AND ZR.Codi_Recinte=R.Codi

-- 73-Aforament total de tots els recintes 
SELECT SUM(ZR.Capacitat) AS Aforo_Total FROM Recintes R, Zones_Recinte ZR WHERE ZR.Codi_Recinte=R.Codi

-- 74-Zona de recinte amb més capacitat i nom del recinte on es troba aquesta zona. 
SELECT ZR.Zona, R.Nom FROM Recintes R, Zones_Recinte ZR WHERE R.Codi = ZR.Codi_Recinte AND ZR.Capacitat >= ( SELECT MAX (ZR2.Capacitat) FROM Zones_Recinte ZR2)

--75-Nom de l'espectacle amb el preu de l'espectacle més car. 
SELECT ES.Nom FROM Espectacles ES, Preus_Espectacles PE WHERE ES.Codi = PE.Codi_Espectacle AND PE.Preu >= ( SELECT MAX (PE2.Preu) FROM Preus_Espectacles PE2)

--76-Nom de l'espectacle amb el preu de l'entrada més barat. 
SELECT DISTINCT ES.Nom 
FROM Espectacles ES, Preus_Espectacles PE 
WHERE ES.Codi = PE.Codi_Espectacle AND PE.Preu <= ( SELECT MIN (PE2.Preu) FROM Preus_Espectacles PE2)

--79-Promig d'ocupació dels espectacles de Barcelona (promig = total entrades venudes/ total seients). 
SELECT T1.Aforo_Venut / T2.Total_Aforo AS Promig 
FROM ( SELECT COUNT(*) AS Aforo_Venut FROM Recintes R, Entrades EN WHERE R.Ciutat = 'Barcelona' AND R.Codi = EN.Codi_Recinte) T1, ( SELECT COUNT(*) AS Total_Aforo FROM Recintes R, Seients S WHERE R.Ciutat = 'Barcelona' AND R.Codi = S.Codi_Recinte) T2

--81-Nombre d'espectadors per espectacle en promig (total entrades venudes/total espectacles). 
SELECT T1.Total_Espect / T2.Tot_Espect AS Promig 
FROM ( SELECT COUNT(*) AS Total_Espect FROM Entrades) T1, ( SELECT COUNT(*) AS Tot_Espect FROM Espectacles) T2

--85-Espectadors (DNI, nom i cognoms) que no han vist cap espectacle de El Tricicle. 
(SELECT E.DNI, E.NOM, E.COGNOMS 
FROM ESPECTADORS E) 
MINUS 
(SELECT DISTINCT ES1.DNI,ES1.NOM, ES1.COGNOMS 
FROM ESPECTACLES E1, ESPECTADORS ES1, REPRESENTACIONS R1,ENTRADES EN1 
WHERE E1.CODI=R1.CODI_ESPECTACLE AND R1.CODI_ESPECTACLE=EN1.CODI_ESPECTACLE AND EN1.DATA=R1.DATA AND EN1.HORA=R1.HORA AND EN1.DNI_CLIENT=ES1.DNI AND E1.INTERPRET='El Tricicle')

--89-Nom i codi del recinte del que s'han venut més entrades. 
SELECT R.Nom, R.codi 
FROM RECINTES R, ENTRADES E 
WHERE R.CODI=E.CODI_RECINTE 
GROUP BY R.CODI, R.NOM 
HAVING COUNT(*)>=(SELECT MAX(COUNT(*)) FROM ENTRADES E1 GROUP BY E1.CODI_RECINTE)

--90-Nom de l'intèrpret que ha fet més representacions i nombre de representacions. 
SELECT E.INTERPRET, COUNT(*) AS num_representacions 
FROM ESPECTACLES E, REPRESENTACIONS R 
WHERE E.CODI=R.CODI_ESPECTACLE 
GROUP BY E.INTERPRET HAVING COUNT(*)>=(SELECT (MAX(COUNT(*))) FROM ESPECTACLES E1, REPRESENTACIONS R1 WHERE E1.CODI=R1.CODI_ESPECTACLE GROUP BY E1.INTERPRET)

--91-Codi i nom del recinte on s'han fet més representacions i nombre de representacions. 
SELECT R.CODI, R.NOM, COUNT(*) AS num_representacions 
FROM RECINTES R, REPRESENTACIONS RE, ESPECTACLES E 
WHERE E.CODI=RE.CODI_ESPECTACLE AND R.CODI=E.CODI_RECINTE 
GROUP BY R.CODI, R.NOM 
HAVING COUNT(*)>=(SELECT MAX(COUNT(*)) FROM REPRESENTACIONS R1, ESPECTACLES E1 WHERE E1.CODI=R1.CODI_ESPECTACLE GROUP BY E1.CODI_RECINTE)

--97-Codi, nom i població dels recintes que tenen més d'una zona. 
SELECT Z.CODI_RECINTE, R.NOM, R.CIUTAT 
FROM RECINTES R, ZONES_RECINTE Z 
WHERE R.CODI=Z.CODI_RECINTE
GROUP BY z.CODI_RECINTE, R.NOM, R.CIUTAT 
HAVING COUNT(*)>1

--103-Espectador (DNI, nom i cognoms) que ha adquirit més entrades per a qualsevol espectacle. 
SELECT ES.DNI, ES.Nom, ES.Cognoms 
FROM Entrades EN, Espectadors ES 
WHERE ES.DNI=EN.DNI_Client 
GROUP BY ES.DNI,ES.Nom,ES.Cognoms 
HAVING COUNT(*) >= ALL ( SELECT COUNT(*) FROM Entrades EN2 GROUP BY EN2.DNI_Client)