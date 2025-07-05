'Codi i nom de l’assignatura amb el nombre màxim d’alumnes matriculats
en una assignatura durant el curs 2020-21.
Atributs de sortida: Codi i nom de l’assignatura, nombre d’alumnes'

SELECT M.CODI_ASSIGNATURA AS CODI, A.NOM, COUNT(*)
FROM MATRICULACIO M
JOIN ASSIGNATURES A
ON A.CODI = M.CODI_ASSIGNATURA
WHERE M.CURS = '2020-21'
GROUP BY M.CODI_ASSIGNATURA, A.NOM
HAVING COUNT(*) >= ALL
(
    SELECT COUNT(*) FROM MATRICULACIO M WHERE M.CURS = '2020-21' GROUP BY M.CODI_ASSIGNATURA
)