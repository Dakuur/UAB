'NIA, nom i cognoms dels alumnes que s�han matriculat a totes les
assignatures que imparteix el professor Enric Mart�.'

(SELECT A.NIA, A.NOM, A.COGNOMS, M.CODI_ASSIGNATURA
FROM MATRICULACIO M
JOIN ALUMNES A
ON A.NIA = M.NIA_ALUMNE
JOIN DOCENCIA D
ON D.CODI_ASSIGNATURA = M.CODI_ASSIGNATURA
JOIN PROFESSORS P
ON P.NIA = D.NIA_PROFESSOR
WHERE P.NOM = 'Enric' AND P.COGNOMS LIKE 'Mart�')
/
(SELECT D.CODI_ASSIGNATURA
FROM DOCENCIA D
JOIN PROFESSORS P
ON P.NIA = D.NIA_PROFESSOR
WHERE P.NOM = 'Enric' AND P.COGNOMS LIKE 'Mart�');
