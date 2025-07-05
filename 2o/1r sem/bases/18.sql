'Nom i cognoms dels professors de teoria, amb l’assignatura que li donen,
que té l’alumne Jaume Camps Torrella, ordenats per nom d’assignatura.'

SELECT P.NOM, P.COGNOMS, ASS.NOM
FROM PROFESSORS P, DOCENCIA D, MATRICULACIO M, ALUMNES A, ASSIGNATURES ASS
WHERE P.NIA = D.NIA_PROFESSOR
AND D.CODI_ASSIGNATURA = M.CODI_ASSIGNATURA
AND M.NIA_ALUMNE = A.NIA
AND ASS.CODI = D.CODI_ASSIGNATURA
AND A.NOM = 'Jaume' AND A.COGNOMS =  'Camps Torrella'
AND D.TIPUS_DOCENCIA = 'Teoria'
ORDER BY ASS.NOM