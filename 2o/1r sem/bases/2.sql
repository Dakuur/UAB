SELECT DISTINCT nia_professor, tipus_docencia
FROM DOCENCIA
WHERE CODI_ASSIGNATURA = 104350
AND tipus_docencia IN ('Teoria', 'Problemes')