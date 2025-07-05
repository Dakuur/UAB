SELECT DISTINCT nia_alumne
FROM matriculacio
WHERE codi_assignatura IN (104350, 104351)
AND (convocatoria_1 > 1 OR convocatoria_2 > 1)
AND (curs = '2022-23')