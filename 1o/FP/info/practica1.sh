#!/bin/bash

#corvertim l'arxiu a csv ja que l'original és cvs
cp titles.cvs titles.csv

#PAS 1
#eliminar files que no siguin tmXXXXX or tsXXXXX

grep -v  '^tm[0-9]\|^ts[0-9]' titles.csv | wc -l > num_titlesmal.csv
printf "Línies eliminades per identificador invàlid: ";  cat num_titlesmal.csv
grep '^tm[0-9]\|^ts[0-9]' titles.csv > pas1.csv


#PAS 2
#que el títol comenci per alfanumèric o ", #, ', ¿, ¡
awk -F "," '$2~/^[[[:alnum:]]|#|¿|¡|"]/' pas1.csv > pas2.csv
awk -F "," "$2/^'/" pas1.csv >> pas2.csv


#PAS 3
#separar en movies i shows
awk -F "," '$3~/MOVIE/' pas2.csv > Movies.csv
awk -F "," '$3~/SHOW/' pas2.csv > Shows.csv

cat Movies.csv > total.csv; cat Shows.csv >> total.csv

printf "Nombre d'elements original: "; wc -l < pas2.csv
printf "Nombre de películes: "; wc -l < Movies.csv
printf "Nombre de shows: "; wc -l < Shows.csv
printf "Nombre total: "; wc -l < total.csv


#PAS 4
#eliminar linies sense dades de imdb o tmdb
awk -F "," '!$12 || !$13 || !$14 || !$15' Movies.csv > no_scores.csv
awk -F "," '!$12 || !$13 || !$14 || !$15' Shows.csv >> no_scores.csv
printf "Línies eliminades per columna invàlida: "; wc -l < no_scores.csv

awk -F "," '$12 && $13 && $14 && $15' Movies.csv > pas4_movies.csv
awk -F "," '$12 && $13 && $14 && $15' Shows.csv > pas4_shows.csv


#PAS 5
#noves columnes

#movies
#calculem els maxims a cada arxiu
imdb_max=$(awk -F "," '{print $13}' pas4_movies.csv | sort -n | tail -1)
tmdb_max=$(awk -F "," '{print $14}' pas4_movies.csv | sort -n | tail -1)

#calculem i introduim les noves columnes
awk -v imdb_max="$imdb_max" -v tmdb_max="$tmdb_max" -F "," 'BEGIN {OFS = ","} {$16=$12*$13/imdb_max; $17=$15*$14/tmdb_max; print$0}' pas4_movies.csv > pas5_movies.csv

#shows
#calculem els maxims a cada arxiu
imdb_max=$(awk -F "," '{print $13}' pas4_shows.csv | sort -n | tail -1)
tmdb_max=$(awk -F "," '{print $14}' pas4_shows.csv | sort -n | tail -1)

#calculem i introduim les noves columnes
awk -v imdb_max="$imdb_max" -v tmdb_max="$tmdb_max" -F "," 'BEGIN {OFS = ","} {$16=$12*$13/imdb_max; $17=$15*$14/tmdb_max; print$0}' pas4_shows.csv > pas5_shows.csv

cp pas5_movies.csv Movies.csv
cp pas5_shows.csv Shows.csv

rm pas5_movies.csv pas5_shows.csv no_scores.csv pas2.csv num_titlesmal.csv pas1.csv pas4_movies.csv pas4_shows.csv total.csv titles.csv


#PAS 6

#imdb
printf "\nIMDB:\n"
#puntuació
sort -t, -nk12 Movies.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $12}'
sort -t, -nk12 Shows.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $12}'

#popular
sort -t, -nk13 Movies.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $13}'
sort -t, -nk13 Shows.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $13}'

#fiabilitat
sort -t, -nk16 Movies.csv | head -1 | awk -F "," '{print $1 "," $2 "," $9 "," $16}'
sort -t, -nk16 Shows.csv | head -1 | awk -F "," '{print $1 "," $2 "," $9 "," $16}'

#tmdb
printf "\nTMDB:\n"
#puntuació
sort -t, -nk15 Movies.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $15}'
sort -t, -nk15 Shows.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $15}'

#popular
sort -t, -nk14 Movies.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $14}'
sort -t, -nk14 Shows.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $14}'

#fiabilitat
sort -t, -gk17 Movies.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $17}'
sort -t, -gk17 Shows.csv | tail -1 | awk -F "," '{print $1 "," $2 "," $9 "," $17}'