#include <stdio.h>
#include <stdlib.h>

#define N 1000

int *V;

void ordena(int vector[], int n){
/*Aquí heu de copiar (sense canvis) el codi que heu programat per l'exercici preliminar*/  
    for ( int i = 0; i < n; i++ ){ //Iterem per element a intercanviar
        int min_i = i;
        float min_v = vector[min_i];
        for ( int x = i; x < n; x++ ){ //Iterem per element per a trobar el mínim
            if ( vector[x] < vector[min_i] ) {
                min_i = x; // Index minim actual
            }
        }
        int temp = vector[i];
        vector[i] = vector[min_i];
        vector[min_i] = temp;
    }
}

/* 
    Funció que verifica si l'array rebut com a paràmetre està
    ordenat (retorna 1) o no (retorna 0)
*/
int verifica(int vector[], int n){
    for( int i = 1; i < n; i++ ) 
        if ( vector[i] < vector[i-1] ) return 0;
    return 1;
}

/*
    Funció que inicialitza l'array rebut com a paràmetre amb 
    valors enters aleatòris en el rang (-500,500)
*/
void init(int vector[], int n){
    for( int i = 0; i < n; i++ )
        vector[i] = ( i%10 ) ? random()%500 : -1*random()%500;
}

int main(int argc, char *argv[]){
    /*
        Aquí heu d'escriure el codi que verifiqui que l'usuari ha proporcionat el
        nombre d'elements a ordenar i que aquest nombre és > 2
        En cas de que l'usuari no proporcioni el nombre d'elements o aquest sigui
        <=2 es farà servir la constant N.
    */
    int mida;

    if (argc < 2) {
        printf("Argument no proporcionat.\n");
        mida = N;
    }
    else if (*argv[1] <= 2) {
        printf("Argument donat igual o menor a 2.\n");
        mida = N;
    }
    else {
        mida = *argv[1];
    }

    // Ara cal reservar la memòria de l'array, abans de fer-lo servir
    V = (int *)malloc(mida * sizeof(int));

    init(V,mida);

    if (!verifica(V,mida)) printf("Inicialment l'array no està ordenat (ja ho sabiem però ens assegurem)\n");

    ordena(V,mida);

    if(!verifica(V,mida)) printf("Aquesta rutina d'ordenació no ordena! Try again!\n");
    else printf("Molt bé! Array ordenat!\n");

    // Abans d'acabar cal alliberar la memòria reservada per l'array
    free(V);
}