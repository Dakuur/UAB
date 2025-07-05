#include <stdio.h>
#include <stdlib.h>

#define N 1000

int V[N];

void ordena(int vector[], int n){
/*Aquí heu d'escriure el codi que ordeni de forma ascendent
  els elements del vector
  
  El que es farà és (Selection sort):
    Començar des del primer element del vector (iteració 0)
    Buscar l'element del vector (des de l'element de l'iteració actual) amb el valor mínim
    Intercanviar el valor mínim pel valor de l'iteració actual: V[i] <-> V[min]
    Repetir els passos 2 i 3 fins arribar al final del vector
    */

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

/*  
    Funció auxiliar
*/
void imprimirVector(int vector[], int size) {
    printf("Vector: [");
    for (int i = 0; i < size; i++) {
        printf("%d", vector[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main(){
    init(V,N);

    if (!verifica(V,N)) printf("Inicialment l'array no està ordenat (ja ho sabiem però ens assegurem)\n");

    //imprimirVector(V, N);
    ordena(V,N);
    //imprimirVector(V, N);
    
    if(!verifica(V,N)) printf("Aquesta rutina d'ordenació no ordena! Try again!\n");
    else printf("Molt bé! Array ordenat!\n");
}