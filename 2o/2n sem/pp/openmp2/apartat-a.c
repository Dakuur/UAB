#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

#define N 100

int main(int argc, char* argv[]){

    int i,j,z;
    float sum_d = 0; 
    float  a[N], b[N], c[N], d[N];

    /* Init matrius, no cal modificar */ 
    for( i = 0; i < N; i++ )
        for( j = 0; j < N; j++ ){
            a[i] = i*0.25; b[i] = (2*i)*0.5; 
            c[i] = 0; d[i] = 0; 
        }
    
    /* Fi init matrius */




    /* CODI A MODIFICAR : */

    #pragma omp parallel num_threads(4) if (N >= 100) //Fragment paral·lelitzat
    //(si es compleixen les condicions). Fixat a 4 threads
    {
        
        #pragma omp sections private(i) //Execució en paralel per seccions (una per thread).
        //Privatitzem la i ja que comparteixen aquesta variable pel bucle for

        {

            #pragma omp section //Executada en un thread qualsevol (no utilitzat per cap altre secció)s
            {
                printf("Section 1 is executed by thread %d.\n", omp_get_thread_num());
                for( i = 0; i < N; i++){
                    d[i] = i * b[i];
                }
                printf("Finished Section 1 by core %d.\n", omp_get_thread_num());
            }

            #pragma omp section //Executada en un thread qualsevol (no utilitzat per cap altre secció)
            {
                printf("Section 2 is executed by thread %d.\n", omp_get_thread_num());
                for( i = 0; i < N; i++){
                    for( j = 0; j < N; j++){
                        c[i] += a[i] + b[i];
                    }
                }
                printf("Finished Section 2 by core %d.\n", omp_get_thread_num());
            }
        } //Fi de les  execucions en seccions. Esperen a acabar totes per seguir amb el codi

        #pragma omp for reduction(+:sum_d) // For paralel amb reducció de suma
        //(els 4 threads que l’executen comparteixen un resultat en la variable
        //sum_d que al final sumarán fent servir reducció
        for( i = 0; i < N; i++ ){ 
                sum_d += d[i]+c[i]; //Arrays d i c ja acabats de computar en les seccions
        }
        printf("Finished Section 3 by core %d.\n", omp_get_thread_num());
    }

    /* FI DEL CODI A MODIFICAR : */




    /* Checksum : Printa el resultat */
    // - No cal modificar
    float checksum_c = 0, checksum_d = 0, checksum_C = 0; 
    for( i = 0; i <  N; i++ ) {
        checksum_d += d[i];
        checksum_c += c[i];
    }
    
    printf("Printing the results : \n");
    printf("array c: %f\n", checksum_c); 
    printf("array d: %f\n", checksum_d); 
    printf("sum_d: %f\n", sum_d); 

}
