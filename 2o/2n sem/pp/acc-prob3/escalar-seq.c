#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<sys/time.h>

#define N 1024*1024

float V1[N], V2[N];

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

// Inicialitza dos vectors d'enters amb nombres aleatoris entre -99.0 i +99.0
void init(float V1[], float V2[], unsigned long n){

    for( unsigned long i = 0; i < n; i++ ){ 
        V1[i] = ((i%2) ? -1 : 1)*(1.0*rand()/(1.0*RAND_MAX))*100;
        V2[i] = ((i%3) ? -1 : 1)*(1.0*rand()/(1.0*RAND_MAX))*100;
    }
}

float multpunt(float V1[], float vV2[], unsigned long n){
    float sum = 0.0;

    for( unsigned int i = 0; i < n; i++) 
        sum += V1[i]*V2[i];

    return sum;
}

int main(){
    float res; 

    init(V1, V2, N);

    double ini = cp_Wtime();
    res = multpunt(V1, V2, N);
    double tot = cp_Wtime() - ini;

    printf("Resultat del producte escalar de V1 i V2: %f i ha trigat %lf sec\n", res, tot);

    return 0;
}
