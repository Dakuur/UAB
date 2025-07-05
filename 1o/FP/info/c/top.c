#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const int N = 512;

void InitData(){
    float Mat[N][N], MatDD[N][N], V1[N], V2[N], V3[N], V4[N];
    int i,j;
    srand(8824553);
    for( i = 0; i < N; i++ ){
        for( j = 0; j < N; j++ ){
            Mat[i][j]=(((i*j)%3)?-1:1)*(100.0*(rand()/(1.0*RAND_MAX)));
            if ( (abs(i - j) <= 3) && (i != j))
                MatDD[i][j] = (((i*j)%3) ? -1 : 1)*(rand()/(1.0*RAND_MAX));
            else if ( i == j )
                MatDD[i][j]=(((i*j)%3)?-1:1)*(10000.0*(rand()/(1.0*RAND_MAX)));
            else MatDD[i][j] = 0.0;
        }
    }
    for( i = 0; i < N; i++ ){
        V1[i]=(i<N/2)?(((i*j)%3)?-1:1)*(100.0*(rand()/(1.0*RAND_MAX))):0.0;
        V2[i]=(i>=N/2)?(((i*j)%3)?-1:1)*(100.0*(rand()/(1.0*RAND_MAX))):0.0;
        V3[i]=(((i*j)%5)?-1:1)*(100.0*(rand()/(1.0*RAND_MAX)));
    }
}


void PrintVect( float vect[N], int from, int numel ){
    int i;
    for (i = from; i < numel+from; i++) {
        printf("%f ", vect[i]);
    }
}


void PrintRow( float Mat[N][N], int row, int from, int numel ){
    int i;
    for (i = from; i < numel+from; i++) {
        printf("%f ", Mat[row][i]);
    }
}


void MultEscalar( float vect[N], float vectres[N], float alfa ){
    int i;
    for (i = 0; i < N; i++) {
        vectres[i]=vect[i]*alfa;
    }
}


float Scalar(float vect1[N], float vect2[N]){
    float res=0;
    int i;
    for (i = 0; i < N; i++) {
        res += vect1[i]*vect2[i];
    }
    return res;
}


float Magnitude( float vect[N] ){
    float magnitud;
    int i;
    for (i = 0; i < N; i++) {
        magnitud += pow(vect[i],2);
    }
    return magnitud;
}


float Infininorm(float Mat[N][N] ){
    float max = 0.0;
    int i, j;
    for (i = 0; i < N; i++) {
        float suma = 0;
        for (j = 0; j < N; j++){
            suma += abs(Mat[i][j]);
        }
        if (suma > max){
            max = suma;
        }
    }
    return max;
}


float Onenorm( float M[N][N] ){
    float max = 0.0;
    int i, j;
    for (j = 0; j < N; j++) {
        float suma = 0;
        for (i = 0; i < N; i++){
            suma += abs(M[i][j]);
        }
        if (suma > max){
            max = suma;
        }
    }
    return max;
}


float NormFrobenius( float M[N][N] ){
    float res, sumaquadrats;
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++){
            sumaquadrats += pow(M[i][j],2);
        }
    }
    res = sqrt(sumaquadrats);
    return res;
}


int Ortogonal(float vect1[N], float vect2[N]){
    float res=0;
    int i;
    for (i = 0; i < N; i++) {
        res += vect1[i]*vect2[i];
    }
    if (res==0){
        return 1;
    }
    else {
    return 0;
    }
}


void Projection( float vect1[N], float vect2[N], float vectres[N] ){
    float res=0;
    int i;
    for (i = 0; i < N; i++) {
        res += vect1[i]*vect2[i];
    }
    float magnitud;
    for (i = 0; i < N; i++) {
        magnitud += pow(vect2[i],2);
    }
    int pas1;
    pas1 = res/magnitud;
    for (i = 0; i < N; i++) {
        vectres[i]=vect2[i]*pas1;
    }
    return vectres[N];
}


int DiagonalDom( float M[N][N] ){
    float sumalinia = 0;
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N/2; j++){
            sumalinia += abs(M[i][j]);
        }
        if (sumalinia > abs(M[i][i])) {return 0;}
    }
    return 1;
}