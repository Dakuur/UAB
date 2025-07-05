#include <stdio.h>
#include <stdlib.h>

float VectSum( float V[], float W[], int N )
{
  int i;
  float sum=0;
  for (i=0; i<N; i++)
    sum = sum + V[i] + W[i];
  return sum;
}

float VectSum_V1( float V[], float W[], int N )
{
  int i;
  float sum=0;
  for (i=0; i<N; i++)
    sum = sum + (V[i] + W[i]);
  return sum;
}

float VectSum_V2( float V[], float W[], int N )
{
  int i;
  float sum=0;
  for (i=0; i<N; i+=2)
    sum = sum + V[i] + W[i] + V[i+1] + W[i+1];
  return sum;
}

float VectSum_V3( float V[], float W[], int N )
{
  int i;
  float sum=0;
  for (i=0; i<N; i+=2)
    sum = sum + (V[i] + W[i] + V[i+1] + W[i+1]);
  return sum;
}

int main(int argc, char **argv) 
{
  int N= 5000, R= 2000000, Op=0;
  float  *V, *W;

  // get runtime arguments 
  if (argc>1) { N  = atoi(argv[1]); }
  if (argc>2) { R  = atoi(argv[2]); }
  if (argc>3) { Op = atoi(argv[3]); }

  printf("Lesson 1. E1: suma de los elementos de 2 vectores.\n");
  printf("   Datos reales de precisión simple.\n");
  printf("   N= %d. Versión= %d. Repeticiones=%d\n", N, Op, R);

  V = calloc(N, sizeof(float));
  W = calloc(N, sizeof(float));

  for (int i=0; i<N; i+=2)
  {
    V[i] = 1; V[i+1]= -1;
    W[i] = 1; W[i+1]= -1;
  }

  switch (Op) {
    case 0:
      for (int t=0; t<R; t++) 
        V[0] =  VectSum ( V, W, N );
      break;
    case 1:
      for (int t=0; t<R; t++) 
        V[0] = VectSum_V1 ( V, W, N );
      break;
    case 2:
      for (int t=0; t<R; t++) 
        V[0] = VectSum_V2 ( V, W, N );
      break;
    case 3:
      for (int t=0; t<R; t++) 
        V[0] = VectSum_V3 ( V, W, N );
      break;
   }

  printf("Sum = %e\n", V[0]);

  free(V);  free(W);
}