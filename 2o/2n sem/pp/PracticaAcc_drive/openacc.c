#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENACC
#include <openacc.h>
#endif

void laplace_init(float *in, int n) //Inicialització de la matriu, no accelerem aquesta funció ja que només es fa una vegada i es més costos fer la creació de la regió paral·lela que la acceleració que obtenim.
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, n*n*sizeof(float));
  for (i=0; i<n; i++) {
    float V = in[i*n] = sinf(pi*i / (n-1));
    in[ i*n+n-1 ] = V*expf(-pi);
  }
}

int main(int argc, char** argv)
{
  int n = 4096; //llargada files i columnes matriu
  int iter_max = 1000; //iteracions maximes
  float *A, *temp;

  const float tol = 1.0e-5f;
  float error= 1.0f;

  // get runtime arguments
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }

  A    = (float*) malloc( n*n*sizeof(float) );
  temp = (float*) malloc( n*n*sizeof(float) );

  //  set boundary conditions
  laplace_init (A, n);
  laplace_init (temp, n);
  A[(n/128)*n+n/128] = 1.0f; // set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n",
         n, n, iter_max );

  int iter = 0;
  float t;
  #pragma acc data copy(A[:n*n]) create(temp[:n*n]) //Copiem la matriu A en la seva totalitat i reservem espai (tamany de la matriu Temp)  a la GPU per als càlculs posteriors.
  {
  while ( error > tol*tol && iter < iter_max )
  {
    //Single
    iter++;
    //Funció laplace step
    int i, j;
    error = 0.0f;
   
    #pragma acc parallel loop gang vector reduction(max: error) collapse(2) //regió paral·lela: indiquem el loop pel bucle, el reduction pendrà el valor global error de la variable error local que sigui més gran 
    for ( j=1; j < n-1; j++ )  							//de totes les calculades i el collapse fusiona els 2 bucles. El vector força l'ús d'instruccions SIMD. 
      for ( i=1; i < n-1; i++ )
      {
        temp[j*n+i]= (A[j*n+i+1] + A[j*n+i-1] + A[(j-1)*n+i] + A[(j+1)*n+i]) * 0.25f; //funció stencil
        t = fabsf(A[j*n+i] - temp[j*n+i]); //funció maxerror
        error = t>error ? t: error;	//si t és més gran que error retorna t, sinó retorna error.
      }
    //acaba laplace
    
    //Actualitzem els valors de la matriu A per la següent iteració.
    #pragma acc parallel loop collapse(2) // regió paral·lela: indiquem el loop pel bucle i el collapse per fusionar els dos fors.
    for(int j = 1; j < n - 1; j++){
	for( int i = 1; i < n - 1; i++){
		A[j*n + i] = temp[j*n + i];
        }
    }
    

    
  }
  }
  //acaba while
  error = sqrtf( error );
  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n/128, n/128, A[(n/128)*n+n/128]);

  free(A); free(temp);
}

