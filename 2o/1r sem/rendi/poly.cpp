#include <iostream>
#include <iomanip>
using namespace std;

#ifndef REAL
#define REAL double
#endif

template <typename T>
T poly (T A[], int degree, T x)
{
  T result= A[0], xpwr= x;
  for (int i = 1; i <= degree; i++) {
    result = result + A[i] * xpwr;
    xpwr = x * xpwr;
  }
  return result;
}


template <typename T> 
T polyh (T A[], int degree, T x)
{
  T result = A[degree];
  for (int i = degree-1; i >= 0; i--)
    result = A[i] + x*result;
  return result;
}


int main(int argc, char **argv) 
{
  int  Option=0, N= 0, M= 0;
  REAL *Poly, *Input, R=0;

  if (argc != 4) {
    cout << "Three arguments are required: N, M, Option" << endl; 
    return 1;
  }

  N     = atoi(argv[1]);
  M     = atoi(argv[2]);
  Option= atoi(argv[3]);
  
  Poly  = new REAL[N+1];
  Input = new REAL[M];

  srand48(0);
  for (int i=0; i<N+1; i++)
    Poly[i]= drand48();

  for (int i=0; i<M; i++)
    Input[i]= drand48();
 
  switch(Option) 
  {
    case 0: 
      cout << "**************** DIRECT METHOD *****************\n";
      for (int i=0; i<M; i++)   
        Input[i] = poly<REAL> (Poly, N, Input[i]);
      break;

    case 1: 
      cout << "*************** HORNER's METHOD ****************\n";
      for (int i=0; i<M; i++)   
        Input[i] = polyh<REAL> (Poly, N, Input[i]);
      break;
  }

  for (int i=0; i<M; i++)
    R += Input[i];

  cout << "N= " << N << ", M= " << M << setprecision(16) << ", R = " << R << "\n"; 

  delete [] Poly;
  delete [] Input;
  return 0;
}
