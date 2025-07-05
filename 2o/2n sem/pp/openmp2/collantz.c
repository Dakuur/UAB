#include <stdio.h>
#include <stdlib.h>

unsigned long f(unsigned long x){
        return (x%2) ? 3*x + 1 : x/2;
}

int stop( unsigned long x ){
        int cont = 0;

        while ( x > 1 ){ x = f(x); cont++; }
        return cont;
}

int main( int argc, char *argv[] ){

        unsigned long n, i, Num;
        unsigned len, Max = 0, id;

        if ( argc < 2 ){ printf("Error: cal indicar el número natural\n" ); exit(1); }

        if ((n = atoi(argv[1])) <= 0 ) { printf("Error: el número ha de ser un natural (> 0)\n" ); exit(1); }

        #pragma omp parallel num_threads(6) private(len) shared(Max, Num) if (n >= 1000000)
        {
                printf("Using thread: %d.\n", omp_get_thread_num());

                unsigned long private_i;
                unsigned private_max = 0;
                unsigned private_num = 0;

                #pragma omp for schedule(guided)
                for (private_i = 1; private_i <= n; private_i++){
                len = stop(private_i);
                if (len > private_max) {
                        private_max = len;
                        private_num = private_i;
                }
                }

                #pragma omp critical
                {
                if (private_max > Max) {
                        Max = private_max;
                        Num = private_num;
                }
                }
        }
        printf("\nEl nombre menor que %lu amb temps d'auturada més alt és %lu amb %u passos\n", n, Num, Max );
}