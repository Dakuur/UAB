#include <stdio.h>

int pordos(int numero){
    numero *= 2;
    return numero;
}

int entredos(int numero){
    numero /= 2;
    return numero;
}

void main(){
    int num;
    printf("Introdueix nombre: \n");
    scanf("%d", &num);
    printf("1 - Multiplicar per 2\n2 - Dividir per 2\n\nQue vols fer?: \n");
    int opcio, res;
    scanf("%d", &opcio);
    if (opcio == 1){
        res=pordos(num);
        printf("Resultat: %d", res);
    }
    else if (opcio == 2){
        res=entredos(num);
        printf("Resultat: %d", res);
    }
    else{
        printf("Opcio no valida");
    }
}