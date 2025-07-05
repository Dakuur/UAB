stim 

#include <stdio.h>

int doble(num){
    int pordos = num*2;
    return pordos;
}

void main(){
    printf("Introdueix número: ");
    int numero;
    scanf("%d", &numero);
    int imprimir = doble(numero);
    printf("%d", imprimir);
}

void main()
{
    char nom[5];
    int edat;
    printf("Introdueix el teu nom: ");
    scanf("%s", &nom);
    printf("Introdueix la teva edat: ");
    scanf("%d", &edat);
    printf("Et dius %s i tens %d anys\n", nom, edat);
    int restant = 100-edat;
    printf("Et queden %d anys\n", restant);
    int i = 0;
    while(i < 5){
        char lletra = nom[i];
        printf("%c\n",lletra);
        i = i +1;
    }
}
*/