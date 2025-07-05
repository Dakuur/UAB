#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    char a[100];
    char buff[10000];
    FILE *fitxer;

    scanf("%s", a);
    int FileDes = open(a, O_RDONLY);
    fitxer = fopen(a, "r");
    fseek(fitxer, 0L, SEEK_END);
    int x = ftell(fitxer);
    fclose(fitxer);

    if (FileDes == -1){
        printf("Arxiu no està al directort\n");
    } else{
        int Readstatus = read(FileDes, buff,x);
        int FileDescopy = open("copia.ft", O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
        int Writestatus = write(FileDescopy, buff, x);
        close(FileDes);
        close(FileDescopy);
    }
}