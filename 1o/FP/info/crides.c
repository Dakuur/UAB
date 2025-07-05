#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    char buff[10000];
    char a[1000000];
    scanf("%s",a);
    
    int FileDes = open(a, O_RDONLY);
    if (FileDes == -1){
        printf("Arxiu no està al directort\n");
    }
    else{
        int Readstatus = read(FileDes, buff,10000);
        printf("%s", buff);
    }
}