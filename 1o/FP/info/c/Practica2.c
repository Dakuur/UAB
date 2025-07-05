#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define N 512

float Mat[N][N], MatDD[N][N];
float V1[N], V2[N], V3[N], V4[N];

float InitData()
{
    int i, j;
    srand(8824553);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            Mat[i][j] = (((i * j) % 3) ? -1 : 1) * (100.0 * (rand() / (1.0 * RAND_MAX)));
            if ((abs(i - j) <= 3) && (i != j))
                MatDD[i][j] = (((i * j) % 3) ? -1 : 1) * (rand() / (1.0 * RAND_MAX));
            else if (i == j)
                MatDD[i][j] = (((i * j) % 3) ? -1 : 1) * (10000.0 * (rand() / (1.0 * RAND_MAX)));
            else
                MatDD[i][j] = 0.0;
        }
    }
    for (i = 0; i < N; i++)
    {
        V1[i] = (i < N / 2) ? (((i * j) % 3) ? -1 : 1) * (100.0 * (rand() / (1.0 * RAND_MAX))) : 0.0;
        V2[i] = (i >= N / 2) ? (((i * j) % 3) ? -1 : 1) * (100.0 * (rand() / (1.0 * RAND_MAX))) : 0.0;
        V3[i] = (((i * j) % 5) ? -1 : 1) * (100.0 * (rand() / (1.0 * RAND_MAX)));
    }
}

void PrintVect(float vect[N], int from, int numel)
{
    int i;
    for (i = from; i < numel + from; i++)
    {
        printf("%f ", vect[i]);
    }
}

void PrintRow(float Mat[N][N], int row, int from, int numel)
{
    int i;
    for (i = from; i < numel + from; i++)
    {
        printf("%f", Mat[row][i]);
    }
}

void MultEscalar(float vect[N], float vectres[N], float alfa)
{
    int i;
    for (i = 0; i < N; i++)
    {
        vectres[i] = vect[i] * alfa;
    }
}

float Scalar(float vect1[N], float vect2[N])
{
    float res = 0;
    int i;
    for (i = 0; i < N; i++)
    {
        res += vect1[i] * vect2[i];
    }
    return res;
}

float Magnitude(float vect[N])
{
    float magnitud;
    int i;
    for (i = 0; i < N; i++)
    {
        magnitud += pow(vect[i], 2);
    }
    magnitud = sqrt(magnitud);
    return magnitud;
}

float Infininorm(float Mat[N][N])
{
    float max = 0.0;
    int i, j;
    for (i = 0; i < N; i++)
    {
        float suma = 0;
        for (j = 0; j < N; j++)
        {
            suma += abs(Mat[i][j]);
        }
        if (suma > max)
        {
            max = suma;
        }
    }
    return max;
}

float Onenorm(float M[N][N])
{
    float max = 0.0;
    int i, j;
    for (j = 0; j < N; j++)
    {
        float suma = 0;
        for (i = 0; i < N; i++)
        {
            suma += abs(M[i][j]);
        }
        if (suma > max)
        {
            max = suma;
        }
    }
    return max;
}

float NormFrobenius(float M[N][N])
{
    float res, sumaquadrats;
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sumaquadrats += pow(M[i][j], 2);
        }
    }
    res = sqrt(sumaquadrats);
    return res;
}

int Ortogonal(float vect1[N], float vect2[N])
{
    float res;
    int i;
    for (i = 0; i < N; i++)
    {
        res += vect1[i] * vect2[i];
    }
    if (res == 0)
        return 1;
    return 0;
}

float Projection(float vect1[N], float vect2[N], float vectres[N])
{
    float escalar = 0;
    int i;
    for (i = 0; i < N; i++)
    {
        escalar += vect1[i] * vect2[i];
    }

    float magnitud = 0;
    float sumaquadrats = 0;
    for (i = 0; i < N; i++)
    {
        sumaquadrats += pow(vect2[i], 2);
        magnitud = sqrt(sumaquadrats);
    }
    float alfa = escalar / pow(magnitud, 2);

    for (i = 0; i < N; i++)
    {
        vectres[i] = vect2[i] * alfa;
    }

    return vectres[N];
}

int DiagonalDom(float M[N][N])
{
    float sumalinia = 0;
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sumalinia += abs(M[i][j]);
        }

        if ((abs(M[i][j]) >= (sumalinia - M[i][i])))
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

int main()
{
    InitData();
    // Pas A
    // a
    printf("V1:\n");
    PrintVect(V1, 0, 10);
    PrintVect(V1, 256, 10);
    printf("\nV2:\n");
    PrintVect(V2, 0, 10);
    PrintVect(V2, 256, 10);
    printf("\nV3:\n");
    PrintVect(V3, 0, 10);
    PrintVect(V3, 256, 10);
    printf("\n");
    // b
    printf("\nComença el PrintRow de Mat\n");
    PrintRow(Mat, 0, 0, 10);
    printf("\n");
    PrintRow(Mat, 100, 0, 10);
    // c
    printf("\nComença el PrintRow de MatDD\n");
    PrintRow(MatDD, 0, 0, 10);
    printf("\n");
    PrintRow(MatDD, 100, 90, 10);
    // Pas B
    // a
    printf("\nLa infininorma de Mat és %f\n", Infininorm(Mat));
    printf("La infininorma de MatDD és %f\n", Infininorm(MatDD));
    // b
    printf("La norma ú de Mat és %f\n", Onenorm(Mat));
    printf("La norma ú de MatDD és %f\n", Onenorm(MatDD));
    // c
    printf("La norma de Frobenius de Mat és %f\n", NormFrobenius(Mat));
    printf("La norma de Frobenius de MatDD és %f\n", NormFrobenius(MatDD));
    // d
    int value, value2;
    value = DiagonalDom(Mat);
    value2 = DiagonalDom(MatDD);
    if (value == 1)
    {
        printf("Mat és diagonal dominant\n");
    }
    else
    {
        printf("Mat no és diagonal dominant\n");
    }
    if (value2 == 1)
    {
        printf("MatDD és diagonal dominant\n");
    }
    else
    {
        printf("MatDD no és diagonal dominant\n");
    }
    // Pas C
    printf("V1*V2 = %f\n", Scalar(V1, V2));
    printf("V1*V3 = %f\n", Scalar(V1, V3));
    printf("V2*V3 = %f\n", Scalar(V2, V3));
    // Pas D
    printf("Magnitud V1: %f\n", Magnitude(V1));
    printf("Magnitud V2: %f\n", Magnitude(V2));
    printf("Magnitud V3: %f\n", Magnitude(V3));
    // Pas E
    int comb1, comb2, comb3;
    comb1 = Ortogonal(V1, V2);
    comb2 = Ortogonal(V1, V3);
    comb3 = Ortogonal(V2, V3);
    if (comb1 == 1)
    {
        printf("V1 i V2 són ortogonals\n");
    }
    else
    {
        printf("V1 i V2 no són ortogonals\n");
    }
    if (comb2 == 1)
    {
        printf("V1 i V3 són ortogonals\n");
    }
    else
    {
        printf("V1 i V3 no són ortogonals\n");
    }
    if (comb3 == 1)
    {
        printf("V2 i V3 són ortogonals\n");
    }
    else
    {
        printf("V2 i V3 no són ortogonals\n");
    }
    // Pas F
    printf("V3*2.0 =\n ");
    float V7[N];
    MultEscalar(V3, V7, 2.0);
    PrintVect(V7, 0, 10);
    printf("\n");
    PrintVect(V7, 256, 10);
    // Pas G
    printf("\nLa projecció d'V1 sobre V2 és: ");
    float V6[N];
    Projection(V1, V2, V6);
    PrintVect(V6, 0, 10);
    printf("\n");
    float V5[N];
    printf("La projecció d'V2 sobre V3 és: ");
    Projection(V2, V3, V5);
    PrintVect(V5, 0, 10);
    printf("\n");
}