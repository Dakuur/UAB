from time import time

def Sudoku(Tablero):


    ########## COMPLETA TU CODIGO AQUÍ ##########


    return Tablero

Tablero=   [[0,8,0,  0,0,2,  0,3,0],
            [0,4,0,  1,3,0,  0,2,0],
            [0,0,0,  7,0,0,  0,0,9],

            [0,0,0,  8,0,0,  6,5,3],
            [0,2,0,  0,4,5,  0,0,8],
            [5,6,0,  0,0,3,  2,4,0],

            [4,0,0,  0,0,0,  5,0,7],
            [7,0,2,  0,0,0,  0,8,4],
            [0,0,0,  4,0,0,  0,0,2]]

start_time = time()

print(Sudoku(Tablero))

elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)