# Nom: David Morillo Massagué
# NIU: 1666540

from time import time
from typing import List


class SudokuGame:

    def __init__(self, board: List[List]):
        self.board = board

    def find_empty(self, pos: list):
        board = self.board

        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    pos[0] = row
                    pos[1] = col
                    return True
        return False

    def in_row(self, row: int, num: int):
        board = self.board
        row_values = board[row]

        for value in row_values:
            if value == num:
                return False

        return True

    def in_col(self, col: int, num: int):
        board = self.board
        column_values = [row[col] for row in board]

        for value in column_values:
            if value == num:
                return False

        return True

    def in_box(self, row: int, col: int, num: int):
        board = self.board

        for i in range(3):
            for j in range(3):
                if board[i + row][j + col] == num:
                    return False
        return True

    def valid(self, row: int, col: int, num: int):

        if self.in_row(row, num):
            if self.in_col(col, num):
                return self.in_box(row - row % 3, col - col % 3, num)
            return False
        return False

    def __str__(self):
        board_str = "\n"
        board = self.board
        for i in range(len(board)):
            if i % 3 == 0 and i != 0:
                board_str += "-" * 21 + "\n"
            for j in range(len(board[i])):
                if j % 3 == 0 and j != 0:
                    board_str += "| "
                if j == 8:
                    board_str += str(board[i][j]) + "\n"
                else:
                    board_str += str(board[i][j]) + " "
        return board_str

    def sudoku_back(self):
        global iteracio

        board = self.board
        pos = [0, 0]

        if self.find_empty(pos) == False:  # End of the game
            return True

        row, col = pos

        for num in range(1, 10):
            if self.valid(row, col, num):
                board[row][col] = num
                
                if iteracio <= 20:
                    print(f"Iteracio: {iteracio} - Posicio: {row}, {col} - Valor: {num}")
                    print(str(self))

                iteracio += 1
                if self.sudoku_back():
                    return True  # Continue with the next position
                else:
                    board[row][col] = 0
        return False

    def get_board(self):
        return self.board

iteracio = 1

def Sudoku(board: list):
    game = SudokuGame(board)
    game.sudoku_back()
    return game


board = [
    [0, 8, 0,  0, 0, 2,  0, 3, 0],
    [0, 4, 0,  1, 3, 0,  0, 2, 0],
    [0, 0, 0,  7, 0, 0,  0, 0, 9],
    
    [0, 0, 0,  8, 0, 0,  6, 5, 3],
    [0, 2, 0,  0, 4, 5,  0, 0, 8],
    [5, 6, 0,  0, 0, 3,  2, 4, 0],
    
    [4, 0, 0,  0, 0, 0,  5, 0, 7],
    [7, 0, 2,  0, 0, 0,  0, 8, 4],
    [0, 0, 0,  4, 0, 0,  0, 0, 2],
]

start_board = SudokuGame(board)
print(f"Sudoku inicial: {start_board}")

start_time = time()
solution = Sudoku(board)
elapsed_time = time() - start_time

print(f"\nSudoku solucionat: {solution}")

print("Elapsed time: %.10f seconds." % elapsed_time)