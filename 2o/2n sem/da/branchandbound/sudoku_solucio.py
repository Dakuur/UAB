from time import time
import heapq
from typing import List, Tuple

class SudokuGame:

    def __init__(self, board: List[List]) -> None:
        self.board = board

    def in_row(self, row: int, num: int):
        # De l'entrega anterior (backtracking)
        board = self.board
        row_values = board[row]
        
        for value in row_values:
            if value == num:
                return True
        
        return False

    def in_col(self, col: int, num: int) -> bool:
        # De l'entrega anterior (backtracking)
        board = self.board
        column_values = [row[col] for row in board]
        
        for value in column_values:
            if value == num:
                return True
        
        return False

    def in_box(self, row: int, col: int, num: int) -> bool:
        # De l'entrega anterior (backtracking)
        board = self.board

        for i in range(3): 
            for j in range(3):
                if(board[i+row][j+col] == num): 
                    return True
        return False

    def is_valid(self, row: int, col: int, num: int) -> bool:
        # De l'entrega anterior (backtracking)

        if self.in_row(row, num) or self.in_col(col, num) or self.in_box(row - row%3, col - col%3, num):
            return False
        return True

    def find_empty(self) -> Tuple[int, int]:
        # De l'entrega anterior (backtracking)
        board = self.board

        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    return row, col
        return -1, -1

    def get_valid_candidates(self, row, col) -> list[int]:
        candidates = set(range(1, 10))

        for i in range(9):
            if self.board[row][i] != 0:
                candidates.discard(self.board[row][i])
            if self.board[i][col] != 0:
                candidates.discard(self.board[i][col])
        box_row_start = (row // 3) * 3
        box_col_start = (col // 3) * 3
        for i in range(box_row_start, box_row_start + 3):
            for j in range(box_col_start, box_col_start + 3):
                if self.board[i][j] != 0:
                    candidates.discard(self.board[i][j])

        return list(candidates)
    
    def solve_sudoku(self) -> bool:
        board = self.board
        row, col = self.find_empty()

        if row == -1 and col == -1:
            return True

        candidates = self.get_valid_candidates(row, col)

        for num in candidates:
            if self.is_valid(row, col, num):
                board[row][col] = num

                if self.solve_sudoku():
                    return True

                board[row][col] = 0

        return False

    """def get_board(self) -> List[List]:
        return self.board"""

    def __str__(self) -> str:
        # De l'entrega anterior (backtracking)
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

def Sudoku(Tablero):

    game = SudokuGame(Tablero)
    game.solve_sudoku()
    Tablero = str(game)

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

start_board = SudokuGame(Tablero)
print(f"Sudoku inicial: {start_board}")

start_time = time()
solution = Sudoku(Tablero)
elapsed_time = time() - start_time

print(f"\nSudoku solucionat: {solution}")

print("Elapsed time: %.10f seconds." % elapsed_time)