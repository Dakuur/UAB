class Taula:
    def __init__(self, n):
        self.n = n
        self.taula = [[0 for _ in range(n)] for _ in range(n)]

    def resetTaula(self):
        self.taula = [[0 for _ in range(self.n)] for _ in range(self.n)]

    def __str__(self):
        # Print chess table
        taula_str = "\n"
        for i in range(len(self.taula)):
            for j in range(len(self.taula[i])):
                if self.taula[i][j] < 10:
                    taula_str += "0" + str(self.taula[i][j]) + " "
                else:
                    taula_str += str(self.taula[i][j]) + " "
            taula_str += "\n"
        return taula_str
    
    def __repr__(self) -> str:
        return str(self)
    
    def getNumCaselles(self):
        return self.n * self.n
    
    def valid(self, x, y):
        if x >= 0 and x < self.n and y >= 0 and y < self.n and self.taula[x][y] == 0:
            return True
        return False

    def backtrack(self, x, y, movei, moves):
        if movei == self.getNumCaselles() + 1:
            return True
        
        for move in moves:
            next_x, next_y = x + move[0], y + move[1]
            if self.valid(next_x, next_y):
                self.taula[next_x][next_y] = movei
                if self.backtrack(next_x, next_y, movei + 1, moves):
                    return True
                # Backtrack if move doesn't lead to a solution
                self.taula[next_x][next_y] = 0
        return False
    
    def solve_horses(self, x, y):

        moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

        self.resetTaula()
        self.taula[x][y] = 1
        if not self.backtrack(x, y, 2, moves):
            print("Error: no solution found")
        return self
    
    def solve_king(self, x, y):

        moves = [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, -1), (-1, 0)]

        self.resetTaula()
        self.taula[x][y] = 1
        if not self.backtrack(x, y, 2, moves):
            print("Error: no solution found")
        return self
    
    def solve_queen(self, x, y):

        moves = [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, -1), (-1, 0)]

        self.resetTaula()
        self.taula[x][y] = 1
        if not self.backtrack(x, y, 2, moves):
            print("Error: no solution found")
        return self

# Example usage
t = Taula(5)
print(t)

print(t.solve_horses(0, 0))

print(t.solve_king(0, 0))

print(t.solve_queen(0, 0))