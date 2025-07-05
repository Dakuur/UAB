class Motxilla:
    def __init__(self, values, weights, capacity, num_elements):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.num_elements = num_elements
        self.M = [[0 for _ in range(capacity + 1)] for _ in range(num_elements + 1)]

    def getNumElements(self):
        return self.num_elements

    def getCapacitat(self):
        return self.capacity

    def knapSack(self):
        for i in range(1, self.getNumElements() + 1):
            v = self.values[i-1]
            w = self.weights[i-1]
            for j in range(1, self.getCapacitat() + 1):
                if w <= j:
                    self.M[i][j] = max(self.M[i-1][j], self.M[i][j-w] + v)
                else:
                    self.M[i][j] = self.M[i-1][j]
        return self.M

    def reconstruccio(self):
        weights = self.weights
        i = self.getNumElements()
        j = self.getCapacitat()
        sol_opt = [0 for _ in range(self.getNumElements())]
        while (i > 0 and j > 0):
            if self.M[i][j] != self.M[i-1][j]:
                sol_opt[i-1] += 1
                j -= weights[i-1]
            else:
                i -= 1
        return sol_opt
    
    def knapSack_multiple(self):
        self.knapSack()
        sol = self.reconstruccio()
        return sol, self.M[self.getNumElements()][self.getCapacitat()]
            

values = [3,1,2]
weights = [5,2,3]

cap = 6

m = Motxilla(values, weights, cap, len(values))

sol, v = m.knapSack_multiple()

for i in range(0, len(sol)):
    print(f"V: {values[i]}, W: {weights[i]} x{sol[i]}")
    
print("\n", sol, v)