class Motxilla:
    # codificar els següents mètodes:
    def __init__(self, valors, pesos, capacitat, n_elements):
        self.values = valors
        self.weights = pesos
        self.capacity = capacitat
        self.num_elements = n_elements
        self.M = None

    def getNumElements(self):
        # retorna el nombre d'elements
        return self.num_elements

    def getCapacitat(self):
        # retorna la capacitat de la motxilla
        return self.capacity

    def knapSack(self, t_Benefici):
        # omple la matriu de resultats parcials i retorna el benefici òptim
        # t_Benefici ja és una matriu de dimensió n_elements+1 x capacitat+1 inicialitzada a 0s
        res = t_Benefici
        for i in range(1, self.getNumElements() + 1):
            v = self.values[i-1]
            w = self.weights[i-1]
            for j in range(0, self.getCapacitat() + 1):
                dalt = res[i-1, j]
                if w <= j:
                    diag = res[i-1, j-w] + v
                else: diag = 0
                if diag > dalt:
                    res[i,j] = diag
                else:
                    res[i,j] = dalt
        self.M = res
        return self.M

    def reconstruccio(self, t_Benefici, sol_opt):
        # reconstrueix la solució a partir dels valors de la matriu de beneficis
        # sol_opt és un vector de longitud n_elements inicialitzat a 0s
        weights = self.weights
        i = self.getNumElements()
        j = self.getCapacitat()
        while (i != 0 and j != 0):
            if t_Benefici[i,j] != t_Benefici[i-1,j]:
                sol_opt[i-1] = 1
                #diagonial
                i -= 1
                j -= weights[i]
            else:
                i -= 1
                #a dalt
        return sol_opt

    def knapSack_multiple(self):
        # implementa l’algorisme de la motxilla amb múltiples unitats de cada element
        pass

        # return sol, ben_max
