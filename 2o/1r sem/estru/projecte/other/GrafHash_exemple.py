import copy
import math

class GrafHash:

    """Graf amb Adjacency Map structure"""

    class Vertex:
        __slots__ = ['_valor']

        def __init__(self, x):
            self._valor=x
                  
        def __str__(self):
            return str(self._valor)
    
    ################################ Definicio Class _Vertex

    __slots__ = ["_nodes", "_out", "_in"]

    def __init__(self, ln=[],lv=[],lp=[]):
        self._nodes = {}
        self._out = { }
        self._in = { }
        #nodes={}
        for n in ln:
            v=self.insert_vertex(n)
            #nodes[n]=v
        if lp==[]: # sense pesos inicials
            for v in lv:
                self.insert_edge(v[0],v[1])
        else: # amb pesos inicials
            for vA,pA in zip(lv,lp):
                self.insert_edge(vA[0],vA[1],pA)
    
    def getOut(self):
        return self._out
        
    def insert_vertex(self, x):
        v= self.Vertex(x)
        self._nodes[x] = v
        self._out[x] = { }
        self._in[x] = {}
       
        return v

    def insert_edge(self, n1, n2, p1=1):
        if n2 in self._out.get(n1, {}):
            self._out[n1][n2] += 1
            self._in[n2][n1] += 1
        else:
            self._out[n1][n2] = p1
            self._in[n2][n1] = p1
        
    def grauOut(self, x):
        return len(self._out[x])

    def grauIn(self, x):
        return len(self._in[x])
    
    def vertices(self):
        """Return una iteracio de tots els vertexs del graf."""
        return self._nodes.__iter__( )

    def edges(self,x):
        """Return una iteracio de tots els nodes veins de sortida de x al graf."""
        #return self._out[x].items()
        return iter(self._out[x].__iter__())
    
    def __contains__(self, key) -> bool:
        return (key in self._nodes.keys())
    
    def __getitem__(self, key):
        return self._nodes[key]._value
    
    def __delitem__(self, key):
        self._out.pop(key, None)
        for dict_vertex in self._out.values():
            dict_vertex.pop(key, None)

    def __str__(self):
        cad="========================= GRAF =========================\n"

        for it in self._out.items():
            cad1="--------------------------------------------------------\n"
            cad1 = f"{cad1}{it[0]}: "
            for valor in it[1].items():
                cad1 = f"{cad1}{str(valor[0])}({str(valor[1])}), "

            cad = cad + cad1 + "\n"
        
        return cad
    
ln = [1,2,3,4,5,"miau"]
lv = [(1,2),(1,3),(3,4),(1,5),(2, "miau"),(1,2),(1,2),("miau", 2)]

a = GrafHash(ln, lv)
a.insert_vertex("si")
a.insert_vertex("no")
a.insert_edge("si", "no")

print(a._out)
print()
del a[2]
print(a._out)