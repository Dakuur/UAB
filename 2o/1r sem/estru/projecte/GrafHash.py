from ElementData import ElementData
import uuid
import math

class GrafHash:

    class Vertex:

        __slots__ = ["_key" ,"_value"]

        def __init__(self, k: uuid.UUID, e: ElementData):
            """
            Inicialitza una instància de la classe Vertex amb una clau i un valor.

            Args:
                k (uuid.UUID): Clau associada a l'element.
                e (ElementData): Valor associat a la clau, que pot ser un nom de fitxer (str) quan no està carregat
                                i ElementData quan està carregat.
            """
            self._key = k
            self._value = e

        def __hash__(self):
            """
            Retorna el valor hash de la clau i el valor de l'objecte.

            Returns:
                int: Valor hash de la clau i el valor.
            """
            return hash(tuple((self._key, self._value)))

        def __eq__(self, other) -> bool:
            """
            Compara si aquest vertex és igual a un altre.

            Args:
                other (Vertex): Altres vertex per comparar.

            Returns:
                bool: True si aquest vertex és igual a l'altre, False altrament.
            """
            return self._key == other._key

        def __ne__(self, other) -> bool:
            """
            Compara si aquest vertex no és igual a un altre.

            Args:
                other (Vertex): Altres vertex per comparar.

            Returns:
                bool: True si aquest vertex no és igual a l'altre, False altrament.
            """
            return not (self == other)

        def __str__(self) -> str:
            """
            Retorna una representació de cadena de l'objecte Vertex.

            Returns:
                str: Cadena que representa l'objecte Vertex amb la seva clau i valor.
            """
            return f"({self._key}: {self._value})"

        def __repr__(self) -> str:
            """
            Retorna una cadena que pot ser utilitzada per recrear l'objecte Vertex.

            Returns:
                str: Cadena amb el format "Vertex(key=clau, value=valor)".
            """
            return f"Vertex(key={self._key}, value={self._value})"

        @property
        def key(self):
            """
            Retorna la clau del vertex.

            Returns:
                Clau del vertex.
            """
            return self._key

        @key.setter
        def key(self, new_key):
            """
            Estableix una nova clau per al vertex.

            Args:
                new_key: Nova clau per al vertex.
            """
            self._key = new_key

        @property
        def value(self):
            """
            Retorna el valor del vertex.

            Returns:
                Valor del vertex.
            """
            return self._value

        @value.setter
        def value(self, new_value):
            """
            Estableix un nou valor per al vertex.

            Args:
                new_value: Nou valor per al vertex.
            """
            self._value = new_value
    
    __slots__ = ["_nodes", "_out", "_in"]

    def __init__(self, digraf: bool = True):
        """
        Inicialitza una instància de la classe GrafHash amb els diccionaris buits per a nodes, arestes d'eixida i arestes d'entrada.
        """
        self._nodes = {}
        self._out = {}
        self._in = {} if digraf else self._out

    def es_digraf(self) -> bool:
        """
        Verifica si el graf és un graf dirigit.

        Returns:
            bool: True si el graf és un graf dirigit, False si no ho és.
        """
        return self._out != self._in

    def getOut(self) -> dict:
        """
        Retorna el diccionari d'arestes de sortida del graf.

        Returns:
            dict: Diccionari d'arestes de sortida.
        """
        return self._out
    
    def getIn(self) -> dict:
        """
        Retorna el diccionari d'arestes d'entrada del graf.

        Returns:
            dict: Diccionari d'arestes d'entrada.
        """
        return self._in
        
    def insert_vertex(self, k: uuid.UUID, valor: ElementData) -> None:
        """
        Insereix un nou vèrtex al graf.

        Args:
            k (uuid.UUID): Clau del nou vèrtex.
            valor (ElementData): Valor associat al nou vèrtex.
        """
        if type(valor) != ElementData:
            if k == valor:
                return None
        v = self.Vertex(k, valor)
        self._nodes[k] = v
        self._out[k] = {}
        self._in[k] = {}
    
    def get(self, key: uuid.UUID) -> ElementData:
        return self[key]

    def insert_edge(self, n1: uuid.UUID, n2: uuid.UUID, p=1) -> None:
        """
        Insereix una nova aresta entre dos vèrtexs al graf.

        Args:
            n1: Clau del primer vèrtex.
            n2: Clau del segon vèrtex.
            p (int): Pes de la nova aresta (per defecte, 1).
        """
        if n2 in self._out.get(n1, {}): # si existeix el edge, augmenta en p el valor del pes
            self._out[n1][n2] += p
            self._in[n2][n1] += p
        else:
            self._out[n1][n2] = p
            self._in[n2][n1] = p

    def grauOut(self, x: uuid.UUID) -> int:
        """
        Retorna el grau de sortida del vèrtex donat.

        Args:
            x (uuid.UUID): Clau del vèrtex.

        Returns:
            int: Grau de sortida del vèrtex.
        """
        return len(self._out[x])

    def grauIn(self, x: uuid.UUID) -> int:
        """
        Retorna el grau d'entrada del vèrtex donat.

        Args:
            x (uuid.UUID): Clau del vèrtex.

        Returns:
            int: Grau d'entrada del vèrtex.
        """
        return len(self._in[x])

    def vertices(self):
        """
        Retorna una iteració de tots els vèrtexs del graf.

        Returns:
            iter: Iteració de tots els vèrtexs.
        """
        return self._nodes.__iter__()

    def edges_out(self, x: uuid.UUID) -> dict:
        """
        Retorna una iteració de tots els nodes veins de sortida de x al graf.

        Args:
            x (uuid.UUID): Clau del vèrtex.

        Returns:
            iter: Iteració de tots els nodes veins de sortida.
        """
        return self._out[x]
    
    def edges_in(self, x: uuid.UUID) -> dict:
        """
        Retorna una iteració de tots els nodes veins d'entrada de x al graf.

        Args:
            x (uuid.UUID): Clau del vèrtex.

        Returns:
            iter: Iteració de tots els nodes veins d'entrada.
        """
        return self._in[x]
    
    def keys(self) -> iter:
        """
        Retorna una iteració de les claus de tots els vèrtexs del graf.

        Returns:
            iter: Iteració de les claus dels vèrtexs.
        """
        return self._nodes.keys()
    
    def __hash__(self) -> int:
        """
        Retorna el hash de la representació ordenada dels nodes del graf.

        Returns:
            int: Valor hash de la tupla ordenada dels nodes del graf.
        """
        items = sorted(self._nodes.items())
        tupla_ordenada = tuple(items)
        return hash(tupla_ordenada)
    
    def __iter__(self):
        """
        Retorna una iteració dels vèrtexs del graf.

        Returns:
            iter: Iteració dels vèrtexs.
        """
        for i in self._nodes:
            yield i

    def __contains__(self, key: uuid.UUID) -> bool:
        """
        Comprova si una clau està present al graf.

        Args:
            key (uuid.UUID): Clau a comprovar.

        Returns:
            bool: True si la clau està present, False altrament.
        """
        return (key in self._nodes.keys())
    
    def __setitem__(self, key: uuid.UUID, value: ElementData):
        """
        Insereix un nou vèrtex amb la clau i el valor donats al graf.

        Args:
            key (uuid.UUID): Clau del nou vèrtex.
            value (ElementData): Valor associat al nou vèrtex.
        """
        self.insert_vertex(key, value)

    def __getitem__(self, key: uuid.UUID) -> ElementData:
        """
        Retorna el valor associat a la clau donada.

        Args:
            key (uuid.UUID): Clau a consultar.

        Returns:
            ElementData: Valor associat a la clau.
        """
        if key not in self._nodes.keys():
            return None
        return self._nodes[key]._value
    
    def __delitem__(self, key: uuid.UUID):
        """
        Elimina un vèrtex amb la clau donada i les arestes associades al graf.

        Args:
            key (uuid.UUID): Clau del vèrtex a eliminar.
        """
        
        self._nodes.pop(key, None)
        self._out.pop(key, None)
        self._in.pop(key, None)

        for dict_vertex in self._out.values():
            dict_vertex.pop(key, None)

        for dict_vertex in self._in.values():
            dict_vertex.pop(key, None)

    def __eq__(self, other) -> bool:
        """
        Compara si aquest graf és igual a un altre.

        Args:
            other: Altres graf per comparar.

        Returns:
            bool: True si aquest graf és igual a l'altre, False altrament.
        """
        return self._out == other._out
    
    def __ne__(self, other) -> bool:
        """
        Compara si aquest graf no és igual a un altre.

        Args:
            other: Altres graf per comparar.

        Returns:
            bool: True si aquest graf no és igual a l'altre, False altrament.
        """
        return not (self == other)

    def __len__(self) -> int:
        """
        Retorna el nombre de vèrtexs del graf.

        Returns:
            int: Nombre de vèrtexs del graf.
        """
        return len(self._nodes)

    def __lt__(self, other) -> bool:
        """
        Compara si aquest graf té menys vèrtexs que un altre.

        Args:
            other: Altres graf per comparar.

        Returns:
            bool: True si aquest graf té menys vèrtexs que l'altre, False altrament.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara si aquest graf té igual o menys vèrtexs que un altre.

        Args:
            other: Altres graf per comparar.

        Returns:
            bool: True si aquest graf té igual o menys vèrtexs que l'altre, False altrament.
        """
        return (self < other) or (len(self) == len(other))

    def __str__(self) -> str:
        """
        Retorna una representació de cadena del graf.

        Returns:
            str: Cadena que representa el graf amb les seves arestes i pesos.
        """
        cad = "========================= GRAF =========================\n"

        for it in self._out.items():
            cad1 = "--------------------------------------------------------\n"
            cad1 = f"{cad1}{it[0]}: "
            for valor in it[1].items():
                cad1 = f"{cad1}{str(valor[0])}({str(valor[1])}), "
            cad = cad + cad1 + "\n"

        return cad

    def __repr__(self) -> str:
        """
        Retorna una cadena que pot ser utilitzada per recrear el graf.

        Returns:
            str: Cadena que representa el graf amb les seves arestes i pesos.
        """
        return self.__str__()

    def minDistance(self, dist: dict, visitat: set) -> str:
        """
        Troba el node amb la distància mínima que encara no ha estat visitat.

        Args:
            dist (dict): Diccionari de distàncies.
            visitat (set): Conjunt de nodes ja visitats.

        Returns:
            str: Node amb la distància mínima no visitat.
        """
        minim = math.inf
        res = ""
        for node,distancia in dist.items():
            if node not in visitat and distancia < minim:
                minim = distancia
                res = node
        return res
    
    def dijkstraModif(self, n1: str, n3: str) -> tuple:
        """
        Algorisme de Dijkstra per trobar el camí més curt entre dos nodes.

        Args:
            n1 (str): Node d'inici.
            n3 (str): Node de destí.

        Returns:
            tuple[dict, dict]: Tupla amb els diccionaris de distàncies i predecesors.
        """
        dist = {nAux: math.inf for nAux in self._out}
        visitat = {}
        dist[n1] = 0
        predecessor = {}
        predecessor[n1] = None
        count = 0
        
        while count < len(self._nodes) - 1:
            nveiAct = self.minDistance(dist, visitat)
            visitat[nveiAct] = True
            if nveiAct == n3:
                return dist, predecessor
            elif nveiAct in self._out:
                for n2, p2 in self._out[nveiAct].items():
                    if n2 not in visitat:
                        if dist[nveiAct] + p2 < dist[n2]:
                            dist[n2] = dist[nveiAct] + p2
                            predecessor[n2] = nveiAct
            count += 1

        return dist, predecessor

    def camiMesCurt(self, n1: str, n2: str) -> list:
        """
        Troba el camí més curt entre dos nodes.

        Args:
            n1 (str): Node d'inici.
            n2 (str): Node de destí.

        Returns:
            list: Llista amb els nodes del camí més curt.
        """
        cami = [ ]
        if n1 in self._nodes and n2 in self._nodes:
            dist,predecessor=self.dijkstraModif(n1,n2)
            if n2 in predecessor:
                cami.append(n2)
                nodeAct = n2
                while not nodeAct == n1:
                    vei = predecessor[nodeAct]
                    cami.append(vei)
                    nodeAct = vei
            cami.reverse()
        return cami
