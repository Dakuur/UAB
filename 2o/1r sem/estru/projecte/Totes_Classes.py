import os
import cfg
import eyed3
import numpy
import vlc
from time import time
import uuid
import math

class MusicFiles:

    __slots__ = ['_files', '_files_added', '_files_removed']

    def __init__(self, path=None):
        """
        Inicialitza la classe MusicFiles.

        Args:
            path (str, opcional): Ruta del directori d'arxius de música. Si no es proporciona, la instància estarà buida.
        """
        self._files: set = set()  # Conjunt dels els arxius de música emmagatzemats
        self._files_added: set = set()  # Conjunt dels arxius afegits
        self._files_removed: set = set()  # Conjunt dels arxius eliminats

        if path is not None:
            self.reload_fs(path)

    def __contains__(self, key: str):
        """
        Comprova si una clau es troba a la llista de fitxers.

        Args:
            key (str): La clau a verificar.

        Returns:
            bool: True si la clau existeix, False si no.
        """
        return key in self._files

    def __setitem__(self, key: int, value: str):
        """
        Afegeix un fitxer a la llista.

        Args:
            key (int): La clau del fitxer.
            value (str): El valor associat al fitxer.
        """
        self._files[key] = value

    def __getitem__(self, key: int):
        """
        Obté un fitxer pel seu índex.

        Args:
            key (int): L'índex del fitxer a obtenir.

        Returns:
            str: La clau del fitxer.
        """
        return list(self._files)[key]

    def __delitem__(self, key: int):
        """
        Elimina un fitxer pel seu índex.

        Args:
            key (int): L'índex del fitxer a eliminar.
        """
        del self._files[key]

    def __repr__(self) -> str:
        """
        Retorna la representació de la llista d'arxius .mp3.

        Returns:
            str: Llista d'arxius.
        """
        data = ',\n  '.join([cfg.get_canonical_pathfile(i) for i in self._files])
        return f"MusicFiles(\n [\n  {data}\n ]\n)"

    def __len__(self) -> int:
        """
        Retorna el nombre d'arxius en la llista.

        Returns:
            int: Longitud de la llista.
        """
        return len(self._files)

    def __iter__(self):
        """
        Funció que recorre tots els element de la llista en una sèrie d'iteracions.

        Returns:
            str: El nom de l'arxiu en l'iteració actual.
        """
        for i in self._files:
            yield i

    def __hash__(self) -> int:
        """
        Retorna el valor hash del conjunt de fitxers de música.

        Returns:
            int: Valor hash del conjunt de fitxers de música.
        """
        items = sorted(list(self._files))
        tupla_ordenada = tuple(items)
        return hash(tupla_ordenada)

    def __eq__(self, other) -> bool:
        """
        Compara dos classes MusicFiles i determina si contenen o no les mateixes cançons.
        No diferencia entre arxius repetits i/o desordenats.

        Args:
            other (MusicFiles): Classe d'arxius a comparar.

        Returns:
            bool: True si les dues classes contenen les mateixes cançons, False en cas contrari.
        """
        return self._files == other._files # Ambdós son sets

    def __ne__(self, other: object) -> bool:
        """
        Compara dos classes MusicFiles i determina si contenen o no les mateixes cançons.
        No diferencia entre arxius repetits i/o desordenats.

        Args:
            other (MusicFiles): Classe d'arxius a comparar.

        Returns:
            bool: True si els dos sets no contenen els mateixos arxius, False en cas contrari.
        """
        return not (self == other)

    def __lt__(self, other) -> bool:
        """
        Compara les longituds de dos classes MusicFiles i comprova si la primera és menor que la segona.

        Args:
            other (MusicFiles): Classe d'arxius a comparar.

        Returns:
            bool: True si el primer conjunt té una longitud menor que el segon, False en cas contrari.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara les longituds de dos classes MusicFiles i comprova si la primera és menor o igual que la segona.

        Args:
            other (MusicFiles): Classe d'arxius a comparar.

        Returns:
            bool: True si el primer conjunt té una longitud menor o igual que el segon, False en cas contrari.
        """
        return (self < other) or (len(self) == len(other))

    def get_short_file(self, filename) -> str:
        """
        Retorna el nom curt (des de root_dir) a partir d'un nom complet. Fent servir cfg

        Args:
            filename (str): Directori complet de l'arxiu.

        Returns:
            str: Directori de l'arxiu relatiu al root de cfg.
        """
        return cfg.get_canonical_pathfile(filename)  # Obté la ruta relativa

    def reload_fs(self, path: str) -> None:
        """
        Escaneja un directori donat i actualitza els conjunts d'arxius.

        Args:
            path (str): Ruta del directori a escanejar.
        """
        old_files = self._files
        new_files = set()

        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.lower().endswith('.mp3'):
                    file = os.path.join(root, filename)
                    file = file.replace(os.sep, '/')
                    new_files.add(file)

        self._files = new_files
        self._files_added = new_files.difference(old_files)
        self._files_removed = old_files.difference(new_files)

    def files_added(self) -> list:
        """
        Retorna una llista dels arxius afegits recentment.

        Returns:
            list: Llista dels arxius afegits.
        """
        return list(self._files_added)

    def files_removed(self) -> list:
        """
        Retorna una llista dels arxius eliminats.

        Returns:
            list: Llista dels arxius eliminats.
        """
        return list(self._files_removed)

class MusicID:
    
    __slots__ = ['_uuids']
    
    def __init__(self, files: MusicFiles = None):
        """
        Inicialitza una classe MusicID.

        Args:
            files (MusicFiles, opcional): Classe MusicFiles amb els arxius de música. Si no es proporciona, el diccionari estarà buit.
        """
        self._uuids: dict = dict()  # Diccionari per emmagatzemar els UUIDs dels arxius
        if files is not None:
            for i in files:
                file = cfg.get_canonical_pathfile(i)
                self.generate_uuid(file)
    
    def __contains__(self, key: str) -> bool:
        """
        Verifica si la clau (nom de fitxer) està present al diccionari d'UUIDs.

        Args:
            key (str): Nom de fitxer a verificar.

        Returns:
            bool: True si la clau està present, False si no ho està.
        """
        return (key in self._uuids)

    def __setitem__(self, key: str, value: uuid.UUID):
        """
        Estableix una relació entre el nom de fitxer i el seu UUID corresponent al diccionari.

        Args:
            key (str): Nom de fitxer.
            value (uuid.UUID): UUID associat al fitxer.
        """
        self._uuids[key] = value

    def __getitem__(self, key: str) -> uuid.UUID:
        """
        Obté l'UUID associat al nom de fitxer proporcionat.

        Args:
            key (str): Nom de fitxer.

        Returns:
            uuid.UUID: UUID associat al fitxer.
        """
        return self._uuids[key]

    def __delitem__(self, key: str):
        """
        Elimina l'entrada corresponent al nom de fitxer del diccionari d'UUIDs.

        Args:
            key (str): Nom de fitxer a eliminar.
        """
        del self._uuids[key]

    def keys(self):
        """
        Retorna les claus (noms de fitxer) presents al diccionari d'UUIDs.

        Returns:
            dict_keys: Claus presents al diccionari d'UUIDs.
        """
        return self._uuids.keys()

    def __repr__(self) -> str:
        """
        Retorna una representació en forma de cadena de text del diccionari d'UUIDs.

        Returns:
            str: Cadena de text que representa el diccionari d'UUIDs.
        """
        data = ',\n'.join([f"{i}: \n  {x}" for i, x in self._uuids.items()])
        return "MusicID(\n {\n"  + data + "\n }\n)"
    
    def __len__(self) -> int:
        """
        Retorna el nombre d'UUIDs emmagatzemats al diccionari.

        Returns:
            int: Nombre d'UUIDs emmagatzemats.
        """
        return len(self._uuids)
    
    def __iter__(self):
        """
        Retorna un iterador sobre les claus (noms de fitxer) del diccionari d'UUIDs.

        Returns:
            str: La clau (nom de fitxer) en l'iteració actual.
        """
        for i in self._uuids:
            yield i
        
    def __hash__(self) -> int:
        """
        Retorna el valor hash associat a un nom de fitxer específic en el diccionari d'UUIDs.

        Args:
            value (str): Nom de fitxer per al qual es vol obtenir el valor hash.

        Returns:
            int: Valor hash del nom de fitxer donat.
        """
        items = sorted(list(self._uuids.items()))
        tupla_ordenada = tuple(items)
        return hash(tupla_ordenada)
    
    def __eq__(self, other) -> bool:
        """
        Compara dues instàncies de la classe MusicID per determinar si contenen els mateixos ítems (Fitxer, UUID).

        Args:
            other (MusicID): Altra instància de MusicID a comparar.

        Returns:
            bool: True si les dues instàncies contenen les mateixes cançons i UUIDs amb la mateixa correspondència,
            False en cas contrari.
        """
        return self._uuids == other._uuids #comprovem que cada clau es correspon amb cada

    def __ne__(self, other: object) -> bool:
        """
        Compara dues instàncies de la classe MusicID per determinar si no contenen els mateixos ítems (Fitxer, UUID)
            o si difereixen en el valor d'alguna clau.

        Args:
            other (MusicID): Altra instància de MusicID a comparar.

        Returns:
            bool: True si les dues instàncies no contenen les mateixes cançons i UUIDs, o si difereixen en el valor
            d'alguna clau; False en cas contrari.
        """
        return not (self == other)
    
    def __lt__(self, other) -> bool:
        """
        Compara les longituds de dues instàncies de la classe MusicID per determinar si la primera és menor que la segona.

        Args:
            other (MusicID): Altra instància de MusicID a comparar.

        Returns:
            bool: True si la longitud del primer diccionari és menor que la del segon, False en cas contrari.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara les longituds de dues instàncies de la classe MusicID per determinar si la primera és menor o igual que la segona.

        Args:
            other (MusicID): Altra instància de MusicID a comparar.

        Returns:
            bool: True si la longitud del primer diccionari és menor o igual que la del segon, False en cas contrari.
        """
        return (self < other) or (len(self) == len(other))

    def generate_uuid(self, file: str):
        """
        Genera un UUID per a un arxiu donat.

        Args:
            file (str): Ruta de l'arxiu.

        Returns:
            uuid.UUID: UUID generat o None si ja existeix un l'UUID.
        """
        mp3_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file)
        mp3_uuid = str(mp3_uuid) # Per simplificació
        if mp3_uuid not in self._uuids.values():
            self._uuids[file] = mp3_uuid
            return mp3_uuid
        else:
            print(f"Avís: L'arxiu {file} no es farà servir (error per col·lisió)")
            return None

    def get_uuid(self, file: str) -> str:
        """
        Retorna l'UUID associat a d'arxiu donat.

        Args:
            file (str): Ruta de l'arxiu.

        Returns:
            uuid.UUID: UUID de l'arxiu o missatge d'error si no existeix.
        """
        try:
            #file_short = cfg.get_canonical_pathfile(file)
            return self._uuids[file]
        except KeyError:
            print("UUID no existent")

    def remove_uuid(self, uuid: uuid.UUID):
        """
        Elimina l'UUID associat a un UUID donat.

        Args:
            uuid (uuid.UUID): UUID a eliminar.
        """
        for k, v in self._uuids.items():
            if v == uuid:
                self._uuids.pop(k)
                return None

class ElementData:
    
    __slots__ = ["_title", "_artist", "_album", "_genre", "_duration", "_filename"]

    def __init__(self, title: str = "", artist: str = "", album: str = "",genre: str = "", duration: int = 0, filename: str = "") -> None:
        """
        Inicialitza la classe ElementData.

        Args:
            title (str): Títol de la cançó
            artist (str): Autor(s) de la cançó
            album (str): Àlbum el qual pertany la cançó
            genre (str): Gènere(s) de la cançó
            duration (str): Duració en segons de la cançó
            filename (str): Nom de l'arxiu de la cançó la qual s'han extret de les metadades
        """
        self._title = title
        self._artist = artist
        self._album = album
        self._genre = genre
        self._duration = duration
        self._filename = filename

    def __hash__(self) -> int:
        """
        Retorna el hash del nom de l'arxiu.

        Returns:
            int: Valor hash del nom de l'arxiu.
        """
        return hash(self._filename)

    def __eq__(self, other) -> bool:
        """
        Compara dos elements de dades i determina si són iguals pel nom de l'arxiu.

        Args:
            other (ElementData): Element de dades a comparar.

        Returns:
            bool: True si els elements tenen el mateix nom d'arxiu, False en cas contrari.
        """
        return self._filename == other._filename

    def __ne__(self, other: object) -> bool:
        """
        Compara dos elements de dades i determina si són diferents pel nom de l'arxiu.

        Args:
            other (ElementData): Element de dades a comparar.

        Returns:
            bool: True si els elements tenen diferent nom d'arxiu, False en cas contrari.
        """
        return not (self == other)

    def __lt__(self, other) -> bool:
        """
        Compara dos elements de dades i determina si el primer és menor pel nom de l'arxiu.

        Args:
            other (ElementData): Element de dades a comparar.

        Returns:
            bool: True si el primer element és menor pel nom d'arxiu, False en cas contrari.
        """
        return self._filename < other._filename

    def __le__(self, other) -> bool:
        """
        Compara dos elements de dades i determina si el primer és menor o igual pel nom de l'arxiu.

        Args:
            other (ElementData): Element de dades a comparar.

        Returns:
            bool: True si el primer element és menor o igual pel nom d'arxiu, False en cas contrari.
        """
        return (self < other) or (self == other)

    def __repr__(self) -> str:
        """
        Retorna una representació  de l'objecte ElementData.

        Returns:
            str: Cadena que representa l'objecte ElementData amb els seus atributs.
        """
        return f"ElementData(\n   title='{self._title}',\n   artist='{self._artist}',\n   album='{self._album}',\n   genre='{self._genre}',\n   duration={self._duration},\n   filename='{self._filename}'\n)"

    @property
    def title(self) -> str:
        """
        Retorna el títol de l'element.

        Returns:
            str: Títol de l'element.
        """
        return self._title

    @title.setter
    def title(self, new_title: str):
        """
        Estableix un nou títol per a l'element.

        Args:
            new_title (str): Nou títol de l'element.
        """
        self._title = new_title

    @property
    def artist(self) -> str:
        """
        Retorna l'artista de l'element.

        Returns:
            str: Artista de l'element.
        """
        return self._artist

    @artist.setter
    def artist(self, new_artist: str):
        """
        Estableix un nou artista per a l'element.

        Args:
            new_artist (str): Nou artista de l'element.
        """
        self._artist = new_artist

    @property
    def album(self) -> str:
        """
        Retorna l'àlbum de l'element.

        Returns:
            str: Àlbum de l'element.
        """
        return self._album

    @album.setter
    def album(self, new_album: str):
        """
        Estableix un nou àlbum per a l'element.

        Args:
            new_album (str): Nou àlbum de l'element.
        """
        self._album = new_album

    @property
    def genre(self) -> str:
        """
        Retorna el gènere de l'element.

        Returns:
            str: Gènere de l'element.
        """
        return self._genre

    @genre.setter
    def genre(self, new_genre: str):
        """
        Estableix un nou gènere per a l'element.

        Args:
            new_genre (str): Nou gènere de l'element.
        """
        self._genre = new_genre

    @property
    def duration(self) -> int:
        """
        Retorna la durada de l'element en segons.

        Returns:
            int: Durada de l'element en segons.
        """
        return self._duration

    @duration.setter
    def duration(self, new_duration: int):
        """
        Estableix una nova durada per a l'element.

        Args:
            new_duration (int): Nova durada de l'element en segons.
        """
        self._duration = new_duration

    @property
    def filename(self) -> str:
        """
        Retorna el nom del fitxer associat a l'element.

        Returns:
            str: Nom del fitxer de l'element.
        """
        return self._filename

    @filename.setter
    def filename(self, new_filename: str):
        """
        Estableix un nou nom de fitxer per a l'element.

        Args:
            new_filename (str): Nou nom de fitxer de l'element.
        """
        self._filename = new_filename

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

class MusicData:
    
    __slots__ = ["_songs"]

    def __init__(self, ids: MusicID = None):
        """
        Inicialitza una classe MusicData.

        Args:
            ids (MusicID, opcional): Classe MusicID amb els UUIDs dels arxius de música. Si no es proporciona, la classe estarà buida.
        """
        self._songs: GrafHash = GrafHash()  # Diccionari per emmagatzemar les dades de les cançons (UUID -> (fitxer, metadades))
        if ids is not None:
            for filename in ids:
                song_id = ids[filename]
                self.add_song(song_id, filename)
                self.load_metadata(song_id)

    def __hash__(self) -> int:
        """
        Retorna el valor hash de les dades de les cançons.

        Returns:
            int: Valor hash de les dades de les cançons.
        """
        return hash(self._songs)

    def __repr__(self) -> str:
        """
        Representació de cada UUID (key) amb la metadata corresponent (value).
        El valor és un objecte de eyed3.mp3.Mp3AudioFile, es representa com ho fa en la llibreria

        Returns:
            str: Llista dels valors del diccionari.
        """
        data = ',\n'.join([f"{i}: \n  {self._songs[i]}" for i in self._songs])
        return "MusicData(\n {\n"  + data + "\n }\n)"
    
    def __len__(self) -> int:
        """
        Retorna el nombre de cançons emmagatzemades.

        Returns:
            int: Nombre de cançons.
        """
        return len(self._songs)
    
    def __iter__(self):
        """
        Funció que recorre totes les claus del diccionari en una sèrie d'iteracions.

        Returns:
            uuid.UUID: La clau del diccionari en l'iteració actual.
        """
        for i in self._songs:
            yield i
    
    def __eq__(self, other) -> bool:
        """
        Compara dos classes MusicData i determina si contenen o no els mateixos ítems (UUID, Metadata).

        Args:
            other (MusicData): Classe MusicData a comparar.

        Returns:
            bool: True si les dues classes contenen les mateixes UUIDs i Metadata
            (amb la meteixa correspondència), False en cas contrari.
        """
        return self._songs == other._songs #comprovem que cada clau es correspon amb cada

    def __ne__(self, other: object) -> bool:
        """
        Compara dos classes MusicData i determina si contenen o no els mateixos ítems (UUID, Metadata).

        Args:
            other (MusicData): Classe MusicData a comparar.

        Returns:
            bool: True si les dues classes no contenen les mateixes UUIDs i Metadata,
            o si difereixen en el valor d'alguna clau, False en cas contrari.
        """
        return not (self == other)
    
    def __lt__(self, other) -> bool:
        """
        Compara les longituds de dos classes MusicData i comprova si la primera és menor que la segona.

        Args:
            other (MusicData): Classe MusicData a comparar.

        Returns:
            bool: True si el primer diccionari té una longitud menor que el segon, False en cas contrari.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara les longituds de dos classes MusicData i comprova si la primera és menor o igual que la segona.

        Args:
            other (MusicData): Classe MusicData a comparar.

        Returns:
            bool: True si el primer diccionari té una longitud menor o igual que el segon, False en cas contrari.
        """
        return (self < other) or (len(self) == len(other))

    def __contains__(self, key: uuid.UUID) -> bool:
        """
        Comprova si una clau està present al graf.

        Args:
            key (uuid.UUID): Clau a comprovar.

        Returns:
            bool: True si la clau està present, False altrament.
        """
        return (key in self._songs)

    def __setitem__(self, key: uuid.UUID, value: ElementData):
        """
        Insereix un nou vèrtex amb la clau i el valor donats al graf.

        Args:
            key (uuid.UUID): Clau del nou vèrtex.
            value: Valor associat al nou vèrtex.
        """
        self._songs[key] = value

    def __getitem__(self, key: uuid.UUID) -> ElementData:
        """
        Retorna el valor associat a la clau donada.

        Args:
            key (uuid.UUID): Clau a consultar.

        Returns:
            ElementData: Valor associat a la clau.
        """
        return self._songs[key]

    def __delitem__(self, key: uuid.UUID) -> None:
        """
        Elimina un vèrtex amb la clau donada i les arestes associades al graf.

        Args:
            key (uuid.UUID): Clau del vèrtex a eliminar.
        """
        del self._songs[key]

    def keys(self) -> list:
        """
        Retorna una llista amb les claus (UUIDs) del graf.

        Returns:
            list: Llista amb les claus del graf.
        """
        return self._songs.keys()

    def add_song(self, uuid: uuid.UUID, file: str) -> None:
        """
        Afegeix una cançó a la col·lecció.

        Args:
            uuid (uuid.UUID): UUID de la cançó.
            file (str): Ruta del fitxer de la cançó.
        """
        empty = [None, ""]
        if uuid not in empty and file not in empty:
            self[uuid] = file

    def remove_song(self, uuid: uuid.UUID) -> None:
        """
        Elimina una cançó de la col·lecció.

        Args:
            uuid (uuid.UUID): UUID de la cançó a eliminar.
        """
        del self[uuid]

    def load_metadata(self, uuid: uuid.UUID) -> None:
        """
        Carrega les metadades d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.
        """

        if self[uuid] == type(ElementData()): # Ja carregat
            return None

        file = self[uuid] # assume not loaded yet
        root = cfg.get_root()
        file = os.path.join(root, file) #Carrega el fitxer amb el path complet, no relatiu
        metadata = eyed3.load(file)

        try:
            genre = metadata.tag.genre.name
        except:
            genre = ""
        try:
            data = ElementData(
                title=metadata.tag.title,
                artist=metadata.tag.artist,
                album=metadata.tag.album,
                genre=genre,
                duration=int(numpy.ceil(metadata.info.time_secs)),
                filename=file
            )
        except:
            data = ElementData(filename=file)
        self[uuid] = data
        return None

    def get_file(self, uuid: uuid.UUID) -> str:
        """
        Retorna el nom del fitxer d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            str: Nom del fitxer.
        """
        try:
            return cfg.get_canonical_pathfile(self[uuid]._filename)
        except:
            #print("Metadata errònea")
            pass

    def get_filename(self, uuid: uuid.UUID) -> str: 
        """
        Retorna el nom del fitxer d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            str: Nom del fitxer.
        """
        
        return self.get_file(uuid)

    def get_title(self, uuid: uuid.UUID) -> str:
        """
        Retorna el títol d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            str: Títol de la cançó.
        """
        try:
            return self[uuid]._title
        except:
            #print("Metadata errònea")
            pass

    def get_artist(self, uuid: uuid.UUID) -> str:
        """
        Retorna l'artista d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            str: Artista de la cançó.
        """
        try:
            return self[uuid]._artist
        except:
            #print("Metadata errònea")
            pass

    def get_album(self, uuid: uuid.UUID) -> str:
        """
        Retorna l'àlbum d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            str: Àlbum de la cançó.
        """
        try:
            return self[uuid]._album
        except:
            #print("Metadata errònea")
            pass

    def get_genre(self, uuid: uuid.UUID) -> str:
        """
        Retorna el(s) gènere(s) d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            str: Gènere(s) de la cançó.
        """
        try:
            return self[uuid]._genre
        except:
            #print("Metadata errònea")
            pass
        
    def get_duration(self, uuid: uuid.UUID) -> int:
        """
        Retorna la duració d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            int: Duració de la cançó.
        """
        try:
            return self[uuid]._duration

        except:
            #print("Metadata errònea")
            return 0

    def read_playlist(self, playlist) -> None:
        """
        Llegeix una llista de reproducció i la converteix en arestes del graf.

        Args:
            playlist (list): Llista de UUIDs de les cançons a reproduir.
        """
        if len(playlist) < 2:
            print("La llista ha de contenir dos o més cançons.")
            return None
        for i in range(len(playlist) - 1):
            n1 = playlist[i]
            n2 = playlist[i + 1]
            self._songs.insert_edge(n1, n2)

    def get_song_rank(self, uuid: uuid.UUID) -> int:
        """
        Retorna la classificació d'una cançó basada en la suma dels pesos de les arestes d'entrada i sortida.

        Args:
            uuid (uuid.UUID): UUID de la cançó.

        Returns:
            int: Suma dels pesos de les arestes d'entrada i sortida.
        """
        suma = 0
        entrada = self._songs.edges_in(uuid)
        for i in entrada.values():
            suma += i
        sortida = self._songs.edges_out(uuid)
        for i in sortida.values():
            suma += i
        return suma
    
    def get_next_songs(self, uuid: uuid.UUID) -> iter:
        """
        Retorna una iteració de les cançons següents al node donat.

        Args:
            uuid (uuid.UUID): UUID del node.

        Returns:
            iter: Iteració de tuples (uuid.UUID, pes de l'aresta) de les cançons següents.
        """
        dict_next = self._songs.edges_out(uuid)
        if len(dict_next) < 1:
            print(f"0 next songs for uuid: {uuid}")
        return iter(tuple((k, v)) for k, v in dict_next.items())
    
    def get_previous_songs(self, uuid: uuid.UUID) -> iter:
        """
        Retorna una iteració de les cançons anteriors al node donat.

        Args:
            uuid (uuid.UUID): UUID del node.

        Returns:
            iter: Iteració de tuples (uuid.UUID, pes de l'aresta) de les cançons anteriors.
        """
        dict_prev = self._songs.edges_in(uuid)
        if len(dict_prev) < 1:
            print(f"0 previous songs for uuid: {uuid}")
        return iter(tuple((k, v)) for k, v in dict_prev.items())
    
    def get_song_distance(self, uuid1: uuid.UUID, uuid2: uuid.UUID) -> tuple:
        """
        Retorna la distància entre dues cançons en termes del nombre d'arestes i la suma dels pesos.

        Args:
            uuid1 (uuid.UUID): UUID de la primera cançó.
            uuid2 (uuid.UUID): UUID de la segona cançó.

        Returns:
            tuple: Tupla amb el nombre d'arestes i la suma dels pesos.
        """
        cami = self._songs.camiMesCurt(uuid1, uuid2)
        
        num_arestes = len(cami) - 1

        if num_arestes < 0:
            num_arestes = 0
        
        #print(f"Camí: {cami}")

        suma_pesos = 0
        for i in range(0, num_arestes):
            n1 = cami[i]
            n2 = cami[i + 1]
            pes = self._songs._out[n1][n2]
            #print(f"Pes entre {n1} i {n2}: {pes}")
            suma_pesos += pes
        
        #print(f"Total pesos camí: {suma_pesos}\n")

        return num_arestes, suma_pesos

class MusicPlayer:
    
    __slots__ = ['_songs_data', '_player']

    def __init__(self, data: MusicData):
        """
        Inicialitza una classe MusicPlayer.

        Args:
            data (MusicData, opcional): Classe MusicData amb les dades de les cançons. Si no es proporciona, la classe estarà buida.
        """
        self._songs_data: MusicData = data
        self._player: vlc.MediaPlayer = vlc.MediaPlayer

    def __hash__(self) -> int:
        """
        Retorna el valor hash de les dades de les cançons del reproductor de música.

        Returns:
            int: Valor hash de les dades de les cançons del reproductor de música.
        """
        return hash(self._songs_data)

    def __repr__(self) -> str:
        """
        Retorna la representació de la classe MusicPlayer.

        Returns:
            str: Representació de la classe MusicPlayer.
        """
        return repr(self._songs_data)

    def update_data(self, new_data: MusicData):
        """
        Actualitza les dades de les cançons amb un nou conjunt de dades.

        Args:
            new_data (MusicData): Noves dades de les cançons.
        """
        self._songs_data = new_data
        return None

    def print_song(self, uuid: uuid.UUID):
        """
        Mostra informació detallada d'una cançó.

        Args:
            uuid (uuid.UUID): UUID de la cançó.
        """
        try:
            root = cfg.get_root()
            short = self._songs_data.get_file(uuid)
            long = os.path.join(root, short)
            print(f"\nReproduint [{long}]\n")
            print(f"  Duració: {self._songs_data.get_duration(uuid)} segons")
            print(f"  Títol: {self._songs_data.get_title(uuid)}")
            print(f"  Artista: {self._songs_data.get_artist(uuid)}")
            print(f"  Àlbum: {self._songs_data.get_album(uuid)}")
            print(f"  Gènere(s): {self._songs_data.get_genre(uuid)}")
            print(f"  UUID: {uuid}")
            print(f"  Arxiu: {self._songs_data.get_file(uuid)}\n")  # short
            return None
        except TypeError:
            print("UUID incorrecte")

    def play_file(self, file: str): # Asíncrona
        """
        Reprodueix un fitxer de música.

        Args:
            file (str): Ruta del fitxer a reproduir.
        """
        try:
            root = cfg.get_root()
            file = os.path.join(root, file)
            self._player = vlc.MediaPlayer(file)
            self._player.play()
            return None
        except NameError:
            print("Fitxer incorrecte")

    def play_song(self, uuid: uuid.UUID, mode: int): # Síncrona
        """
        Reprodueix una cançó segons el mode especificat.

        Args:
            uuid (uuid.UUID): UUID de la cançó.
            mode (int): Mode de reproducció (0: només metadades, 1: metadades + àudio, 2: només àudio).
        """
        file = self._songs_data.get_file(uuid)
        if file is None:
            return None

        if mode == 0:  # NOMÉS METADADES
            self.print_song(uuid)
        elif mode == 1:  # METADADES + ÀUDIO
            self.print_song(uuid)
            self.play_file(self._songs_data.get_file(uuid))
        elif mode == 2:  # NOMÉS ÀUDIO
            self.play_file(self._songs_data.get_file(uuid))
        else:
            print("Mode de reproducció incorrecte")
            return None
        
        if mode != 0:
            
            timeout = time.time() + self._songs_data.get_duration(uuid)
            while True:
                if time.time() < timeout: # Per que retorni la funció quan acabi la cançó
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt: # STOP amb <CTRL>+<C> a la consola
                        break # Per que retorni la funció l'usuari vulgui
                else:
                    break
            self._player.stop()
        
        return None

class PlayList:
    
    __slots__ = ['_uuids', '_player', '_playlist']
    
    def __init__(self, uuids: MusicID, player: MusicPlayer, playlist: list = [], file: str = None):
        """
        Inicialitza una classe PlayList.

        Args:
            uuids (MusicID, opcional): Classe MusicID amb els UUIDs de les cançons. Si no es proporciona, estarà buida.
            player (MusicPlayer, opcional): Classe MusicPlayer per reproduir les cançons. Si no es proporciona, estarà buida.
            playlist (list, opcional): Llista de UUIDs de les cançons a la llista de reproducció. Si no es proporciona, la llista estarà buida.
        """
        self._uuids: MusicID = uuids # Llista d'UUIDs emmagatzemats
        self._player: MusicPlayer = player  # Reproductor
        self._playlist: list = [] # Llista d'UUIDs en la llista de reproducció
        if len(playlist) > 0:
            self.reset_playlist() # Suposem que es reemplaça en comptes d'afegir
            self.read_list(playlist)
        if file != None:
            self.load_file(file)
    
    def __contains__(self, key: uuid.UUID) -> bool:
        """
        Comprova si un UUID està a la llista de reproducció.

        Args:
            key (uuid.UUID): UUID a comprovar.

        Returns:
            bool: True si l'UUID és a la llista de reproducció, False en cas contrari.
        """
        return (key in self._playlist)

    def __setitem__(self, key: int, value: uuid.UUID):
        """
        Defineix un element de la llista de reproducció.

        Args:
            key (int): Índex de l'element.
            value (uuid.UUID): Valor de l'element.
        """
        self._playlist[key] = value

    def __getitem__(self, key: int) -> uuid.UUID:
        """
        Obté un element de la llista de reproducció.

        Args:
            key (int): Índex de l'element.

        Returns:
            uuid.UUID: Valor de l'element.
        """
        return self._playlist[key]

    def __delitem__(self, key: int):
        """
        Elimina un element de la llista de reproducció.

        Args:
            key (int): Índex de l'element.
        """
        del self._playlist[key]

    def __repr__(self) -> str:
        """
        Retorna una representació en cadena de la llista de reproducció.

        Returns:
            str: Representació de la llista de reproducció.
        """
        llista = ',\n   '.join([i for i in self._playlist])
        return f"Playlist(\n [\n   {llista}\n ]\n)"
    
    def __len__(self) -> int:
        """
        Retorna el nombre de cançons a la llista de reproducció.

        Returns:
            int: Nombre de cançons a la llista de reproducció.
        """
        return len(self._playlist)
    
    def __iter__(self) -> uuid.UUID:
        """
        Retorna un iterador per recórrer els elements de la llista de reproducció.

        Returns:
            uuid.UUID: L'UUID de la cançó en l'iteració actual.
        """
        for i in self._playlist:
            yield i
        
    def __hash__(self) -> int:
        """
        Retorna el valor hash de la llista de reproducció.

        Returns:
            int: Valor hash de la llista de reproducció.
        """
        items = sorted(self._playlist)
        tupla_ordenada = tuple(items)
        return hash(tupla_ordenada)

    def __eq__(self, other) -> bool:
        """
        Compara dues llistes de reproducció i determina si contenen les mateixes cançons (en el mateix ordre).

        Args:
            other (PlayList): Llista de reproducció a comparar.

        Returns:
            bool: True si les dues llistes són idèntiques, False en cas contrari.
        """
        return self._playlist == other._playlist

    def __ne__(self, other: object) -> bool:
        """
        Compara dues llistes de reproducció i determina si no contenen les mateixes cançons (o no en el mateix ordre).

        Args:
            other (PlayList): Llista de reproducció a comparar.

        Returns:
            bool: True si les dues llistes són diferents, False en cas contrari.
        """
        return not (self == other)
    
    def __lt__(self, other) -> bool:
        """
        Compara dues llistes de reproducció i verifica si la primera té menys elements que la segona.

        Args:
            other (PlayList): Llista de reproducció a comparar.

        Returns:
            bool: True si la primera llista té menys elements que la segona, False en cas contrari.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara dues llistes de reproducció i verifica si la primera té menys o igual nombre d'elements que la segona.

        Args:
            other (PlayList): Llista de reproducció a comparar.

        Returns:
            bool: True si la primera llista té menys o igual nombre d'elements que la segona, False en cas contrari.
        """
        return (self < other) or (len(self) == len(other))

    def load_file(self, filename: str):
        """
        Carrega una llista de reproducció des d'un fitxer.

        Args:
            filename (str): Nom del fitxer de la llista de reproducció en format M3U.
        """
        self.reset_playlist()  # Suposem que es reemplaça en comptes d'afegir

        with open(filename, 'r', encoding='latin-1') as file:
            for line in file:
                line = line.strip()
                if not line.startswith("#") and line.endswith(".mp3"):
                    song_uuid = self._uuids.get_uuid(line)
                    if (song_uuid is not None) and (song_uuid not in self._playlist):
                        self.add_song_at_end(song_uuid)
                        # print(f"Afegit: {line}: {song_uuid}")

        print(f"Llista de reproducció creada amb {len(self._playlist)} cançons")

        return None

    def play(self, play_mode: int):
        """
        Reprodueix les cançons de la llista de reproducció segons el mode especificat.

        Args:
            play_mode (int): Mode de reproducció (0: només metadades, 1: metadades + àudio, 2: només àudio).
        """
        for i in self._playlist:
            self._player.play_song(i, play_mode)

    def add_song_at_end(self, uuid: uuid.UUID):
        """
        Afegeix una cançó al final de la llista de reproducció.

        Args:
            uuid (uuid.UUID): UUID de la cançó a afegir.
        """
        #if (uuid is not None) and (uuid not in self._playlist):
        if (uuid is not None):
            self._playlist.append(uuid)
        return None

    def remove_first_song(self):
        """
        Elimina la primera cançó de la llista de reproducció.
        """
        if len(self._playlist) > 0:
            self._playlist.pop(0)
        return None

    def remove_last_song(self):
        """
        Elimina l'última cançó de la llista de reproducció.
        """
        if len(self._playlist) > 0:
            self._playlist.pop(-1)
        return None
    
    def reset_playlist(self):
        """
        Restableix la llista de reproducció buida.
        """
        self._playlist = []
    
    def read_list(self, p_llista: list):
        """
        Afegeix les cançons de la llista donada al final de la llista de reproducció.

        Args:
            p_llista (list): Llista d'UUIDs de les cançons a afegir.
        """
        self._playlist = []
        for i in p_llista:
            if i not in self._playlist:
                self.add_song_at_end(i)

class SearchMetadata:
    
    __slots__ = ['_songs']
    
    def __init__(self, data: MusicData):
        """
        Inicialitza una classe SearchMetadata.

        Args:
            data (MusicData, opcional): Classe MusicData amb les dades de les cançons. Si no es proporciona, la classe estarà buida.
        """
        if type(data) != MusicData:
            raise TypeError
        self._songs: MusicData = data

    def __repr__(self) -> str:
        """
        Retorna la representació de les dades del diccionari MusicData

        Returns:
            str: Llista dels valors del diccionari.
        """
        return repr(self._songs)

    def __len__(self):
        """
        Retorna el nombre de cançons emmagatzemades.

        Returns:
            int: Nombre de cançons.
        """
        return len(self._songs)

    def __hash__(self) -> int:
        """
        Retorna el valor hash de les dades de les cançons.

        Returns:
            int: Valor hash de les dades de les cançons.
        """
        return hash(self._songs)

    def __iter__(self):
        """
        Funció que itera sobre les cançons de les dades.

        Returns:
            uuid.UUID: L'uuid de la cançó en l'iteració actual.
        """
        for i in self._songs:
            yield i

    def update_songs(self, data: MusicData):
        """
        Actualitza les dades de les cançons.

        Args:
            data (MusicData): Noves dades de les cançons.
        """
        self._songs = data
        return None

    def search_by_attribute(self, consulta: str, attribute: str) -> list:
        """
        Cerca cançons segons un atribut i un string de consulta.

        Args:
            consulta (str): String a cercar.
            attribute (str): Atribut per fer la cerca (títol, artista, àlbum o gènere(s)).

        Returns:
            list: Llista d'UUIDs de les cançons que coincideixen amb la cerca.
        """
        null_values = [None, "", "NONE"]

        results = []
        for i in self._songs: # itera per KEYS
            attr_value = str(getattr(self._songs, f'get_{attribute}')(i)).upper()
            if consulta.upper() in null_values and attr_value in null_values:
                results.append(i)
            elif attr_value.find(consulta.upper()) != -1:
                results.append(i)
            else:
                pass
        return results

    def title(self, consulta: str) -> list:
        """
        Cerca cançons pel títol.

        Args:
            consulta (str): String a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb el títol que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(consulta), "title")

    def artist(self, consulta: str) -> list:
        """
        Cerca cançons per l'artista.

        Args:
            consulta (str): String a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb l'artista que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(consulta), "artist")

    def album(self, consulta: str) -> list:
        """
        Cerca cançons per l'àlbum.

        Args:
            consulta (str): String a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb l'àlbum que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(consulta), "album")

    def genre(self, consulta: str) -> list:
        """
        Cerca cançons pel gènere.

        Args:
            consulta (str): String a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb el gènere que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(consulta), "genre")

    def and_operator(self, list1: list, list2: list) -> list:
        """
        Operador AND entre dues llistes d'UUIDs.

        Args:
            list1 (list): Primera llista d'UUIDs.
            list2 (list): Segona llista d'UUIDs.

        Returns:
            list: Llista d'UUIDs que apareixen a les dues llistes.
        """
        return list(set(list1).intersection(set(list2)))

    def or_operator(self, list1: list, list2: list) -> list:
        """
        Operador OR entre dues llistes d'UUIDs.

        Args:
            list1 (list): Primera llista d'UUIDs.
            list2 (list): Segona llista d'UUIDs.

        Returns:
            list: Llista d'UUIDs que apareixen en una o l'altra llista.
        """
        return list(set(list1).union(set(list2)))

    def song_similarity(self, uuid_a: uuid.UUID, uuid_b: uuid.UUID) -> float:
        """
        Calcula la similitud entre dues cançons utilitzant el seu UUID.

        Args:
            uuid_a (uuid.UUID): UUID de la primera cançó.
            uuid_b (uuid.UUID): UUID de la segona cançó.

        Returns:
            float: Valor de similitud entre les dues cançons.
        """
        if uuid_a == uuid_b:
            return -1
        x = self._songs

        ab_nodes, ab_value = x.get_song_distance(uuid_a, uuid_b) # Distància A -> B
        ab = 0
        if (ab_nodes > 0):
            ab = (ab_value / ab_nodes) * (x.get_song_rank(uuid_a) / 2)

        ba_nodes, ba_value = x.get_song_distance(uuid_b, uuid_a) # Distància B -> A
        ba = 0
        if (ba_nodes > 0):
            ba = (ba_value / ba_nodes) * (x.get_song_rank(uuid_b) / 2)
            
        return ab + ba

    def get_similar(self, uuid: uuid.UUID, max_list: int) -> list:
        """
        Obté una llista de les cançons més similars a una determinada cançó (amb similaritat > 0).

        Args:
            uuid (uuid.UUID): UUID de la cançó.
            max_list (int): Nombre màxim d'elements a incloure a la llista.

        Returns:
            list: Llista d'UUIDs de les cançons més similars.
        """
        res = [(song, self.song_similarity(uuid, song)) for song in self._songs]
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [x[0] for x in res if x[1] > 0] # Treu tracks amb similaritat 0
        if uuid in res:
            res.remove(uuid) # Treu la pròpia cançó
        """for i in res:
            print(f"Similarity: {self.song_similarity(uuid, i)}")"""
        return res[:max_list - 1]
    
    def get_topfive(self) -> list:
        """
        Obté una llista amb les cinc millors cançons basades en els seus rangs i similituds.

        Returns:
            list: Llista d'UUIDs de les cinc millors cançons.
        """
        rankings = []
        for i in self._songs:
            rank = self._songs.get_song_rank(i)
            rankings.append(tuple((i, rank)))

        best_ranking = sorted(rankings, key=lambda x: x[1], reverse=True)
        best_ranking = [key for key, value in best_ranking[:5]]

        similars = set()
        similars.update(best_ranking)
        for i in best_ranking:
            similars.update(self.get_similar(i, 5))
        top_25 = list(similars) # 25 elements
        
        idv_semblanca = dict()
        for i in top_25:
            idv_semblanca[i] = 0
            for x in top_25:
                if x == i:
                    continue
                idv_semblanca[i] += self.song_similarity(i, x)
        
        sorted_items = sorted(idv_semblanca.items(), key=lambda x: x[1], reverse=True)
        top_n_keys = [key for key, value in sorted_items[:5]]
        
        return top_n_keys



def update_from(path):
        files = MusicFiles(path)
        ids = MusicID(files)
        data = MusicData(ids)
        player = MusicPlayer(data)
        search = SearchMetadata(data)
        playlist = PlayList(ids, player)
        return files, ids, data, player, search, playlist

def comprovar_prints(classe):
        print(f"\nObjecte: {type(classe)}")
        if not hasattr(classe, "__len__"):
            print(f"Funció __len__() no implementada per la classe {type(classe)} (intencional)")
        else:
            print(f"Longitud: {len(classe)}")

        # Comprovem representació
        print("\nComprovant representació")
        print(classe)

        # Comprovem iterabilitat i representació dels elements
        print("\nComprovant iteracions:")
        if not hasattr(classe, "__iter__"):
            print(f"Funció __iter__() no implementada per la classe {type(classe)} (intencional)")
        else:
            for i in classe:
                print(i)

        # Comprovem accessibilitats als ítems
        if not hasattr(classe, "__getitem__"):
            print(f"Funció __getitem__() no implementada per la classe {type(classe)} (intencional)")
        else:
            try:
                primer = classe[0]
                print("\nComprovant getitem (epr índex de llista):")
                print(primer)
                print(classe[10])
                print(classe[-2])
                print("\n")
            except KeyError:
                print("\nComprovant getitem (per claus de diccionari):")
                claus = list(classe.keys())
                print(classe[claus[0]])
                print(classe[claus[10]])
                print(classe[claus[-2]])
                print("\n")

def main():

    path = cfg.get_root()
    files, ids, data, player, search, playlist = update_from(path)

    classes_comprovar = [files, ids, data, player, search]
    if len(playlist) > 0:
        classes_comprovar.append(playlist)
    for c in classes_comprovar:
        comprovar_prints(c)

    p1 = PlayList(ids, player, [], "CORPUS-2324-VPL-P2/blues.m3u")
    p2 = PlayList(ids, player, [], "CORPUS-2324-VPL-P2/pop.m3u")
    p3 = PlayList(ids, player, [], "CORPUS-2324-VPL-P2/classical.m3u")
    play_lists = [p1, p2, p3]
    for i in play_lists:
        data.read_playlist(i)

    t5 = search.get_topfive()
    print("\nTop five:")
    for i in t5:
        print("     ", data.get_title(i))

    def buscar_titol(search: SearchMetadata):
        valor = input("\nTítol a buscar a la base de dades: ")
        res = search.title(valor)

        print(f"Resultats: {len(res)}")
        for i in res:
            print(f"Match found: {data.get_title(i)}")

        playlist.read_list(res)
        playlist.play(0)

    buscar_titol(search)

if __name__ == "__main__":
    s = time()
    main()
    f = time()
    t = f-s
    print(f"\nTemps d'execucuó: {t:.4f} segons")