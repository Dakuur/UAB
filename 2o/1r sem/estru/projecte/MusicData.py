from MusicID import MusicID
import eyed3
import numpy
import cfg
import os
import uuid

from GrafHash import GrafHash
from ElementData import ElementData

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
