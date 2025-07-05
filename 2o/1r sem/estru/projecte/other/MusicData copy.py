import MusicID
import eyed3
import numpy
import cfg
import os

class MusicData:
    
    __slots__ = ['_songs', '_keys']

    def __init__(self, ids: MusicID = None):
        """
        Inicialitza una classe MusicData.

        Args:
            ids (MusicID, opcional): Classe MusicID amb els UUIDs dels arxius de música. Si no es proporciona, la classe estarà buida.
        """
        self._songs: dict = dict()  # Diccionari per emmagatzemar les dades de les cançons (UUID -> (fitxer, metadades))
        if ids is not None:
            for filename, song_id in ids:
                self.add_song(song_id, filename)
                self.load_metadata(song_id)
        self._keys: list = list(self._songs.keys()) # Llista de keys per a l'iterador
    
    def __repr__(self) -> str:
        """
        Representació de cada UUID (key) amb la metadata corresponent (value).
        El valor és un objecte de eyed3.mp3.Mp3AudioFile, es representa com ho fa en la llibreria

        Returns:
            str: Llista dels valors del diccionari.
        """
        data = ',\n'.join([f"{i}: \n  {x}" for i, x in self._songs.items()])
        return "{\n"  + data + "\n}\n"
    
    def __len__(self):
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

    def __hash__(self, value: str) -> int:
        """
        Retorna el hash del valor passat en el diccionari de la classe.

        Args:
            value (str): UUID corresponent a la key del valor a consultar.

        Returns:
            int: valor del hash.
        """
        return hash(self._songs[value])
    
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

    def get_uuids(self) -> list:
        """
        Retorna una llista d'UUIDs de les cançons.

        Returns:
            list: Llista d'UUIDs.
        """
        return list(self._songs.keys())

    def add_song(self, uuid: str, file: str):
        """
        Afegeix una cançó a la col·lecció.

        Args:
            uuid (str): UUID de la cançó.
            file (str): Ruta del fitxer de la cançó.
        """
        if uuid not in [None, ""] and file not in [None, ""]:
            self._songs[uuid] = tuple((file, None))  # Tuple (fitxer, metadades)
            return None
        else:
            pass

    def remove_song(self, uuid: str):
        """
        Elimina una cançó de la col·lecció.

        Args:
            uuid (str): UUID de la cançó a eliminar.
        """
        try:
            self._songs.pop(uuid)
            return None
        except KeyError:
            print("UUID no existent")

    def load_metadata(self, uuid: str):
        """
        Carrega les metadades d'una cançó.

        Args:
            uuid (str): UUID de la cançó.
        """
        file = self._songs[uuid][0]
        root = cfg.get_root()
        file = os.path.join(root, file) #Carrega el fitxer amb el path complet, no relatiu
        metadata = eyed3.load(file)
        info = list(self._songs[uuid])
        info[1] = metadata
        self._songs[uuid] = tuple(info)
        return None

    def get_file(self, uuid: str) -> str:
        """
        Retorna el nom del fitxer d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Nom del fitxer.
        """
        try:
            return self._songs[uuid][0]
        except:
            #print("Metadata errònea")
            pass

    def get_filename(self, uuid: str) -> str: #funció duplicada ja que el main.py de Caronte crida la funció amb noms diferents
        """
        Retorna el nom del fitxer d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Nom del fitxer.
        """
        return self.get_file(uuid)

    def get_title(self, uuid: str) -> str:
        """
        Retorna el títol d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Títol de la cançó.
        """
        try:
            return self._songs[uuid][1].tag.title
        except:
            #print("Metadata errònea")
            pass

    def get_artist(self, uuid: str) -> str:
        """
        Retorna l'artista d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Artista de la cançó.
        """
        try:
            return self._songs[uuid][1].tag.artist
        except:
            #print("Metadata errònea")
            pass

    def get_album(self, uuid: str) -> str:
        """
        Retorna l'àlbum d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Àlbum de la cançó.
        """
        try:
            return self._songs[uuid][1].tag.album
        except:
            #print("Metadata errònea")
            pass

    def get_genre(self, uuid: str) -> str:
        """
        Retorna el(s) gènere(s) d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Gènere(s) de la cançó.
        """
        try:
            return self._songs[uuid][1].tag.genre.name
        except:
            #print("Metadata errònea")
            pass
        
    def get_duration(self, uuid: str) -> str:
        """
        Retorna la duració d'una cançó.

        Args:
            uuid (str): UUID de la cançó.

        Returns:
            str: Duració de la cançó.
        """
        try:
            return int(numpy.ceil(self._songs[uuid][1].info.time_secs))
        except:
            #print("Metadata errònea")
            pass
        