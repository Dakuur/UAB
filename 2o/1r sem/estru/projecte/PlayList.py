from MusicID import MusicID
from MusicPlayer import MusicPlayer
import uuid

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
