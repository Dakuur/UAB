import cfg
import os.path
import numpy
import uuid
import eyed3
import vlc
import time

class MusicFiles:

    __slots__ = ['_files', '_files_added', '_files_removed', '_iterador']

    def __init__(self, path = None):
        """
        Inicialitza la classe MusicFiles.

        Args:
            path (str, opcional): Ruta del directori d'arxius de música. Si no es proporciona, la instància estarà buida.
        """
        self._files: set = set()  # Conjunt dels els arxius de música emmagatzemats
        self._files_added: set = set()  # Conjunt dels arxius afegits
        self._files_removed: set = set()  # Conjunt dels arxius eliminats
        self._iterador: int = 0 # Índex de l'element actual per a l'iteració

        if path is not None:
            self.reload_fs(path)

    def __repr__(self) -> str:
        """
        Retorna la representació de la llista d'arxius .mp3.

        Returns:
            str: Llista d'arxius.
        """
        #data = '\n  '.join(self._files)
            #versió amb el nom llarg
        data = ',\n  '.join([cfg.get_canonical_pathfile(i) for i in self._files])
            #versió amb el nom curt
        return f"[\n  {data}\n]\n"
    
    def __len__(self) -> int:
        """
        Retorna el nombre d'arxius en la llista.

        Returns:
            int: Longitud de la llista.
        """
        return len(self._files)
    
    def __iter__(self):
        """
        Retorna un iterable que recorre tots els noms d'arxius.

        Returns:
            self: L'objecte per a iterar.
        """
        self._iterador = 0
        return self
    
    def __next__(self):
        """
        Retorna l'element de l'iteració actual.

        Returns:
            str: Nom de l'arxiu .mp3 de l'iteració.
        """
        if self._iterador != len(self):
            data = list(self._files)[self._iterador]
            self._iterador += 1
            return data
        else:
            raise StopIteration
        
    def __hash__(self) -> int: # No implementem aquesta funció ja que creiem que no cal en cas d'una llista
        raise NotImplementedError
    
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
    
    def reload_fs(self, path: str):
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
                    relative = cfg.get_canonical_pathfile(file) # Obté la ruta relativa
                    print("found:  " + relative)
                    file = file.replace(os.sep, '/')
                    new_files.add(file)
        
        self._files = new_files
        self._files_added = new_files.difference(old_files)
        self._files_removed = old_files.difference(new_files)

    def files_added(self) -> list:
        """
        Retorna una llista d'arxius afegits recentment.

        Returns:
            list: Llista d'arxius afegits.
        """
        return list(self._files_added)

    def files_removed(self) -> list:
        """
        Retorna una llista d'arxius eliminats.

        Returns:
            list: Llista d'arxius eliminats.
        """
        return list(self._files_removed)
    
class MusicID:
    
    __slots__ = ['_uuids', '_iterador', '_keys']
    
    def __init__(self, files: MusicFiles = None):
        """
        Inicialitza una classe MusicID.

        Args:
            files (MusicFiles, opcional): Classe MusicFiles amb els arxius de música. Si no es proporciona, el diccionari estarà buit.
        """
        self._uuids: dict = dict()  # Diccionari per emmagatzemar els UUIDs dels arxius
        self._iterador: int = 0 # Índex de l'element actual per a l'iteració
        if files is not None:
            for i in files:
                file = cfg.get_canonical_pathfile(i)
                self.generate_uuid(file)
        self._keys: list = list(self._uuids.keys()) # Llista de keys per a l'iterador
    
    def __repr__(self) -> str:
        """
        Representació de cada arxiu (key) amb el seu UUID corresponent (value).

        Returns:
            str: Llista dels valors del diccionari.
        """
        data = ',\n'.join([f"{i}: \n  {x}" for i, x in self._uuids.items()])
        return "{\n"  + data + "\n}\n"
    
    def __len__(self) -> int:
        """
        Retorna el nombre d'UUIDs emmagatzemats.

        Returns:
            int: Nombre d'UUIDs.
        """
        return len(self._uuids)
    
    def __iter__(self):
        """
        Retorna un iterable que recorre tots els items del diccionari.

        Returns:
            self: L'objecte per a iterar.
        """
        self._iterador = 0
        self._keys = list(self._uuids.keys())
        return self
    
    def __next__(self):
        """
        Retorna l'item (key i value) del diccionari de l'iteració actual.

        Returns:
            tuple: (str, uuid.UUID) -> (File, UUID) de l'iteració.
        """
        if self._iterador != len(self):
            k = self._keys[self._iterador]
            data = self._uuids[k]
            self._iterador += 1
            return k, data # key, uuid
        else:
            raise StopIteration
        
    def __hash__(self, value: str) -> int:
        """
        Retorna el hash del valor passat en el diccionari de la classe.

        Args:
            value (str): Nom de l'arxiu corresponent a la key del valor a consultar.

        Returns:
            int: valor del hash.
        """
        return hash(self._uuids[value])
    
    def __eq__(self, other) -> bool:
        """
        Compara dos classes MusicID i determina si contenen o no els mateixos ítems (File, UUID).

        Args:
            other (MusicID): Classe d'IDs a comparar.

        Returns:
            bool: True si les dues classes contenen les mateixes cançons i IDs
            (amb la meteixa correspondència), False en cas contrari.
        """
        return self._uuids == other._uuids #comprovem que cada clau es correspon amb cada

    def __ne__(self, other: object) -> bool:
        """
        Compara dos classes MusicID i determina si contenen o no els mateixos ítems (File, UUID).

        Args:
            other (MusicID): Classe d'IDs a comparar.

        Returns:
            bool: True si les dues classes no contenen les mateixes cançons i IDs,
            o si difereixen en el valor d'alguna clau, False en cas contrari.
        """
        return not (self == other)
    
    def __lt__(self, other) -> bool:
        """
        Compara les longituds de dos classes MusicID i comprova si la primera és menor que la segona.

        Args:
            other (MusicID): Classe d'IDs a comparar.

        Returns:
            bool: True si el primer diccionari té una longitud menor que el segon, False en cas contrari.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara les longituds de dos classes MusicID i comprova si la primera és menor o igual que la segona.

        Args:
            other (MusicID): Classe d'IDs a comparar.

        Returns:
            bool: True si el primer diccionari té una longitud menor o igual que el segon, False en cas contrari.
        """
        return (self < other) or (len(self) == len(other))

    def generate_uuid(self, file: str) -> str:
        """
        Genera un UUID per a un arxiu donat.

        Args:
            file (str): Ruta de l'arxiu.

        Returns:
            str: UUID generat o None si ja existeix un l'UUID.
        """
        mp3_uuid = uuid.uuid5(uuid.NAMESPACE_URL, file)
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
            str: UUID de l'arxiu o missatge d'error si no existeix.
        """
        try:
            return self._uuids[file]
        except KeyError:
            print("UUID no existent")

    def remove_uuid(self, uuid: str):
        """
        Elimina l'UUID associat a un UUID donat.

        Args:
            uuid (str): UUID a eliminar.
        """
        for k, v in self._uuids.items():
            if v == uuid:
                self._uuids.pop(k)
                return None

class MusicData:
    
    __slots__ = ['_songs', '_iterador', '_keys']

    def __init__(self, ids: MusicID = None):
        """
        Inicialitza una classe MusicData.

        Args:
            ids (MusicID, opcional): Classe MusicID amb els UUIDs dels arxius de música. Si no es proporciona, la classe estarà buida.
        """
        self._songs: dict = dict()  # Diccionari per emmagatzemar les dades de les cançons (UUID -> (fitxer, metadades))
        self._iterador: int = 0 # Índex de l'element actual per a l'iteració
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
        Retorna un iterable que recorre tots els items del diccionari.

        Returns:
            self: L'objecte per a iterar.
        """
        self._iterador = 0
        self._keys = list(self._songs.keys())
        return self
    
    def __next__(self):
        """
        Retorna l'item (key i value) del diccionari de l'iteració actual.

        Returns:
            tuple: (uuid.UUID, eyed3.mp3.Mp3AudioFile) -> (UUID, Metadata) de l'iteració.
        """
        if self._iterador != len(self):
            k = self._keys[self._iterador]
            data = self._songs[k]
            self._iterador += 1
            return k, data # uuid, metadata
        else:
            raise StopIteration
        
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

class MusicPlayer:
    
    __slots__ = ['_songs_data', '_player']

    def __init__(self, data: MusicData = None):
        """
        Inicialitza una classe MusicPlayer.

        Args:
            data (MusicData, opcional): Classe MusicData amb les dades de les cançons. Si no es proporciona, la classe estarà buida.
        """
        self._songs_data: MusicData = data
        self._player: vlc.MediaPlayer = vlc.MediaPlayer

    def __repr__(self) -> str:
        """
        Fa servir la representació del diccionari amb les dades (MusicData)

        Returns:
            str: Llista dels valors de MusicData.
        """
        return repr(self._songs_data)

    def update_data(self, new_data): #per si es vol canviar manualment les dades en format MusicData
        """
        Actualitza les dades de les cançons.

        Args:
            new_data: (MusicData): Noves dades de les cançons.
        """
        self._songs_data = new_data
        return None

    def print_song(self, uuid: str): # Asíncrona
        """
        Mostra informació detallada d'una cançó.

        Args:
            uuid (str): UUID de la cançó.
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

    def play_song(self, uuid: str, mode: int): # Síncrona
        """
        Reprodueix una cançó segons el mode especificat.

        Args:
            uuid (str): UUID de la cançó.
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
    
    __slots__ = ['_uuids', '_player', '_playlist', '_iterador']
    
    def __init__(self, uuids: MusicID = None, player: MusicPlayer = None, playlist: list = None):
        """
        Inicialitza una classe PlayList.

        Args:
            uuids (MusicID, opcional): Classe MusicID amb els UUIDs de les cançons. Si no es proporciona, estarà buida.
            player (MusicPlayer, opcional): Classe MusicPlayer per reproduir les cançons. Si no es proporciona, estarà buida.
            playlist (list, opcional): Llista de UUIDs de les cançons a la llista de reproducció. Si no es proporciona, la llista estarà buida.
        """
        self._uuids: MusicID = uuids  # Llista d'UUIDs
        self._player: MusicPlayer = player  # Reproductor
        self._playlist: list = list()
        self._iterador: int = 0 # Índex de l'element actual per a l'iteració
    
    def __repr__(self) -> str:
        """
        Retorna el diccionari de cançons emmagatzemades (MusicID) i de les guardades a la llista de reproducció.

        Returns:
            str: Representació del diccionari de MusicID i de les cançons afegides a la playlist.
        """
        data_repr = repr(self._player)
        ids_repr = repr(self._uuids)
        llista = ',\n  '.join([f"{i}" for i in self._playlist])
        return f"Diccionari d'IDs:\n{ids_repr}\n" + f"Diccionari de data:\n{data_repr}" + f"\nLlista de reproducció:\n[\n  {llista}\n]\n"
    
    def __len__(self):
        """
        Retorna el nombre de cançons a la llista de reproducció.

        Returns:
            int: Nombre de cançons.
        """
        return len(self._playlist)
    
    def __iter__(self):
        """
        Retorna un iterable que recorre tots els items de la llista de reproducció.

        Returns:
            self: L'objecte per a iterar.
        """
        self._iterador = 0
        return self
    
    def __next__(self):
        """
        Retorna l'element de l'iteració actual.

        Returns:
            uuid.UUID: UUID de l'iteració de la playlist.
        """
        if self._iterador != len(self):
            data = list(self._playlist)[self._iterador]
            self._iterador += 1
            return data
        else:
            raise StopIteration
        
    def __hash__(self) -> int: # No implementem aquesta funció ja que creiem que no cal en cas d'una llista
        raise NotImplementedError
    
    def __eq__(self, other) -> bool:
        """
        Compara dos classes Playlist i determina si contenen o no les mateixes cançons (i en el mateix ordre).

        Args:
            other (Playlist): Classe Playlist a comparar.

        Returns:
            bool: True si les dues llistes són les mateixes, False en cas contrari
        """
        return self._playlist == other._playlist #comprovem que cada clau es correspon amb cada

    def __ne__(self, other: object) -> bool:
        """
        Compara dos classes Playlist i determina si contenen o no les mateixes cançons (i en el mateix ordre).

        Args:
            other (Playlist): Classe Playlist a comparar.

        Returns:
            bool: True si les dues llistes no són les mateixes, False en cas contrari
        """
        return not (self == other)
    
    def __lt__(self, other) -> bool:
        """
        Compara les longituds de dos Playlists.

        Args:
            other (Playlist): Classe Playlist a comparar.

        Returns:
            bool: True si la primera Playlist té menys elements que la segona, False en cas contrari.
        """
        return len(self) < len(other)

    def __le__(self, other) -> bool:
        """
        Compara les longituds de dos Playlists.

        Args:
            other (Playlist): Classe Playlist a comparar.

        Returns:
            bool: True si la primera Playlist té menys o igual nombre d'elements que la segona, False en cas contrari.
        """
        return (self < other) or (len(self) == len(other))

    def load_file(self, filename: str):
        """
        Carrega una llista de reproducció des d'un fitxer.

        Args:
            filename (str): Nom del fitxer de la llista de reproducció en format M3U.
        """
        playlist = []
        self._playlist = playlist

        with open(filename, 'r', encoding='latin-1') as file:
            for line in file:
                line = line.strip()
                if not line.startswith("#") and line.endswith(".mp3"):
                    song_uuid = self._uuids.get_uuid(line)
                    if song_uuid is not None and song_uuid not in playlist:
                        playlist.append(song_uuid)
                        #print(f"Afegit: {line}: {song_uuid}")

        print(f"Llista de reproducció creada amb {len(playlist)} cançons")

        self._playlist = playlist  # Suposem que es reemplaça en comptes d'afegir
        return None

    def play(self, play_mode):
        """
        Reprodueix les cançons de la llista de reproducció segons el mode especificat.

        Args:
            play_mode (int): Mode de reproducció (0: només metadades, 1: metadades + àudio, 2: només àudio).
        """
        for i in self._playlist:
            self._player.play_song(i, play_mode)

    def add_song_at_end(self, uuid: str):
        """
        Afegeix una cançó al final de la llista de reproducció.

        Args:
            uuid (str): UUID de la cançó a afegir.
        """
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

class SearchMetadata:
    
    __slots__ = ['_songs']
    
    def __init__(self, data: MusicData = None):
        """
        Inicialitza una classe SearchMetadata.

        Args:
            data (MusicData, opcional): Classe MusicData amb les dades de les cançons. Si no es proporciona, la classe estarà buida.
        """
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
        
    def __hash__(self) -> int: # No implementem aquesta funció ja que creiem que no cal en cas d'una llista
        raise NotImplementedError
    
    def update_songs(self, data: MusicData):
        """
        Actualitza les dades de les cançons.

        Args:
            data (MusicData): Noves dades de les cançons.
        """
        self._songs = data
        return None

    def search_by_attribute(self, sub: str, attribute: str) -> list:
        """
        Cerca cançons segons un atribut i un subcadena.

        Args:
            sub (str): Subcadena a cercar.
            attribute (str): Atribut per fer la cerca (títol, artista, àlbum o gènere(s)).

        Returns:
            list: Llista d'UUIDs de les cançons que coincideixen amb la cerca.
        """
        results = []
        for i in self._songs.get_uuids():
            attr_value = str(getattr(self._songs, f'get_{attribute}')(i)).upper()
            if attr_value.find(sub.upper()) != -1:
                results.append(i)
            else:
                pass
        return results

    def title(self, sub: str) -> list:
        """
        Cerca cançons pel títol.

        Args:
            sub (str): Subcadena a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb el títol que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(sub), "title")

    def artist(self, sub: str) -> list:
        """
        Cerca cançons per l'artista.

        Args:
            sub (str): Subcadena a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb l'artista que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(sub), "artist")

    def album(self, sub: str) -> list:
        """
        Cerca cançons per l'àlbum.

        Args:
            sub (str): Subcadena a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb l'àlbum que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(sub), "album")

    def genre(self, sub: str) -> list:
        """
        Cerca cançons pel gènere.

        Args:
            sub (str): Subcadena a cercar.

        Returns:
            list: Llista d'UUIDs de les cançons amb el gènere que coincideix amb la cerca.
        """
        return self.search_by_attribute(str(sub), "genre")

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
