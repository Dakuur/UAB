from MusicData import MusicData
import vlc
import time
import cfg
import os
import uuid

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
