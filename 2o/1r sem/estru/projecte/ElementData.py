
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
