import os
import cfg

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
