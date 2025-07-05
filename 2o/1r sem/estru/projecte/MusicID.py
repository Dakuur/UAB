from MusicFiles import MusicFiles
import uuid
import cfg

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
