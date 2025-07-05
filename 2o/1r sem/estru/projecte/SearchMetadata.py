from MusicData import MusicData
import uuid

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
