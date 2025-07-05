from MusicFiles import MusicFiles
from MusicID import MusicID
from MusicData import MusicData
from MusicPlayer import MusicPlayer
from SearchMetadata import SearchMetadata
from PlayList import PlayList

import cfg

path = cfg.get_root()

#Arxius (1)
files = MusicFiles(path)
print("MusicFiles done")

#Uuid identificadors (2)
ids = MusicID(files)
print("MusicID done")

#Metadata (3)
data = MusicData(ids)
print("MusicData done")

#Reproductor (4)
player = MusicPlayer(data)
print("MusicPlayer done")

#Search metadata (6)
search = SearchMetadata(data)
print("SearchMetadata done")

#Playlist (5 i 7)
playlist = PlayList(ids, player)
print("Playlist done")

def reproduir_una(player: MusicPlayer):
    play_mode = cfg.PLAY_MODE
    print("\nTria una:")
    i = 0
    for file in files:
        print(f"{i}: {cfg.get_canonical_pathfile(file)}")
        i += 1
    i = int(input("\nSelect: "))
    file = files[i]
    player.play_song(ids.get_uuid(file), play_mode)

def buscar_titol(search):
    valor = input("\nTítol a buscar a la base de dades: ")
    res = search.title(valor)
    print(f"Resultats: {len(res)}")
    for i in res:
        print(f"Match found: {data.get_title(i)}")
        playlist.add_song_at_end(i)
    playlist.play(1)

def buscar_artista(search):
    valor = input("\nArtista a buscar a la base de dades: ")
    res = search.artist(valor)
    print(f"Resultats: {len(res)}")
    for i in res:
        print(f"Match found: {data.get_title(i)}")
        playlist.add_song_at_end(i)
    playlist.play(1)

"""
resultat1 = search.title("The")
resultat2 = search.title("Love")

and_result = search.and_operator(resultat1, resultat2)
print(f"Results: {len(and_result)}")
print(f"'Love' and 'The': {and_result}")

or_result = search.or_operator(resultat1, resultat2)
print(f"Results: {len(or_result)}")
print(f"'Love' or 'The': {or_result}")
"""

buscar_titol(search)