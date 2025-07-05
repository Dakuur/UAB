from MusicFiles import MusicFiles
from MusicID import MusicID
from MusicData import MusicData
from MusicPlayer import MusicPlayer
from SearchMetadata import SearchMetadata
from PlayList import PlayList

import cfg

path = cfg.get_root()

def update_from_path():
    path = cfg.get_root()
    files = MusicFiles(path)
    ids = MusicID(files)
    data = MusicData(ids)
    player = MusicPlayer(data)
    search = SearchMetadata(data)
    playlist = PlayList(ids, player)
    return files, ids, data, player, search, playlist

files, ids, data, player, search, playlist = update_from_path()

v = list(ids._uuids.values())

n1 = v[0]
n2 = v[3]
n3= v[4]
n10= v[10]

llista = [n1, n2, n3]
playlist = PlayList(ids, player, llista)
data.read_playlist(playlist)

llista = [v[0], v[34], v[4], v[0], v[1], v[9]]
playlist = PlayList(ids, player, llista)
data.read_playlist(playlist)

print(search.get_topfive())

for i in llista:
    print(i)