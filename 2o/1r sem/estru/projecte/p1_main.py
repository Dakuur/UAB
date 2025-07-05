from MusicFiles import MusicFiles
from MusicID import MusicID
from MusicData import MusicData
from MusicPlayer import MusicPlayer
from SearchMetadata import SearchMetadata
from PlayList import PlayList
from time import time
import cfg

# Funció per actualitzar des d'una ruta donada
def update_from(path):
    # Crear classes a partir de path
    files = MusicFiles(path)
    ids = MusicID(files)
    data = MusicData(ids)
    player = MusicPlayer(data)
    search = SearchMetadata(data)
    playlist = PlayList(ids, player)
    
    return files, ids, data, player, search, playlist

# Funció per verificar i imprimir informació sobre les classes
def comprovar_prints(classe):
    print(f"\nObjecte: {type(classe)}")
    if not hasattr(classe, "__len__"):
        print(f"Funció __len__() no implementada per la classe {type(classe)} (intencionada)")
    else:
        print(f"Longitud: {len(classe)}")

    # Verifiquem representació
    print("\nVerificant representació")
    print(classe)

    # Verifiquem iter i representació dels elements
    print("\nVerificant iteracions:")
    if not hasattr(classe, "__iter__"):
        print(f"Funció __iter__() no implementada per la classe {type(classe)} (intencionada)")
    else:
        for i in classe:
            print(i)

    # Verifiquem accessibilitats als ítems
    if not hasattr(classe, "__getitem__"):
        print(f"Funció __getitem__() no implementada per la classe {type(classe)} (intencionada)")
    else:
        try:
            primer = classe[0]
            print("\nVerificant getitem (per índex de llista):")
            print(primer)
            print(classe[10])
            print(classe[-2])
            print("\n")
        except KeyError:
            print("\nVerificant getitem (per claus de diccionari):")
            claus = list(classe.keys())
            print(classe[claus[0]])
            print(classe[claus[10]])
            print(classe[claus[-2]])
            print("\n")

# Funció per buscar un títol de cançó
def buscar_titol(search: SearchMetadata):
    valor = input("\nTítol a cercar a la base de dades: ")
    res = search.title(valor)

    print(f"Resultats: {len(res)}")
    for i in res:
        print(f"Coincidència trobada: {search._songs.get_title(i)}")

# Funció principal del programa
def main():
    # Obtindre la ruta del root
    path = cfg.get_root()
    
    # Actualitzar classes des de root
    files, ids, data, player, search, playlist = update_from(path)

    # Llista de classes a verificar
    classes_comprovar = [files, ids, data, player, search]
    
    # Verificar i imprimir informació de cada classe a la llista
    for c in classes_comprovar:
        comprovar_prints(c)

    # Crear playlists específiques i llegir informació d'elles
    p1 = PlayList(ids, player, [], "CORPUS-2324-VPL-P2/blues.m3u")
    p2 = PlayList(ids, player, [], "CORPUS-2324-VPL-P2/pop.m3u")
    p3 = PlayList(ids, player, [], "CORPUS-2324-VPL-P2/classical.m3u")
    play_lists = [p1, p2, p3]
    
    # Llegir playlists
    for i in play_lists:
        data.read_playlist(i)

    # Obtenim i imprimir el top 5 de cançons en la colecció
    t5 = search.get_topfive()
    print("\nTop five:")
    for i in t5:
        print("     ", data.get_title(i))

    playlist.add_song_at_end(t5[0])
    playlist.play(0)

    # Cridar la funció de cercar títol
    #buscar_titol(search)

# Executar main
if __name__ == "__main__":
    s = time()
    main()
    f = time()
    t = f - s
    print(f"\nTemps d'execució: {t:.4f} segons")
