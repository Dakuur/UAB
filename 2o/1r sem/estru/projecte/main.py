# -*- coding: utf-8 -*-
"""
main.py : Script Test Part II per validar el resultat
"""

import cfg
from MusicFiles     import MusicFiles
from MusicID        import MusicID
from GrafHash       import GrafHash
from ElementData    import ElementData
from MusicData      import MusicData
from MusicPlayer    import MusicPlayer
from PlayList       import PlayList
from SearchMetadata import SearchMetadata
import os
import sys
import shutil

# RESUM DE LES FUNCIONALITATS:
    # Func1: /* llistat d’arxius mp3 */
    # Func2: /* generar ID de cançons */
    # Func3: /* consultar metadades cançons */
    # Func4: /* reproduir cançó amb metadades */
    # Func5: /* reproduir llistes de reproducció m3u */
    # Func6: /* cercar cançons segons certs criteris */
    # Func7: /* generar llista basada en una cerca */
    # Func2.1: /* iteradors i helpers */
    # Func2.2: /* constructors i extensions */
    # Func2.3: /* class GrafHash i ElementData */
    # Func2.4: /* reimplementació optimitzada classe MusicData */
    # Func2.5: /* xarxa de llistes de cançons */
    # Func2.6: /* generar recomanacions de cançons */

# NOTES:
    # - El test 3.6 te un comportament diferent en Test I i II.
    #   La variable 'empty_metadata' del Corpus indica la diferència.

# Comentaris per a executar el Test d'valuació
print ("    Test validació de la part I del projecte      ")
print ("    ============================================= ")
print ("    ")                                              
print ("    NOTA IMPORTANT: ")                              
print ("       Aquest test és una versió per a executar   ")
print ("       tant dintre del vostre entorn de PC, com a ")
print ("       l'entorn d'avaluació. Aquest arxiu MAIN.PY ")
print ("       serà doncs el mateix, no així el CFG.PY    ")
print ("       Cal utilitzar les mateixes cançons MP3.    ")
print ("    ")                                              
print ("    ")                                              
print ("    Llista de comprovació inicial (MANUAL!):      ")
print ("     - Tots els arxius `*.py` inclouen import cfg ")
print ("     - L'arxiu `cfg.py` te configurat el ROOT_DIR ")
print ("     - Tots els MP3 del Corpus estan a ROOT_DIR   ")
print ("     - El PLAY_MODE es 0 al `cfg.py`              ")
print ("     - Dins el ROOT_DIR existeix `CORPUS.TXT`     ")
print ("    ")                                              
print ("    Llista de requisits de les classes:           ")
print ("     - MusicPlayer: constructor rep un objecte    ")
print ("        MusicData --> MusicPlayer(music_data)     ")
print ("     - SearchMetadata: constructor rep un objecte ")
print ("        MusicData --> SearchMetadata(music_data)  ")
print ("     - MusicData: afegir get_filename(uuid) que   ")
print ("        retorna el string del filename guardat    ")
print ("     - MusicData: cal implementer el mètode       ")
print ("        per a obtenir la longitut: __len()__      ")
print ("     - PlayList: cal implementer el mètode        ")
print ("        per a obtenir la longitut: __len()__      ")
print ("     - Totes les classes de les que es farà test  ")
print ("        han d'implementar el mètode __str__()     ")
print ("    ")                                              
print ("    Llista d'arxius necessaris pels tests:        ")
print ("       * MusicFiles.py                            ")
print ("       * MusicID.py                               ")
print ("       * GrafHash.py                              ")
print ("       * ElementData.py                           ")
print ("       * MusicData.py                             ")
print ("       * MusicPlayer.py                           ")
print ("       * PlayList.py                              ")
print ("       * SearchMetadata.py                        ")
print ("    ")


# Valors específics de la prova
#
#corpus_label   = "CORPUS-2324-LOCAL-P1" # Identificador del CORPUS dins CORPUS.TXT
#empty_metadata = True            # Si es retorna None o "" quan les metadades són empty
#sum_mp3_repo   = 5               # Total MP3 files found from the ROOT_DIR repository
#m3u_file1_repo = "pop.m3u"       # Primer arxiu M3U existent al repository
#m3u_file1_len  = 2               # Nombre total de cançons dins primer arxiu M3U
#m3u_file1_miss = 0               # Nombre de cançons que falten al Corpus dins primer M3U
#m3u_file2_repo = "classical.m3u" # Segon arxiu M3U existent al repository
#m3u_file2_len  = 6               # Nombre total de cançons dins segon arxiu M3U
#m3u_file2_miss = 5               # Nombre de cançons que falten al Corpus dins segon M3U
#m3u_file3_repo = "blues.m3u"     # Tercer arxiu M3U existent al repository
#m3u_file3_len  = 3               # Nombre total de cançons dins tercer arxiu M3U
#m3u_file3_miss = 1               # Nombre de cançons que falten al Corpus dins tercer M3U
#search_63      = [0,2,0,0]       # Resultats Test 6.3
#search_64      = [2,0,0]         # Resultats Test 6.4
#search_65      = [1,1,1,2,1]     # Resultats Test 6.5
#search_75    = [1,2,2,0,4,4,1,1] # Resultats Test 7.5
#
#corpus_label   = "CORPUS-2324-VPL-P1" # Identificador del CORPUS dins CORPUS.TXT
#empty_metadata = True            # Si es retorna None o "" quan les metadades són empty
#sum_mp3_repo   = 138             # Total MP3 files found from the ROOT_DIR repository
#m3u_file1_repo = "pop.m3u"       # Primer arxiu M3U existent al repository
#m3u_file1_len  = 60              # Nombre total de cançons dins primer arxiu M3U
#m3u_file1_miss = 0               # Nombre de cançons que falten al Corpus dins primer M3U
#m3u_file2_repo = "classical.m3u" # Segon arxiu M3U existent al repository
#m3u_file2_len  = 57              # Nombre total de cançons dins segon arxiu M3U  # 2 repetits & 2 no existents
#m3u_file2_miss = 2               # Nombre de cançons que falten al Corpus dins segon M3U
#m3u_file3_repo = "blues.m3u"     # Tercer arxiu M3U existent al repository
#m3u_file3_len  = 20              # Nombre total de cançons dins tercer arxiu M3U
#m3u_file3_miss = 0               # Nombre de cançons que falten al Corpus dins tercer M3U
#search_63      = [10,23,20,49]   # Resultats Test 6.3
#search_64      = [23,58,11]      # Resultats Test 6.4
#search_65      = [2,1,3,3,1]     # Resultats Test 6.5
#search_75 = [40,24,9,2,33,35,13,2] # Resultats Test 7.5
#
corpus_label   = "CORPUS-2324-VPL-P2" # Identificador del CORPUS dins CORPUS.TXT
empty_metadata = False           # Si es retorna None o "" quan les metadades són empty
sum_mp3_repo   = 158             # Total MP3 files found from the ROOT_DIR repository
m3u_file1_repo = "pop.m3u"       # Primer arxiu M3U existent al repository
m3u_file1_len  = 60              # Nombre total de cançons dins primer arxiu M3U
m3u_file1_miss = 0               # Nombre de cançons que falten al Corpus dins primer M3U
m3u_file2_repo = "classical.m3u" # Segon arxiu M3U existent al repository
m3u_file2_len  = 75              # Nombre total de cançons dins segon arxiu M3U  # 2 repetits & 2 no existents
m3u_file2_miss = 2               # Nombre de cançons que falten al Corpus dins segon M3U
m3u_file3_repo = "blues.m3u"     # Tercer arxiu M3U existent al repository
m3u_file3_len  = 20              # Nombre total de cançons dins tercer arxiu M3U
m3u_file3_miss = 0               # Nombre de cançons que falten al Corpus dins tercer M3U
search_63      = [10,23,20,49]   # Resultats Test 6.3
search_64      = [23,58,11]      # Resultats Test 6.4
search_65      = [2,1,3,3,1]     # Resultats Test 6.5
search_75 = [40,24,9,12,33,45,13,2] # Resultats Test 7.5


# Comprovació del CORPUS utilitzat
ROOT = cfg.get_root()
cfck = os.path.join(ROOT, "CORPUS.TXT")

if not os.path.exists(cfck):
    print (" ERROR: Descriptor del CORPUS no trobat!")
    print ("        Comprova la teva configuració.  ")
    print ("")
    sys.exit(1)

with open(cfck, "r") as f:
    l = f.readline().strip()
if (l == corpus_label):
    print ("")
    print ("Comment :=>> CORPUS OK")
    print ("")
else:
    print (" ERROR: El CORPUS configurat no és el d'aquest test!")
    print ("        L'indenditificardor actual és: " + l)
    print ("        i aquest test està fet per:    " + corpus_label)
    print ("")
    sys.exit(1)


# Inicialització del test
grade = 0
acumg = 0

print ("Comment :=>> ============================================")
print ("Comment :=>>                               Iniciant test ")
print ("Comment :=>> ============================================")
print ("Comment :=>> Partial Grade: ", grade)
print ("Comment :=>> --------------------------------------------")


# Instanciació dels objectes
music_collection = MusicFiles()
files_uuids      = MusicID()
music_data       = MusicData()
audio_player     = MusicPlayer(music_data)
list_playlist    = []
searcher_worker  = SearchMetadata(music_data)

if (grade < 0):
    grade = 0
grade += 1
print ("Comment :=>> First initialization complete")
print ("Comment :=>> Partial Grade: ", grade)
print ("Comment :=>> --------------------------------------------")


# ###########################################################################
# Func1: /* llistat d’arxius mp3 */
def func1(debug: bool = False):
    global sum_mp3_repo
    global grade
    global acumg
    global music_collection
    local_grade = 15
    print ("Comment :=>> Test: Func1 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Comprovar que les llistes son buïdes
    # Step 2: Llegir les cançons des de zero
    # Step 3: Tornar a llegir les cançons sense canvis
    # Step 4a: Fem una còpia de dues cançons a un directory temporal
    # Step 4b: Tornem a llegir, ara amb les dues cançons "noves"
    # Step 4c: I tornem a llegir, per a comprovar que les dues "noves" ja hi són
    # Step 5a: Esborrem el directory temporal, eliminant les dues cançons "noves"
    # Step 5b: Tornem a llegir, pero ara faltaran dues cançons eliminades
    # Step 6: Esborrem tot i tornem a llegir les cançons des de zero
    #         perquè poguem consultar amb files_added() tota la col·lecció
    #########################################################################

    if (music_collection is not None) and (isinstance(music_collection,MusicFiles)):
        print("Comment :=>> [1.0] OK: initializing MusicFiles")
        grade += 1
    else:
        print("Comment :=>> [1.0] FAIL: initializing MusicFiles")

    if debug:
        print ("            ", music_collection)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Comprovar que les llistes son buïdes
    if len(music_collection.files_added()) == 0:
        print("Comment :=>> [1.1] OK: empty files_added")
        grade += 1
    else:
        print("Comment :=>> [1.1] FAIL: not empty files_added")

    if len(music_collection.files_removed()) == 0:
        print("Comment :=>> [1:1] OK: empty files_removed")
        grade += 1
    else:
        print("Comment :=>> [1:1] FAIL: not empty files_removed")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Llegir les cançons des de zero
    music_collection.reload_fs(ROOT)
    if debug:
        print ("            ", music_collection)
    add = music_collection.files_added()
    rem = music_collection.files_removed()
    sa  = len(add)
    sr  = len(rem)
    sc  = 0  # Inicialment no hi ha cap cançó!
    grade += 1

    # Adicionalment guardem el path de dos arxius MP3 per a un STEP futur
    if sa < 2:
        raise NotImplementedError("ERROR: Cal tenir-ne com a mínim 2 arxius MP3 al ROOT_DIR!")
    f1 = add[0]
    f2 = add[1]

    if debug :
        print(" new files: " + str(sa) + " and removed: " + str(sr))

    sc = sc + sa - sr
    if (sa == sum_mp3_repo) and (sr == 0) and (sc == sum_mp3_repo):
        print("Comment :=>> [1.2] OK: loading collection from disk")
        grade += 1
    else:
        print("Comment :=>> [1.2] FAIL: loading collection from disk")

    if debug:
        print ("             Total in collection: " + str(sc))
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Tornar a llegir les cançons sense canvis
    music_collection.reload_fs(ROOT)
    if debug:
        print ("            ", music_collection)
    add = music_collection.files_added()
    rem = music_collection.files_removed()
    sa  = len(add)
    sr  = len(rem)
    grade += 1

    if debug :
        print("\n ADDED: ")
        print("  ", *add, sep = "\n  ")
        print("\n REMOVED: ")
        print("  ", *rem, sep = "\n  ")
        print("\n new files: " + str(sa) + " and removed: " + str(sr))

    sc = sc + sa - sr
    if (sa == 0) and (sr == 0) and (sc == sum_mp3_repo):
        print("Comment :=>> [1.3] OK: reloading collection from disk")
        grade += 1
    else:
        print("Comment :=>> [1.3] FAIL: reloading collection from disk")

    if debug:
        print ("             Total in collection: " + str(sc))
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4a: Fem una còpia de dues cançons a un directory temporal
    temp_dir = os.path.join(ROOT , "__TEMP__-__TEMP__")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    shutil.copy(f1, temp_dir)
    shutil.copy(f2, temp_dir)

    #########################################################################
    # Step 4b: Tornem a llegir, ara amb les dues cançons "noves"
    music_collection.reload_fs(ROOT)
    if debug:
        print ("            ", music_collection)
    add = music_collection.files_added()
    rem = music_collection.files_removed()
    sa  = len(add)
    sr  = len(rem)
    grade += 1

    if debug :
        print("\n ADDED: ")
        print("  ", *add, sep = "\n  ")
        print("\n REMOVED: ")
        print("  ", *rem, sep = "\n  ")
        print("\n new files: " + str(sa) + " and removed: " + str(sr))

    sc = sc + sa - sr
    if (sa == 2) and (sr == 0) and (sc == (sum_mp3_repo+2)):
        print("Comment :=>> [1.4] OK: reloading with new files from disk")
        grade += 1
    else:
        print("Comment :=>> [1.4] FAIL: reloading with new files from disk")
    if debug:
        print ("             Total in collection: " + str(sc))

    #########################################################################
    # Step 4c: I tornem a llegir, per a comprovar que les dues "noves" ja hi són
    music_collection.reload_fs(ROOT)
    if debug:
       print ("            ", music_collection)
    add = music_collection.files_added()
    rem = music_collection.files_removed()
    sa  = len(add)
    sr  = len(rem)
    grade += 1

    if debug :
        print("\n ADDED: ")
        print("  ", *add, sep = "\n  ")
        print("\n REMOVED: ")
        print("  ", *rem, sep = "\n  ")
        print("\n new files: " + str(sa) + " and removed: " + str(sr))

    sc = sc + sa - sr
    if (sa == 0) and (sr == 0) and (sc == (sum_mp3_repo+2)):
        print("Comment :=>> [1:4] OK: reloading second time with new files from disk")
        grade += 1
    else:
        print("Comment :=>> [1:4] FAIL: reloading second time with new files from disk")
    if debug:
        print ("             Total in collection: " + str(sc))

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5a: Esborrem el directory temporal, eliminant les dues cançons "noves"
    shutil.rmtree(temp_dir)

    #########################################################################
    # Step 5b: Tornem a llegir, pero ara faltaran dues cançons eliminades
    music_collection.reload_fs(ROOT)
    if debug:
        print ("            ", music_collection)
    add = music_collection.files_added()
    rem = music_collection.files_removed()
    sa  = len(add)
    sr  = len(rem)
    grade += 1

    if debug :
        print("\n ADDED: ")
        print("  ", *add, sep = "\n  ")
        print("\n REMOVED: ")
        print("  ", *rem, sep = "\n  ")
        print("\n new files: " + str(sa) + " and removed: " + str(sr))

    sc = sc + sa - sr
    if (sa == 0) and (sr == 2) and (sc == sum_mp3_repo):
        print("Comment :=>> [1.5] OK: reloading third time with deleted files from disk")
        grade += 1
    else:
        print("Comment :=>> [1.5] FAIL: reloading third time with deleted files from disk")
    if debug:
        print ("             Total in collection: " + str(sc))

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: Esborrem tot i tornem a llegir les cançons des de zero
    #         perquè poguem consultar amb files_added() tota la col·lecció
    music_collection = None          # Equivalent a el·liminar la col·lecció
    music_collection = MusicFiles()  # Començem des de zero
    sc  = 0

    music_collection.reload_fs(ROOT)
    if debug:
        print ("            ", music_collection)
    add = music_collection.files_added()
    rem = music_collection.files_removed()
    sa  = len(add)
    sr  = len(rem)
    grade += 1

    if debug:
        print(" new files: " + str(sa) + " and removed: " + str(sr))

    sc = sc + sa - sr
    if (sa == sum_mp3_repo) and (sr == 0) and (sc == sum_mp3_repo):
        print("Comment :=>> [1.6] OK: restarting reloading last time")
        grade += 1
    else:
        print("Comment :=>> [1.6] FAIL: restarting reloading last time")
    if debug:
        print ("             Total in collection: " + str(sc))

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("            ", music_collection)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func2: /* generar ID de cançons */
def func2(debug: bool = False):
    global grade
    global acumg
    global sum_mp3_repo
    global music_collection
    global files_uuids
    local_grade = 10
    print ("Comment :=>> Test: Func2 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Intenta retornar un UUID d'una col·lecció buida
    # Step 2: Intenta esborrar un UUID d'una col·lecció buida
    # Step 3: Genera els UUIDs de tots els MP3 de la col·lecció
    # Step 4: Intenta generar un UUID repetit
    # Step 5: Insereix i esborra multiples vegades
    # Step 6: Check d'un filename ja inserit que ha de retornar el UUID vàlid
    #########################################################################

    if (files_uuids is not None) and (isinstance(files_uuids,MusicID)):
        print("Comment :=>> [2.0] OK: initializing MusicID")
        grade += 1
    else:
        print("Comment :=>> [2.0] FAIL: initializing MusicID")

    uu_fake  = "00000000-1111-2222-3333-444444444444"
    ff_fake  = ROOT + os.sep + "nofile.mp3"

    if debug:
        print ("            ", files_uuids)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Intenta retornar un UUID d'una col·lecció buida
    uuid = files_uuids.get_uuid(ff_fake)
    if uuid is None:
        print("Comment :=>> [2.1] OK: not returning UUID from empty collection")
        grade += 1
    else:
        print("Comment :=>> [2.1] FAIL: returning UUID from empty collection")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Intenta esborrar un UUID d'una col·lecció buida
    files_uuids.remove_uuid(uu_fake)
    if len(files_uuids) == 0:
        print("Comment :=>> [2.2] OK: not deleting UUID from empty collection")
        grade += 1
    else:
        print("Comment :=>> [2.2] FAIL: deleting UUID from empty collection")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Genera els UUIDs de tots els MP3 de la col·lecció
    total = 0
    elems = len(music_collection.files_added())
    for file in music_collection.files_added():
        file = cfg.get_canonical_pathfile(file)
        uu = files_uuids.generate_uuid(file)
        if not uu is None:
            total += 1

    if (total == elems) and (total > 0) and (len(files_uuids) == sum_mp3_repo):
        print("Comment :=>> [2.3] OK: Generated UUIDs: " + str(total) + " from files: " + str(elems))
        grade += 2
    else:
        print("Comment :=>> [2.3] FAIL: Generated UUIDs: " + str(total) + " from files: " + str(elems))

    if debug:
        print ("            ", files_uuids)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Intenta generar un UUID repetit
    file = music_collection.files_added()[0]
    file = cfg.get_canonical_pathfile(file)
    uu = files_uuids.generate_uuid(file)

    if uu is None:
        print("Comment :=>> [2.4] OK: UUID repeated")
        grade += 1
    else:
        print("Comment :=>> [2.4] FAIL: new UUID created: " + str(uu))

    if debug:
        print ("            ", files_uuids)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Insereix i esborra multiples vegades
    if debug:
        print(" generating UUID for: " + ff_fake)
    un = files_uuids.generate_uuid(ff_fake)
    if debug:
        print ("            ", files_uuids)

    if debug:
        print(" deleting for: " + ff_fake)
    files_uuids.remove_uuid(un)
    if debug:
        print ("            ", files_uuids)

    if debug:
        print(" regenerating for: " + ff_fake)
    un = files_uuids.generate_uuid(ff_fake)
    if debug:
        print ("            ", files_uuids)

    if debug:
        print(" redeleting for: " + ff_fake)
    files_uuids.remove_uuid(un)
    if debug:
        print ("            ", files_uuids)

    if debug:
        print(" redeleting second time for: " + ff_fake)
    files_uuids.remove_uuid(un)
    if debug:
        print ("            ", files_uuids)

    if (un is not None) and (len(files_uuids) == sum_mp3_repo):
        print("Comment :=>> [2.5] OK: UUIDs can be deleted")
        grade += 2
    else:
        print("Comment :=>> [2.5] FAIL: new UUID can't be created after delete it: " + str(uu))

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: Check d'un filename ja inserit que ha de retornar el UUID vàlid
    file = music_collection.files_added()[0]
    if debug:
        print(" Check UUID of file: " + file)
    file = cfg.get_canonical_pathfile(file)
    if debug:
        print("   filepath: " + file)
    uu = files_uuids.get_uuid(file)
    if debug:
        print("   uuid: " + uu)

    if not uu is None:
        print("Comment :=>> [2.6] OK: UUID returned")
        grade += 2
    else:
        print("Comment :=>> [2.6] FAIL: can't recuperate UUID of: " + file)

    #########################################################################
    # End actions
    print ("            ", files_uuids)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func3: /* consultar metadades cançons */
def func3(debug: bool = False):
    global empty_metadata
    global grade
    global acumg
    global sum_mp3_repo
    global music_collection
    global files_uuids
    global music_data
    local_grade = 15
    print ("Comment :=>> Test: Func3 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Intenta recuperar dades abans de ficar cap element
    # Step 2: Insereix una cancó ficticia i intenta llegir les serves metadades
    # Step 3: Intenta re-insertar la cancó anterior i inserir sense sentit
    # Step 4: Esborra (dos vegades) la cancó existent anterior
    # Step 5: Afegeix totes les cançons de la col·lecció
    # Step 6: Intenta retornar metadades abans de llegir-les
    # Step 7: Llegeix les metadades de totes les cançons i recupera després els valors
    #########################################################################

    if (music_data is not None) and (isinstance(music_data,MusicData) and (len(music_data) == 0)):
        print("Comment :=>> [3.0] OK: initializing MusicData (0 elements)")
        grade += 1
    else:
        print("Comment :=>> [3.0] FAIL: initializing MusicData")

    ff_fake  = ROOT + os.sep + "nofile.mp3"
    uu_fake  = "00000000-1111-2222-3333-444444444444"

    if debug:
        print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Intenta recuperar dades abans de ficar cap element
    uu       = files_uuids.generate_uuid(ff_fake)
    title    = music_data.get_title(   uu)
    artist   = music_data.get_artist(  uu)
    album    = music_data.get_album(   uu)
    genre    = music_data.get_genre(   uu)
    filename = music_data.get_filename(uu)  # not in public interface

    if  (title    is not None) and \
        (artist   is not None) and \
        (album    is not None) and \
        (genre    is not None) and \
        (filename is not None) :
            print("Comment :=>> [3.1] FAIL: with empty MusicData class")
    else:
            print("Comment :=>> [3.1] OK: with empty MusicData class")
            grade += 1

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Insereix una cancó ficticia i intenta llegir les serves metadades
    music_data.add_song(uu_fake, ff_fake)
    try:
        music_data.load_metadata(uu_fake)
    except OSError:
        print("Comment :=>> [3.2] OK: not loading fake files")
        grade += 1

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Intenta re-insertar la cancó anterior i inserir sense sentit
    music_data.add_song(uu_fake, ff_fake)  # Reinsereix per substitució
    music_data.add_song(  ""   , ff_fake)  # No insereix res
    music_data.add_song(uu_fake,   ""   )  # No es pot inserrir
    #  Cap de les anteriors crides genera un error directe
    if len(music_data) == 1:
        print("Comment :=>> [3.3] OK: adding fake songs (1 element)")
        grade += 2
    else:
        print("Comment :=>> [3.3] FAIL: adding fake songs")

    if debug:
        print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Esborra (dos vegades) la cancó existent anterior
    music_data.remove_song(uu_fake)  # Esborra una primera vegada
    music_data.remove_song(uu_fake)  # Aquesta segona no pot tornar a esborrar
    #  Cap de les anteriors crides genera un error directe
    if len(music_data) == 0:
        print("Comment :=>> [3.4] OK: testing remove songs (0 elements)")
        grade += 2
    else:
        print("Comment :=>> [3.4] FAIL: testing remove songs")

    if debug:
        print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Afegeix totes les cançons de la col·lecció
    for file in music_collection.files_added():
        file = cfg.get_canonical_pathfile(file)
        uuid = files_uuids.get_uuid(file)
        music_data.add_song(uuid, file)
    if len(music_data) == sum_mp3_repo:
        print("Comment :=>> [3.5] OK: testing adding all songs (" + str(sum_mp3_repo) + " elements)")
        grade += 2
    else:
        print("Comment :=>> [3.5] FAIL: testing adding all songs")

    if debug:
        print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: Intenta retornar metadades abans de llegir-les
    errors = 0
    for file in music_collection.files_added():
        file = cfg.get_canonical_pathfile(file)
        uuid = files_uuids.get_uuid(file)
        title    = music_data.get_title(   uuid)
        artist   = music_data.get_artist(  uuid)
        album    = music_data.get_album(   uuid)
        genre    = music_data.get_genre(   uuid)
        filename = music_data.get_filename(uuid)  # not in public interface
        # Nota: Dues variants segons empty_metadata (Part I o Part II):
        #       TRUE:  abans de llegir les metadades retorna None
        #       FALSE: abans de llegir les metadades retorna ""
        # Nota: Amb la implementació amb GrafHash el filename existeix abans de llegir les metadades,
        #       doncs el que es retorna son Strings buits i un filename.
        #       És dins el CORPUS que s'ha de configurar 'empty_metadata'.
        rtest = False
        if empty_metadata:
            if  (title    is not None) and \
                (artist   is not None) and \
                (album    is not None) and \
                (genre    is not None) and \
                (filename is not None) :
                    rtest = True
        else:
            if  (title    != "") and \
                (artist   != "") and \
                (album    != "") and \
                (genre    != "") and \
                (filename == "") :
                    rtest = True
        if rtest :
                print("Comment :=>> warning with empty metadata elements")
                errors += 1
        else:
            pass

    if errors == 0 :
        print("Comment :=>> [3.6] OK: with empty metadata elements")
        grade += 2
    else:
        print("Comment :=>> [3.6] FAIL: with empty metadata elements")

    if debug:
        print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 7: Llegeix les metadades de totes les cançons i recupera després els valors
    for file in music_collection.files_added():
        file = cfg.get_canonical_pathfile(file)
        uuid = files_uuids.get_uuid(file)
        music_data.load_metadata(uuid)
    for file in music_collection.files_added():
        file = cfg.get_canonical_pathfile(file)
        uuid = files_uuids.get_uuid(file)
        title    = music_data.get_title(   uuid)
        artist   = music_data.get_artist(  uuid)
        album    = music_data.get_album(   uuid)
        genre    = music_data.get_genre(   uuid)
        filename = music_data.get_filename(uuid)  # not in public interface


    grade += 4
    print ("Comment :=>> [3.7] OK: (expected) when loading metadata elements")

    if debug:
        print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    files_uuids.remove_uuid(uu)
    print ("            ", files_uuids)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func4: /* reproduir cançó amb metadades */
def func4(debug: bool = False):
    global sum_mp3_repo
    global grade
    global acumg
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    local_grade = 9
    print ("Comment :=>> Test: Func4 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Intenta reproduir una cançó inexistent
    # Step 2: Intenta cridar a print_song() amb un UUID inexistent
    # Step 3: Intenta cridar a play_file() amb un filepath inexistent
    # Step 4: Reprodueix totes les cancons existents dins la col·lecció
    #########################################################################

    if (audio_player is not None) and (isinstance(audio_player,MusicPlayer)):
        print("Comment :=>> [4.0] OK: initializing MusicPlayer")
        grade += 1
    else:
        print("Comment :=>> [4.0] FAIL: initializing MusicPlayer")

    cmode    = cfg.PLAY_MODE
    uu_fake  = "00000000-1111-2222-3333-444444444444"
    ff_fake  = ROOT + os.sep + "nofile.m3u"

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Intenta reproduir una cançó inexistent
    audio_player.play_song(uu_fake, cmode)
    print("Comment :=>> [4.1] OK: (expected) when playing non existent song")
    grade += 1

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Intenta cridar a print_song() amb un UUID inexistent
    audio_player.print_song(uu_fake)
    print("Comment :=>> [4.2] OK: (expected) when print_song() with fake UUID")
    grade += 2

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Intenta cridar a play_file() amb un filepath inexistent
    audio_player.play_file(ff_fake)
    print("Comment :=>> [4.3] OK: (expected) when play_file() with fake filepath")
    grade += 1

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Reprodueix totes les cancons existents dins la col·lecció
    for idx,file in enumerate(music_collection.files_added()):
        file = cfg.get_canonical_pathfile(file)
        uuid = files_uuids.get_uuid(file)
        audio_player.play_song(uuid, cmode)
    if idx+1 == sum_mp3_repo:
        print("Comment :=>> [4.4] OK: playing all songs (" + str(sum_mp3_repo) + " elements)")
        grade += 4
    else:
        print("Comment :=>> [4.4] FAIL: playing all songs")

    if debug:
        print ("             MusicPlayer(\'Total played songs: " + str(idx+1) + "\')")

    #########################################################################
    # End actions
    print ("            ", files_uuids)
    print ("            ", music_data)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func5: /* reproduir llistes de reproducció m3u */
def func5(debug: bool = False):
    global m3u_file1_repo
    global m3u_file1_len
    global m3u_file1_miss
    global grade
    global acumg
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    local_grade = 17
    print ("Comment :=>> Test: Func5 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Intenta reproduir una PlayList sense cançons
    # Step 2: Intenta llegir un arxiu M3U que no existeix
    # Step 3: Llegeix un arxiu M3U
    # Step 4: Intenta tornar a llegit l'arxiu M3U amb cançons a la PlayList
    # Step 5: Intenta afegir dos arxius M3U amb cançons a la PlayList, però cada load_file() esborra el contingut previ
    # Step 6: Reprodueix la PlayList anterior
    #########################################################################

    if len(list_playlist) == 0:
        print("Comment :=>> [5.0] OK: initializing PlayList list")
        grade += 1
    else:
        print("Comment :=>> [5.0] FAIL: initializing PlayList list")

    cmode    = cfg.PLAY_MODE
    ff_fake  = ROOT + os.sep + "nofile.m3u"
    ff_m3u   = ROOT + os.sep + m3u_file1_repo
    ff2_m3u  = ROOT + os.sep + m3u_file2_repo
    ff3_m3u  = ROOT + os.sep + m3u_file3_repo
    pl       = PlayList(files_uuids, audio_player)

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Intenta reproduir una PlayList sense cançons
    if len(pl) == 0:
        print("Comment :=>> [5.1] OK: empty Playlist")
        grade += 1
    else:
        print("Comment :=>> [5.1] FAIL: non empty Playlist")
    pl.play(cmode)
    print("Comment :=>> [5.1] OK: (expected) when playing an empty PlayList")
    grade += 1

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Intenta llegir un arxiu M3U que no existeix
    try:
        pl.load_file(ff_fake)
        print("Comment :=>> [5.2] FAIL: when loading fake M3U files")
    except FileNotFoundError:
        print("Comment :=>> [5.2] OK: not loading fake M3U files")
        grade += 1

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Llegeix un arxiu M3U
    try:
        pl.load_file(ff_m3u)
        grade += 1
    except FileNotFoundError:
        print("Comment :=>> [5.3] FAIL: loading from M3U file: " + ff_m3u)
    lpl = m3u_file1_len - m3u_file1_miss
    if len(pl) == lpl:
        print("Comment :=>> [5.3] OK: loading (" + str(m3u_file1_len) + " songs; " + str(m3u_file1_miss) + " missing) from M3U file: " + ff_m3u)
        grade += 2
    else:
        print("Comment :=>> [5.3] FAIL: loading " + str(len(pl)) + " songs (from M3U file: " + ff_m3u + " )")

    if debug :
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Intenta tornar a llegit l'arxiu M3U amb cançons a la PlayList
    try:
        pl.load_file(ff_m3u)
        grade += 1
    except FileNotFoundError:
        print("Comment :=>> [5.4] FAIL: reloading from M3U file: " + ff_m3u)
    lpl = m3u_file1_len - m3u_file1_miss
    if len(pl) == lpl:
        print("Comment :=>> [5.4] OK: reloading (" + str(m3u_file1_len) + " songs; " + str(m3u_file1_miss) + " missing) from M3U file: " + ff_m3u)
        grade += 1
    else:
        print("Comment :=>> [5.4] FAIL: reloading " + str(len(pl)) + " songs (from M3U file: " + ff_m3u + " )")

    if debug :
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Intenta afegir dos arxius M3U amb cançons a la PlayList, però cada load_file() esborra el contingut previ
    for ff_m3u,ff_m3u_len,ff_m3u_miss in zip((ff2_m3u,ff3_m3u),(m3u_file2_len,m3u_file3_len),(m3u_file2_miss,m3u_file3_miss)):
        try:
            pl.load_file(ff_m3u)
            grade += 1
        except FileNotFoundError:
            print("Comment :=>> [5.5] FAIL: loading-adding from M3U file: " + ff_m3u)
        lpl = ff_m3u_len - ff_m3u_miss
        if len(pl) == lpl:
            print("Comment :=>> [5.5] OK: loading-adding (" + str(ff_m3u_len) + " songs; " + str(ff_m3u_miss) + " missing) from M3U file: " + ff_m3u)
            grade += 1
        else:
            print("Comment :=>> [5.5] FAIL: adding " + str(len(pl)) + " songs (from M3U file: " + ff_m3u + " )")

    if debug :
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: Reprodueix la PlayList anterior
    pl.play(cmode)
    grade += 4
    print ("Comment :=>> [5.6] OK: (expected) when playing the PlayList")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("            ", pl)
    list_playlist.append(pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func6: /* cercar cançons segons certs criteris */
def func6(debug: bool = False):
    global sum_mp3_repo
    global search_63
    global search_64
    global search_65
    global grade
    global acumg
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 15
    print ("Comment :=>> Test: Func6 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Intenta fer cerques amb valors extrems
    # Step 2: Realitza cerques completes
    # Step 3: Realitza cerca NONE
    # Step 4: Opera les cerques anteriors NONE
    # Step 5: Realitza una cerca complexa
    #########################################################################

    if (searcher_worker is not None) and (isinstance(searcher_worker,SearchMetadata)):
        print("Comment :=>> [6.0] OK: initializing SearchMetadata")
        grade += 1
    else:
        print("Comment :=>> [6.0] FAIL: initializing SearchMetadata")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Intenta fer cerques amb valors extrems
    search1 = searcher_worker.title(  1234567890 )
    search2 = searcher_worker.title(  str('\0')  )
    search3 = searcher_worker.artist( 1234567890 )
    search4 = searcher_worker.artist( str('\0')  )
    search5 = searcher_worker.album(  1234567890 )
    search6 = searcher_worker.album(  str('\0')  )
    search7 = searcher_worker.genre(  1234567890 )
    search8 = searcher_worker.genre(  str('\0')  )

    if ((len(search1) == 0) and \
        (len(search2) == 0) and \
        (len(search3) == 0) and \
        (len(search4) == 0) and \
        (len(search5) == 0) and \
        (len(search6) == 0) and \
        (len(search7) == 0) and \
        (len(search8) == 0)) :
            print("Comment :=>> [6.1] OK: search A returns 0 elements")
            grade += 2
    else:
            print("Comment :=>> [6.1] FAIL: search A returns N elements")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Realitza cerques completes
    search1 = searcher_worker.title(  "" )
    search2 = searcher_worker.artist( "" )
    search3 = searcher_worker.album(  "" )
    search4 = searcher_worker.genre(  "" )

    l1 = len(search1)
    l2 = len(search2)
    l3 = len(search3)
    l4 = len(search4)
    lo = sum_mp3_repo

    if ((l1 == lo) and \
        (l2 == lo) and \
        (l3 == lo) and \
        (l4 == lo)) :
            print("Comment :=>> [6.2] OK: search B returns "+str(lo)+" elements")
            grade += 2
    else:
            print("Comment :=>> [6.2] FAIL: search B returns ("+str(l1)+","+str(l2)+","+str(l3)+","+str(l4)+") elements")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Realitza cerca NONE
    search1 = searcher_worker.title(  "None" )
    search2 = searcher_worker.artist( "NONE" )
    search3 = searcher_worker.album(  "none" )
    search4 = searcher_worker.genre(  "NOne" )

    l1 = len(search1)
    l2 = len(search2)
    l3 = len(search3)
    l4 = len(search4)
    lo1= search_63[0]
    lo2= search_63[1]
    lo3= search_63[2]
    lo4= search_63[3]

    if ((l1 == lo1) and \
        (l2 == lo2) and \
        (l3 == lo3) and \
        (l4 == lo4)) :
            print("Comment :=>> [6.3] OK: search C returns ("+str(lo1)+","+str(lo2)+","+str(lo3)+","+str(lo4)+") elements")
            grade += 3
    else:
            print("Comment :=>> [6.3] FAIL: search C returns ("+str(l1)+","+str(l2)+","+str(l3)+","+str(l4)+") elements")
            print("Comment :=>> [6.3] FAIL: search C returns ("+str(lo1)+","+str(lo2)+","+str(lo3)+","+str(lo4)+") elements")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Opera les cerques anteriors NONE
    search5 = searcher_worker.or_operator(  search1 , search2)
    search6 = searcher_worker.or_operator(  search3 , search4)
    search7 = searcher_worker.and_operator( search5 , search6)

    l5 = len(search5)
    l6 = len(search6)
    l7 = len(search7)
    lo5= search_64[0]
    lo6= search_64[1]
    lo7= search_64[2]
    if debug:
        print("             NONE Search operation : "+str(l5)+","+str(l6)+","+str(l7))
    
    if ((l5 == lo5) and \
        (l6 == lo6) and \
        (l7 == lo7)) :
            print("Comment :=>> [6.4] OK: operations OR:"+str(lo5)+" OR:"+str(lo6)+" AND:"+str(lo7)+" elements")
            grade += 3
    else:
            print("Comment :=>> [6.4] FAIL: operations OR/AND returns ("+str(l5)+","+str(l6)+","+str(l7)+") elements")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Realitza una cerca complexa
    search1 = searcher_worker.title(  "love"  )
    search2 = searcher_worker.title(  "vie"   )
    search3 = searcher_worker.artist( "derek" )

    result1 = searcher_worker.or_operator(  search1 , search2)
    result2 = searcher_worker.and_operator( result1 , search3)

    l1 = len(search1)
    l2 = len(search2)
    l3 = len(search3)
    l8 = len(result1)
    l9 = len(result2)
    lo1= search_65[0]
    lo2= search_65[1]
    lo3= search_65[2]
    lo8= search_65[3]
    lo9= search_65[4]
    if debug:
        print("             Complex Search operation : "+str(l1)+","+str(l2)+","+str(l3)+","+str(l8)+","+str(l9))

    if ((l1 == lo1) and \
        (l2 == lo2) and \
        (l3 == lo3) and \
        (l8 == lo8) and \
        (l9 == lo9)) :
            print("Comment :=>> [6.5] OK: complex search return correct element")
            audio_player.print_song(result2[0])
            grade += 4
    else:
            print("Comment :=>> [6.5] FAIL: in complex search returns ("+str(l1)+","+str(l2)+","+str(l3)+","+str(l8)+","+str(l9)+") elements")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func7: /* generar llista basada en una cerca */
def func7(debug: bool = False):
    global search_75
    global grade
    global acumg
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 18
    print ("Comment :=>> Test: Func7 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Intenta llegir cançons en una PlayList amb 1 element d'un M3U inexistent
    # Step 2: Afegeix multiples vegades el mateix UUID (que és vàlid!)
    # Step 3: Esborra la cançó anterior les N vegades inserida
    # Step 4: Torna a intentar esborrar la cançó anterior dues vegades més
    # Step 5: Fer una cerca complexa i generar una PlayList amb el resultat
    # Step 6: Reprodueix la PlayList anterior
    #########################################################################

    cmode    = cfg.PLAY_MODE
    ff_fake  = ROOT + os.sep + "nofile.m3u"
    uu_fake  = "00000000-1111-2222-3333-444444444444"
    pl       = PlayList(files_uuids, audio_player)

    pl.add_song_at_end(uu_fake)
    if len(pl) == 1:
        print("Comment :=>> [7.0] OK: when initializing a PlayList with 1 element")
        grade += 1
    else:
        print("Comment :=>> [7.0] FAIL: Playlist is not empty")

    if debug:
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Intenta llegir cançons en una PlayList amb 1 element d'un M3U inexistent
    try:
        pl.load_file(ff_fake)
    except OSError:
        print("Comment :=>> [7.1] OK not loading fake M3U files")
        grade += 1
    if len(pl) == 0:
        print("Comment :=>> [7.1] OK: non existent M3U generates empty Playlist")
        grade += 1
    else:
        print("Comment :=>> [7.1] FAIL: Playlist is not empty")

    if debug:
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Afegeix multiples vegades el mateix UUID (que és vàlid!)
    pl.add_song_at_end(uu_fake)
    pl.add_song_at_end(uu_fake)
    pl.add_song_at_end(uu_fake)
    pl.add_song_at_end(uu_fake)
    if len(pl) == 4:
        print("Comment :=>> [7.2] OK: adding repeated songs in a PlayList")
        grade += 2
    else:
        print("Comment :=>> [7.2] FAIL: adding repeated songs in a PlayList")

    if debug:
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Esborra la cançó anterior les N vegades inserida
    pl.remove_first_song()
    pl.remove_last_song()
    pl.remove_first_song()
    pl.remove_last_song()
    if len(pl) == 0:
        print("Comment :=>> [7.3] OK: deleting/consuming songs from a PlayList")
        grade += 2
    else:
        print("Comment :=>> [7.3] FAIL: deleting/consuming songs from a PlayList")

    if debug:
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Torna a intentar esborrar la cançó anterior dues vegades més
    pl.remove_last_song()
    pl.remove_first_song()
    if len(pl) == 0:
        print("Comment :=>> [7.4] OK: deleting/consuming songs from an empty PlayList")
        grade += 1
    else:
        print("Comment :=>> [7.4] FAIL: deleting/consuming songs from an empty PlayList")

    if debug:
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Fer una cerca complexa i generar una PlayList amb el resultat
    search1 = searcher_worker.album( "live"  )
    search2 = searcher_worker.genre( "BLUES" )
    search3 = searcher_worker.genre( "FOLK"  )
    search4 = searcher_worker.genre( "JAZZ"  )

    result1 = searcher_worker.or_operator(  search2 , search3)
    result2 = searcher_worker.or_operator(  result1 , search4)
    result3 = searcher_worker.and_operator( result2 , search1)

    l1 = len(search1)
    l2 = len(search2)
    l3 = len(search3)
    l4 = len(search4)
    l7 = len(result1)
    l8 = len(result2)
    l9 = len(result3)
    lo1= search_75[0]
    lo2= search_75[1]
    lo3= search_75[2]
    lo4= search_75[3]
    lo7= search_75[4]
    lo8= search_75[5]
    lo9= search_75[6]
    lo0= search_75[7]
    if debug:
        print("             Complex Search operation with Playlist : "+str(l1)+","+str(l2)+","+str(l3)+","+str(l4)+","+str(l7)+","+str(l8)+","+str(l9))

    if ((l1 == lo1) and \
        (l2 == lo2) and \
        (l3 == lo3) and \
        (l4 == lo4) and \
        (l7 == lo7) and \
        (l8 == lo8) and \
        (l9 == lo9)) :
            print("Comment :=>> [7.5] OK: last complex search "+str(l9)+" elements")
            grade += 4
    else:
            print("Comment :=>> [7.5] FAIL: in last complex search returns ("+str(l1)+","+str(l2)+","+str(l3)+","+str(l4)+","+str(l7)+","+str(l8)+","+str(l9)+") elements")

    final_result = []
    for uuid in result3:
        if debug:
            print(uuid)
        if ' ' in music_data.get_title(uuid) != True:
            if debug:
                print(music_data.get_title(uuid))
        else:
            if debug:
                print(music_data.get_title(uuid))
            final_result.append(uuid)

    for uuid in final_result:
        pl.add_song_at_end(uuid)

    if len(pl) == lo0:
        print("Comment :=>> [7:5] OK: last complex search found correct "+str(lo0)+" elements")
        grade += 4
    else:
        print("Comment :=>> [7:5] FAIL: deleting/consuming songs from an empty PlayList")

    if debug:
        print ("            ", pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: Reprodueix la PlayList anterior
    pl.play(cmode)
    print("Comment :=>> [7.6] OK (expected) when playing the PlayList of the Search")
    grade += 2

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("            ", pl)
    list_playlist.append(pl)
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade


# ###########################################################################
# Func2.1: /* Iteradors i Helpers */
def func21(debug: bool = False):
    global sum_mp3_repo
    global m3u_file3_len
    global grade
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 23
    print ("Comment :=>> Test: Func21 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: __repr__():  MusicFiles, MusicID, GrafHash, ElementData, MusicData, MusicPlayer, PlayList, SearchMetadata .
    # Step 2: __iter__():            , MusicID, GrafHash,            , MusicData,            , PlayList,                .
    # Step 3: __hash__():            ,        ,         , ElementData,          ,            ,         ,                .
    # Step 4: __eq__(),__ne__(),__lt__():     ,         , ElementData,          ,            ,         ,                .
    # Step 5: __len__():             , MusicID, GrafHash,            , MusicData,            , PlayList,                .
    #########################################################################

    first_playlist = list_playlist[0]
    gh_null = GrafHash()
    ed_null  = ElementData("fake", filename="non-existent.mp3")
    ed2_null = ElementData("seco", filename="non-second.mp3")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Comprovem __repr__() amb les classes rellevants
    check = { \
             ( "MusicFiles"     ,music_collection ) , \
             ( "MusicID"        ,files_uuids      ) , \
             ( "GrafHash"       ,gh_null          ) , \
             ( "ElementData"    ,ed_null          ) , \
             ( "MusicData"      ,music_data       ) , \
             ( "PlayList"       ,first_playlist   ) , \
             ( "MusicPlayer"    ,audio_player     ) , \
             ( "SearchMetadata" ,searcher_worker  ) , \
            }

    for nam,obj in check:
        try:
            print("             " + nam + " : " + obj.__repr__())
            grade += 1
            print("Comment :=>> [21.1] OK: "   + nam + ".__repr__()")
        except:
            print("Comment :=>> [21.1] FAIL: " + nam + ".__repr__()")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Comprovem __iter__() amb les classes rellevants
    check = { \
             ( "MusicID"        ,files_uuids     ,sum_mp3_repo  ) , \
             ( "GrafHash"       ,gh_null         ,0             ) , \
             ( "MusicData"      ,music_data      ,sum_mp3_repo  ) , \
             ( "PlayList"       ,first_playlist  ,m3u_file3_len ) , \
            }

    for nam,obj,tot in check:
        count = 0
        try:
            for e in obj:
                print("             " + nam + " Iter: " + str(e))
                count+=1
            if (count == tot):
                grade += 1
                print("Comment :=>> [21.2] OK: ("   + str(count) + " items) in " + nam + ".__iter__()")
            else:
                raise Exception("[21.2]")
        except:
            print("Comment :=>> [21.2] FAIL: ("     + str(count) + " items) in " + nam + ".__iter__()")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Comprovem __hash__() amb les classes rellevants
    check = { \
             ( "ElementData"    ,ed_null          ) , \
             ( "ElementData"    ,ed2_null         ) , \
            }

    checkout = [None,None]
    for ind,(nam,obj) in enumerate(check):
        try:
            checkout[ind] = obj.__hash__()
            print("             " + nam + " Hash: " + str(checkout[ind]))
        except:
            print("Comment :=>> [21.3] FAIL: " + nam + ".__hash__()")

    if ( checkout[0] != None and 
         checkout[1] != None and
         checkout[0] != checkout[1] ):
        grade += 1
        print("Comment :=>> [21.3] OK: all checks of __hash__()")
    else:
        print("Comment :=>> [21.3] FAIL: checks of __hash__()")
        

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Comprovem __eq__(),__ne__(),__lt__() amb les classes rellevants
    check = { \
             ( "ElementData"    ,ed_null  ,ed2_null ) , \
            }

    checkout = [None,None]
    for nam,obj,obj2 in check:
        try:
            out = obj.__eq__(obj)
            print("             " + nam + " EQ(A,A): " + str(out))
            if ( out == True ):
                grade += 1
                print("Comment :=>> [21.4] OK: "   + nam + ".__eq__(A)")
            else:
                raise Exception("[21.4]")
        except:
            print("Comment :=>> [21.4] FAIL: "     + nam + ".__eq__(A)")

        try:
            out = obj.__eq__(obj2)
            print("             " + nam + " EQ(A,B): " + str(out))
            if ( out == False ):
                grade += 1
                print("Comment :=>> [21.4] OK: "   + nam + ".__eq__(B)")
            else:
                raise Exception("[21.4]")
        except:
            print("Comment :=>> [21.4] FAIL: "     + nam + ".__eq__(B)")

        try:
            out = obj.__ne__(obj)
            print("             " + nam + " NE(A,A): " + str(out))
            if ( out == False ):
                grade += 1
                print("Comment :=>> [21.4] OK: "   + nam + ".__ne__(A)")
            else:
                raise Exception("[21.4]")
        except:
            print("Comment :=>> [21.4] FAIL: "     + nam + ".__ne__(A)")

        try:
            out = obj.__ne__(obj2)
            print("             " + nam + " NE(A,B): " + str(out))
            if ( out == True ):
                grade += 1
                print("Comment :=>> [21.4] OK: "   + nam + ".__ne__(B)")
            else:
                raise Exception("[21.4]")
        except:
            print("Comment :=>> [21.4] FAIL: "     + nam + ".__ne__(B)")

        try:
            out = obj.__lt__(obj)
            print("             " + nam + " LT(A,A): " + str(out))
            if ( out == False ):
                grade += 1
                print("Comment :=>> [21.4] OK: "   + nam + ".__lt__(A)")
            else:
                raise Exception("[21.4]")
        except:
            print("Comment :=>> [21.4] FAIL: "     + nam + ".__lt__(A)")

        try:
            out = obj.__lt__(obj2)
            out2 = obj2.__lt__(obj)
            print("             " + nam + " LT(A,B): " + str(out))
            print("             " + nam + " LT(B,A): " + str(out2))
            if ( out != out2 ):
                grade += 1
                print("Comment :=>> [21.4] OK: "   + nam + ".__lt__(A,B)")
            else:
                raise Exception("[21.4]")
        except:
            print("Comment :=>> [21.4] FAIL: "     + nam + ".__lt__(A,B)")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Comprovem __len__() amb les classes rellevants
    check = { \
             ( "MusicID"        ,files_uuids     ,sum_mp3_repo  ) , \
             ( "GrafHash"       ,gh_null         ,0             ) , \
             ( "MusicData"      ,music_data      ,sum_mp3_repo  ) , \
             ( "PlayList"       ,first_playlist  ,m3u_file3_len ) , \
            }

    for nam,obj,tot in check:
        try:
            res = len(obj)
            print("             " + nam + " Len: " + str(res))
            if (res == tot):
                grade += 1
                print("Comment :=>> [21.5] OK: len("   + nam + ")")
            else:
                raise Exception("[21.5]")
        except:
            print("Comment :=>> [21.5] FAIL: len(" + nam + ")")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade
    #########################################################################


# ###########################################################################
# Func2.2: /* Constructors i Extensions */
def func22(debug: bool = False):
    global grade
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 21
    print ("Comment :=>> Test: Func22 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: MusicPlayer.__init__( obj_music_data: MusicData )
    # Step 2: SearchMetadata.__init__( obj_music_data: MusicData )
    # Step 3: PlayList.__init__( obj_music_id: MusicID, obj_music_player: MusicPlayer )
    # Step 4: __slots__ = []
    # Step 5: MusicData.get_filename(uuid: str) -> str
    # Step 6: MusicData.get_duration(uuid: str) -> int
    #########################################################################

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Comprovem __init__() de MusicPlayer

    try:
        MusicPlayer()
        print("Comment :=>> [22.1] FAIL: incorrect MusicPlayer() constructor")
    except:
        grade += 1
        print("Comment :=>> [22.1] OK: non incorrect MusicPlayer() constructor")
    try:
        MusicPlayer("empty","empty")
        print("Comment :=>> [22.1] FAIL: empty MusicPlayer() constructor")
    except:
        grade += 1
        print("Comment :=>> [22.1] OK: non empty MusicPlayer() constructor")
    try:
        MusicPlayer(music_data)
        grade += 1
        print("Comment :=>> [22.1] OK: correct MusicPlayer() constructor")
    except:
        print("Comment :=>> [22.1] FAIL: with correct MusicPlayer() constructor")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Comprovem __init__() de SearchMetadata

    try:
        SearchMetadata()
        print("Comment :=>> [22.2] FAIL: incorrect SearchMetadata() constructor")
    except:
        grade += 1
        print("Comment :=>> [22.2] OK: non incorrect SearchMetadata() constructor")
    try:
        SearchMetadata("empty")
        print("Comment :=>> [22.2] FAIL: empty SearchMetadata() constructor")
    except:
        grade += 1
        print("Comment :=>> [22.2] OK: non empty SearchMetadata() constructor")
    try:
        SearchMetadata(music_data)
        grade += 1
        print("Comment :=>> [22.2] OK: correct SearchMetadata() constructor")
    except:
        print("Comment :=>> [22.2] FAIL: with correct SearchMetadata() constructor")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Comprovem __init__() de PlayList

    try:
        PlayList()
        print("Comment :=>> [22.3] FAIL: incorrect PlayList() constructor")
    except:
        grade += 1
        print("Comment :=>> [22.3] OK: non incorrect PlayList() constructor")
    try:
        PlayList("empty")
        print("Comment :=>> [22.3] FAIL: empty PlayList() constructor")
    except:
        grade += 1
        print("Comment :=>> [22.3] OK: non empty PlayList() constructor")
    try:
        PlayList(files_uuids,audio_player)
        grade += 1
        print("Comment :=>> [22.3] OK: correct PlayList() constructor")
    except:
        print("Comment :=>> [22.3] FAIL: with correct PlayList() constructor")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Comprovem ús de __slots__ en totes les classes

    first_playlist = list_playlist[0]
    gh_null = GrafHash()
    ed_null = ElementData("fake", filename="non-existent.mp3")

    check = { \
             ( "MusicFiles"     ,music_collection ) , \
             ( "MusicID"        ,files_uuids      ) , \
             ( "GrafHash"       ,gh_null          ) , \
             ( "ElementData"    ,ed_null          ) , \
             ( "MusicData"      ,music_data       ) , \
             ( "PlayList"       ,first_playlist   ) , \
             ( "MusicPlayer"    ,audio_player     ) , \
             ( "SearchMetadata" ,searcher_worker  ) , \
            }

    for nam,obj in check:
        try:
            print("             " + nam + " __slots__: ",end="")
            for i in obj.__slots__:
                print(i, end=" ")
            print(" ")
            grade += 1
            print("Comment :=>> [22.4] OK: " + nam + ".__slots__")
        except:
            print(" ")
            print("Comment :=>> [22.4] FAIL: " + nam + ".__slots__")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: MusicData.get_filename(uuid: str) -> str

    uu_fake  = "00000000-1111-2222-3333-444444444444"

    file     = next(files_uuids.__iter__())  # Get the first element of the iterator
    uu_valid = files_uuids.get_uuid(file)    # Get the UUID

    f1 = music_data.get_filename(uu_fake)
    f2 = music_data.get_filename(uu_valid)

    if debug:
        print ("            ", uu_fake,  " -> ", f1)
        print ("            ", uu_valid, " -> ", f2)

    if f1 is None:
        print("Comment :=>> [22.5] OK: get_filename(fake_UUID) returns None")
        grade += 1
    else:
        print("Comment :=>> [22.5] FAIL: get_filename(fake_UUID) returns: " + f1)

    if isinstance(f2, str) and (os.path.splitext(f2)[1].lower() == ".mp3"):
        print("Comment :=>> [22.5] OK: get_filename(UUID) returns a valid MP3 file")
        grade += 1
    else:
        print("Comment :=>> [22.5] FAIL: get_filename(UUID) returns: " + f2)

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: MusicData.get_duration(uuid: str) -> int

    f1 = music_data.get_duration(uu_fake)
    f2 = music_data.get_duration(uu_valid)

    if debug:
        print ("            ", uu_fake,  " -> ", f1)
        print ("            ", uu_valid, " -> ", f2)

    if f1 <= 0:
        print("Comment :=>> [22.6] OK: get_duration(fake_UUID) returns less than 0")
        grade += 1
    else:
        print("Comment :=>> [22.6] FAIL: get_duration(fake_UUID) returns: " + f1)

    if f2 > 0:
        print("Comment :=>> [22.6] OK: get_filename(UUID) returns greater than 0")
        grade += 1
    else:
        print("Comment :=>> [22.6] FAIL: get_filename(UUID) returns: " + f2)

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade
    #########################################################################


# ###########################################################################
# Func2.3: /* classes GrafHash i ElementData */
def func23(debug: bool = False):
    global grade
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 28
    print ("Comment :=>> Test: Func23 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: class ElementData: attributes
    # Step 2: class ElementData: functions
    # Step 3: class GrafHash: check Vertex operations
    # Step 4: class GrafHash: __getitem__, __delitem__, __contains__ 
    # Step 5: class GrafHash: check Edges operations
    # Step 6: class GrafHash: check camiMesCurt()
    #########################################################################

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Comprovem els atributs d'ElementData

    elem1 = ElementData("element-1", filename="non-existent.mp3")
    elem2 = ElementData("element-2", filename="non-existent.mp3")
    elem3 = ElementData("element-3", filename="non-existent3.mp3")
    elem4 = ElementData("element-3", filename="non-existent3.mp3")

    try:
        elem1.title  = "title1"
        elem1.artist = "artist1"
        elem1.album  = "album1"
        elem1.genre  = "genre1"
        elem1.duration = 1
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData setters")
    except:
        print("Comment :=>> [23.1] FAIL: ElementData setters")

    try:
        elem1.filename("filename1")
        print("Comment :=>> [23.1] FAIL: ElementData.filename attribute is mutable")
    except:
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData.filename attribute is inmutable")

    if elem1.title == "title1":
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData.title")
    else:
        print("Comment :=>> [23.1] FAIL: ElementData.title")
    if elem1.artist == "artist1":
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData.artist")
    else:
        print("Comment :=>> [23.1] FAIL: ElementData.artist")
    if elem1.album == "album1":
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData.album")
    else:
        print("Comment :=>> [23.1] FAIL: ElementData.album")
    if elem1.genre == "genre1":
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData.genre")
    else:
        print("Comment :=>> [23.1] FAIL: ElementData.genre")
    if elem1.duration == 1:
        grade += 1
        print("Comment :=>> [23.1] OK: ElementData.duration")
    else:
        print("Comment :=>> [23.1] FAIL: ElementData.duration")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Comprovem les funcions d'ElementData

    if (elem1 == elem2) and (elem3 == elem4):
        grade += 1
        print("Comment :=>> [23.2] OK: ElementData equality")
    else:
        print("Comment :=>> [23.2] FAIL: ElementData equality")

    if (elem1 != elem3) and (elem1 != elem4) and (elem2 != elem3) and (elem2 != elem4):
        grade += 1
        print("Comment :=>> [23.2] OK: ElementData inequality")
    else:
        print("Comment :=>> [23.2] FAIL: ElementData inequality")

    if (elem1 < elem3) and (elem1 < elem4) and (elem3 > elem2) and (elem4 > elem2):
        grade += 1
        print("Comment :=>> [23.2] OK: ElementData less-than")
    else:
        print("Comment :=>> [23.2] FAIL: ElementData less-than")
    if (hash(elem1) == hash(elem2)) and (hash(elem1) != hash(elem3)) and (hash(elem2) != hash(elem4)) and (hash(elem3) == hash(elem4)):
        grade += 1
        print("Comment :=>> [23.2] OK: ElementData hashing")
    else:
        print("Comment :=>> [23.2] FAIL: ElementData hashing")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Comprovem les operacions amb Nodes a GrafHash
    
    uu_fake_1 = "00000001-1111-2222-3333-444444444444"
    uu_fake_2 = "00000002-1111-2222-3333-444444444444"
    uu_fake_3 = "00000003-1111-2222-3333-444444444444"
    uu_fake_4 = "00000004-1111-2222-3333-444444444444"

    #graf = GrafHash()
    graf = GrafHash(digraf=True)  # NOTA: Per defecte és un graf NO dirigit!

    len_ini = len(graf)
    try:
        graf.insert_vertex()
    except:
        if (len(graf) == 0) and (len_ini == 0):
            grade += 1
            print("Comment :=>> [23.3] OK: GrafHash insert_vertex()")
        else:
            print("Comment :=>> [23.3] FAIL: GrafHash insert_vertex() result")
    else:
        print("Comment :=>> [23.3] FAIL: GrafHash insert_vertex()")

    try:
        graf.insert_vertex(uu_fake_1,uu_fake_1)
    except:
        pass
    if len(graf) == 0:
        grade += 1
        print("Comment :=>> [23.3] OK: GrafHash insert_vertex(k,obj) fails without Element")
    else:
        print("Comment :=>> [23.3] FAIL: GrafHash insert_vertex(k,obj) fails without Element")

    try:
        graf.insert_vertex(uu_fake_1,elem1)
        graf.insert_vertex(uu_fake_2,elem2)
        graf.insert_vertex(uu_fake_3,elem3)
    except:
        print("Comment :=>> FAIL: GrafHash insert_vertex(k,e)")
    else:
        if len(graf) == 3:
            grade += 1
            print("Comment :=>> [23.3] OK: GrafHash insert_vertex(k,e)")
        else:
            print("Comment :=>> [23.3] FAIL: GrafHash insert_vertex(k,e) result")

    try:
        graf.insert_vertex(uu_fake_1,elem1)
    except:
        pass
    if len(graf) == 3:
        grade += 1
        print("Comment :=>> [23.3] OK: GrafHash insert_vertex(k,e) with duplicate Key")
    else:
        print("Comment :=>> [23.3] FAIL: GrafHash insert_vertex(k,e) with duplicate Key")

    try:
        graf.insert_vertex(uu_fake_4,elem1)
    except:
        pass
    if len(graf) == 4:
        grade += 1
        print("Comment :=>> [23.3] OK: GrafHash insert_vertex(k,e) with duplicate Element")
    else:
        print("Comment :=>> [23.3] FAIL: GrafHash insert_vertex(k,e) with duplicate Element")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Comprovem les eliminacions i consultes de Nodes a GrafHash

    result = False
    try:
        result = uu_fake_1 in graf
    except:
        print("Comment :=>> [23.4] FAIL: GrafHash __contains__")
    else:
        if result == True:
            grade += 1
            print("Comment :=>> [23.4] OK: GrafHash __contains__ check Key")
        else:
            print("Comment :=>> [23.4] FAIL: GrafHash __contains__ check Key")

    uu_fake_5 = "00000005-1111-2222-3333-444444444444"
    result = False
    try:
        result = uu_fake_5 in graf
    except:
        print("Comment :=>> [23.4] FAIL: GrafHash __contains__ (2)")
    else:
        if result == False:
            grade += 1
            print("Comment :=>> [23.4] OK: GrafHash __contains__ check unknown Key")
        else:
            print("Comment :=>> [23.4] FAIL: GrafHash __contains__ check unknown Key")

    res_elem = None
    try:
        res_elem = graf[uu_fake_1]
    except:
        print("Comment :=>> [23.4] FAIL: GrafHash __getitem__")
    else:
        if res_elem == elem1:
            grade += 1
            print("Comment :=>> [23.4] OK: GrafHash __getitem__ with correct Key")
        else:
            print("Comment :=>> [23.4] FAIL: GrafHash __getitem__ with correct Key")

    res_elem = None
    try:
        res_elem = graf[uu_fake_5]
    except:
        print("Comment :=>> [23.4] FAIL: GrafHash __getitem__ (2)")
    else:
        if res_elem == None:
            grade += 1
            print("Comment :=>> [23.4] OK: GrafHash __getitem__ with incorrect Key")
        else:
            print("Comment :=>> [23.4] FAIL: GrafHash __getitem__ with incorrect Key")

    try:
        del graf[uu_fake_1]
        del graf[uu_fake_2]
    except:
        print("Comment :=>> [23.4] FAIL: GrafHash __delitem__")
    else:
        if len(graf) == 2:
            grade += 1
            print("Comment :=>> [23.4] OK: GrafHash __delitem__ with correct Key")
        else:
            print("Comment :=>> [23.4] FAIL: GrafHash __delitem__ with correct Key")

    if (uu_fake_1 not in graf) and (uu_fake_2 not in graf):
        grade += 1
        print("Comment :=>> [23.4] OK: GrafHash deleted Key")
    else:
        print("Comment :=>> [23.4] FAIL: GrafHash deleted Key")

    result = True
    for e in graf:
        if (e not in {uu_fake_3,uu_fake_4}):
            result = False
    if result == True:
        grade += 1
        print("Comment :=>> [23.4] OK: GrafHash Iteration result")
    else:
        print("Comment :=>> [23.4] FAIL: GrafHash Iteration result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Comprovem les operacions amb Arestes a GrafHash

    graf.insert_vertex(uu_fake_1,elem1)
    graf.insert_vertex(uu_fake_2,elem2)
    
    try:
        graf.insert_edge()
    except:
        grade += 1
        print("Comment :=>> [23.5] OK: GrafHash insert_edge()")
    else:
        print("Comment :=>> [23.5] FAIL: GrafHash insert_edge()")

    try:
        graf.insert_edge(uu_fake_1,uu_fake_2,1)
        graf.insert_edge(uu_fake_2,uu_fake_3,2)
        graf.insert_edge(uu_fake_3,uu_fake_4,3)
        graf.insert_edge(uu_fake_4,uu_fake_1,4)
    except:
        print("Comment :=>> [23.5] FAIL: GrafHash insert_edge(k1,k2)")
    else:
        grade += 1
        print("Comment :=>> [23.5] OK: GrafHash insert_edge(k1,k2)")

    try:
        graf.insert_edge(uu_fake_1,uu_fake_2,5)
        graf.insert_edge(uu_fake_2,uu_fake_3,6)
        graf.insert_edge(uu_fake_3,uu_fake_4,7)
        graf.insert_edge(uu_fake_4,uu_fake_1,8)
    except:
        print("Comment :=>> [23.5] FAIL: GrafHash insert_edge(k1,k2) in previous edges")
    else:
        grade += 1
        print("Comment :=>> [23.5] OK: GrafHash insert_edge(k1,k2) in previous edges")

    result = True
    for e in graf.edges_out(uu_fake_1):
        if (e not in {uu_fake_2,uu_fake_4}):
            result = False
    for e in graf.edges_out(uu_fake_2):
        if (e not in {uu_fake_1,uu_fake_3}):
            result = False
    for e in graf.edges_out(uu_fake_3):
        if (e not in {uu_fake_2,uu_fake_4}):
            result = False
    for e in graf.edges_out(uu_fake_4):
        if (e not in {uu_fake_3,uu_fake_1}):
            result = False
    if result == True:
        grade += 1
        print("Comment :=>> [23.5] OK: GrafHash edges_out() result")
    else:
        print("Comment :=>> [23.5] FAIL: GrafHash edges_out() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 6: Comprovem la cerca de camiMesCurt() amb Dijkstra a GrafHash

    result = None
    try:
        result = graf.camiMesCurt(uu_fake_3,uu_fake_1)
    except:
        pass
    if (len(result) == 3) and (result[1] == uu_fake_4):
        grade += 1
        print("Comment :=>> [23.6] OK: GrafHash camiMesCurt() result")
    else:
        print("Comment :=>> [23.6] FAIL: GrafHash camiMesCurt() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade
    #########################################################################


# ###########################################################################
# Func2.4: /* Reimplementació optimitzada classe MusicData */
def func24(debug: bool = False):
    global grade
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 22
    print ("Comment :=>> Test: Func24 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:  (Nota: GrafHash i ElementData comprovats individualment a 2.3)
    # Step 1: MusicData.get_song_rank(uuid: str) -> int
    # Step 2: MusicData.get_next_songs(uuid: str) -> iterator (uuid, value)
    # Step 3: MusicData.get_previous_songs(uuid: str) -> iterator (uuid, value)
    # Step 4: MusicData.get_song_distance(uuid1: str, uuid2: str) -> (nodes, value)
    # Step 5: remove song with edges
    #########################################################################

    temp_songs = []
    for _, s in zip(range(3), files_uuids):
        temp_songs.append(files_uuids.get_uuid(s))
    temp_song1 = temp_songs[0]
    temp_song2 = temp_songs[1]
    temp_song3 = temp_songs[2]

    # NOTA: No podem garantir l'ordre a cada implementació, per tant utilitzem arxius concrets!
    temp_songs = [ "Blues/Arround_the_Cliffs_ID_1202.mp3"            , \
                   "Blues/Checkie_Brown_-_09_-_Mary_Roose_CB_36.mp3" , \
                   "Blues/Do_not_forget_me_ID_1028.mp3"              ]
    temp_song1 = files_uuids.get_uuid(temp_songs[0])
    temp_song2 = files_uuids.get_uuid(temp_songs[1])
    temp_song3 = files_uuids.get_uuid(temp_songs[2])

    """if (temp_song1 == None) or (temp_song1 == ""):
        raise NotImplementedError(" [24.0] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[0])
    if (temp_song2 == None) or (temp_song2 == ""):
        raise NotImplementedError(" [24.0] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[1])
    if (temp_song3 == None) or (temp_song3 == ""):
        raise NotImplementedError(" [24.0] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[2])
    """
    # FI NOTA

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Comprovem la funció MusicData.get_song_rank()

    rank_song1 = 0
    rank_song2 = 0
    try:
        rank_song1 = music_data.get_song_rank(temp_song1)
        rank_song2 = music_data.get_song_rank(temp_song2)
    except:
        print("Comment :=>> FAIL: MusicData get_song_rank()")
    else:
        if debug:
            print("             rank_song1 =" , rank_song1 , " rank_song2 =" , rank_song2)
        if (rank_song1 == 5) and (rank_song2 == 10):
            grade += 4
            print("Comment :=>> [24.1] OK: MusicData get_song_rank() result")
        else:
            print("Comment :=>> [24.1] FAIL: MusicData get_song_rank() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Comprovem la funció MusicData.get_next_songs()

    rns = []
    rpes = 0
    try:
        for ns,pes in music_data.get_next_songs(temp_song3):
            rns.append(ns)
            rpes += pes
    except:
        print("Comment :=>> [24.2] FAIL: MusicData get_next_songs()")
    else:
        if debug:
            print("             len(rns) =" , len(rns) , " rpes =" , rpes)
        if (len(rns) == 1) and (rpes == 5):
            grade += 3
            print("Comment :=>> [24.2] OK: MusicData get_next_songs() result")
        else:
            print("Comment :=>> [24.2] FAIL: MusicData get_next_songs() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Comprovem la funció MusicData.get_previous_songs()

    rns = []
    rpes = 0
    try:
        for ns,pes in music_data.get_previous_songs(temp_song2):
            rns.append(ns)
            rpes += pes
    except:
        print("Comment :=>> [24.3] FAIL: MusicData get_previous_songs()")
    else:
        if debug:
            print("             len(rns) =" , len(rns) , " rpes =" , rpes)
        if (len(rns) == 1) and (rpes == 5):
            grade += 3
            print("Comment :=>> [24.3] OK: MusicData get_previous_songs() result")
        else:
            print("Comment :=>> [24.3] FAIL: MusicData get_previous_songs() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Comprovem la funció MusicData.get_song_distance()

    rnodes  = 0
    rvalues = 0
    try:
        for s in music_data:
            nodes,values = music_data.get_song_distance(temp_song2,s)
            rnodes  += nodes
            rvalues += values
    except:
        print("Comment :=>> [24.4] FAIL: MusicData get_song_distance()")
    else:
        if debug:
            print("             rnodes =" , rnodes , " rvalues =" , rvalues)
        if (rnodes == 171) and (rvalues == 735):
            grade += 7
            print("Comment :=>> [24.4] OK: MusicData get_song_distance() result")
        else:
            print("Comment :=>> [24.4] FAIL: MusicData get_song_distance() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 5: Esborrem de la col·lecció una cançó que tingui arestes

    try:
        if debug:
            print("            " , music_data)
        music_data.remove_song(temp_song1)
        files_uuids.remove_uuid(temp_song1)
        if debug:
            print("            " , music_data)
    except:
        print("Comment :=>> [24.5] FAIL: Deleting a song with links")
    else:
        grade += 5
        print("Comment :=>> [24.5] OK: Deleting a song with links")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade
    #########################################################################


# ###########################################################################
# Func2.5: /* Xarxa de llistes de cançons */
def func25(debug: bool = False):
    global grade
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 20
    print ("Comment :=>> Test: Func25 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: Afegir els arxius "mini.m3u" i "micro.m3u" a "list_playlist"
    # Step 2: Incloure en MusicData les llistes a "list_playlist"
    # Step 3: Incloure una segona vegada les llistes a "list_playlist"
    # Step 4: PlayList.read_list(p_llista: list)
   #########################################################################

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Afegir els arxius "mini.m3u" i "micro.m3u" a "list_playlist"

    count = 0
    #for file in ["mini.m3u","micro.m3u"]:
    for file in [m3u_file1_repo,m3u_file3_repo]:
        file = ROOT + os.sep + file
        pl   = PlayList(files_uuids, audio_player) 
        pl.load_file(file)
        if len(pl) > 0:
            list_playlist.append(pl)
            print ("             PlayList(\'File loaded songs: " + str(len(pl)) + "\')")
        count += len(pl)
    print ("             PlayList(\'Total loaded songs: " + str(count) + "\')")
    if count == m3u_file1_len + m3u_file3_len :
        grade += 5
        print("Comment :=>> [25.1] OK: PlayList load_file()")
    else:
        print("Comment :=>> [25.1] FAIL: PlayList load_file()")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Incloure en MusicData les llistes a "list_playlist"

    try:
        for i in list_playlist:
            print("            " , i)
            music_data.read_playlist(i)
        if debug:
            print("            " , music_data)
        grade += 5
        print("Comment :=>> [25.2] OK: MusicData read_playlist()")
    except:
        print("Comment :=>> [25.2] FAIL: MusicData read_playlist()")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 3: Incloure una segona vegada les llistes a "list_playlist"

    try:
        for i in list_playlist:
            print("            " , i)
            music_data.read_playlist(i)
        if debug:
            print("            " , music_data)
        grade += 3
        print("Comment :=>> [25.3] OK: MusicData read_playlist() second time")
    except:
        print("Comment :=>> [25.3] FAIL: MusicData read_playlist() second time")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 4: Comprovem la funció PlayList.read_list()

    temp_songs = []
    for _, s in zip(range(5), files_uuids):
        temp_songs.append(files_uuids.get_uuid(s))

    # NOTA: No podem garantir l'ordre a cada implementació, per tant utilitzem arxius concrets!
    temp_songs = [ "Blues/Arround_the_Cliffs_ID_1202.mp3"            , \
                   "Blues/Checkie_Brown_-_09_-_Mary_Roose_CB_36.mp3" , \
                   "Blues/Do_not_forget_me_ID_1028.mp3"              , \
                   "Blues/Dream_your_Dreams_ID_1195.mp3"             , \
                   "Blues/Ending_at_the_Trainstation_ID_1994.mp3"    ]
    if (temp_songs[0] == None) or (temp_songs[0] == ""):
        raise NotImplementedError(" [25.4] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[0])
    if (temp_songs[1] == None) or (temp_songs[1] == ""):
        raise NotImplementedError(" [25.4] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[1])
    if (temp_songs[2] == None) or (temp_songs[2] == ""):
        raise NotImplementedError(" [25.4] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[2])
    if (temp_songs[3] == None) or (temp_songs[3] == ""):
        raise NotImplementedError(" [25.4] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[3])
    if (temp_songs[4] == None) or (temp_songs[4] == ""):
        raise NotImplementedError(" [25.4] FAIL: Cançó no trobada a la col·lecció!\n " + temp_songs[4])
    temp_songs_new = temp_songs

    temp_songs = []
    for s in temp_songs_new:
        temp_songs.append(files_uuids.get_uuid(s))
    # FI NOTA

    first_playlist = list_playlist[0]
    if debug:
        print("             Playlist before: " , str(first_playlist))
    try:
        first_playlist.read_list(temp_songs)
        music_data.read_playlist(first_playlist)
        if debug:
            print("             Playlist after:  " , str(first_playlist))
            print("            " , music_data)
        grade += 7
        print("Comment :=>> [25.4] OK: PlayList read_list()")
    except:
        print("Comment :=>> [25.4] FAIL: PlayList read_list()")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade
    #########################################################################


# ###########################################################################
# Func2.6: /* Generador recomanacions de cançons */
def func26(debug: bool = False):
    global grade
    global music_collection
    global files_uuids
    global music_data
    global audio_player
    global list_playlist
    global searcher_worker
    local_grade = 36
    print ("Comment :=>> Test: Func26 (debug=" + str(debug) + ")")
    #########################################################################
    # TESTS:
    # Step 1: SearchMetadata.get_similar(uuid: str, max_list: int) -> list
    # Step 2: SearchMetadata.get_topfive() -> list
    #########################################################################

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 1: Fer un càlcul de les cançons similars

    total_similar = 0
    try:

        dIdSimilars ={}
        
        cinc = 0
        quatre = 0
        tres = 0
        dos = 0
        un = 0
        zero = 0
        a = 0

        for i in music_data:

            a+= 1
            print(f"\nIter. {a}:")

            sl = searcher_worker.get_similar(i,5)
            total_similar += len(sl)

            print(total_similar)
            if len(sl) == 5:
                cinc += 1
            elif len(sl) == 4:
                quatre += 1
            elif len(sl) == 3:
                tres += 1
            elif len(sl) == 2:
                dos += 1
            elif len(sl) == 1:
                un += 1
            elif len(sl) == 0:
                zero += 1
            print(f"Distribució: {tuple((zero, un, dos, tres, quatre, cinc))}")

            dIdSimilars[music_data.get_filename(i)]=list(sl)

        # NOTA: En el cas del test a Caronte pels alumnes activar aquest debug
        if (True):
        #if (debug):
            print("Comment :=>> [26.1] DEBUG - Similars:")
            for cs in dIdSimilars.keys():
                print("Comment :=>> [26.1] DEBUG - Similars a :", cs)
                print("Comment :=>>                ", dIdSimilars[cs])
                for sls in dIdSimilars[cs]:
                    print("Comment :=>>                ", music_data.get_filename(sls))

        # Això no forma part del test!
##        LLProv=[]
##        #global dIdSimilars
##        #dIdSimilars ={}
##        #print("Comment :=>> len music data", len(music_data))
##        for i in music_data:
##           
##            sl = searcher_worker.get_similar(i,5)
##            total_similar += len(sl)
##           
##            """
##            llidAux=[]
##            for ele in sl:
##                llidAux.append(get_filename(ele))
##            """
##            #print("Comment :=>> filename i", get_filename(i))
##            #dIdSimilars[get_filename(i)]=llidAux
##        print("Comment :=>> similars", dIdSimilars)
        #############################################################

    except:
        print("Comment :=>> [26.1] FAIL: SearchMetadata get_similar()")
    else:
        if total_similar == 366:
            grade += 14
            print("Comment :=>> [26.1] OK: SearchMetadata get_similar() result")
        else:
            print("Comment :=>> [26.1] FAIL: SearchMetadata get_similar() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # Step 2: Fer el càlcul del top 5 de la col·lecció

    files  = []
    result = [ "Blues/Do_not_forget_me_ID_1028.mp3"                      , \
               "Blues/Dream_your_Dreams_ID_1195.mp3"                     , \
               "Blues/Ending_at_the_Trainstation_ID_1994.mp3"            , \
               "Blues/Lobo_Loco_-_01_-_Allright_in_Lousiana_ID_1234.mp3" , \
               "Blues/Lobo_Loco_-_01_-_Brain_ID_1270.mp3"                ]

    try:
        top5 = searcher_worker.get_topfive()
        for i,c in zip(top5,range(1,6)):
            t = music_data.get_title(i)
            f = music_data.get_filename(i)
            print(" TOP FIVE (" + str(c) + "): UUID:  {" + i + "}")
            print("               Title: '" + t + "'")
            print("               File:  <" + f + ">")
            print("")
            files.append(f)
    except:
        print("Comment :=>> FAIL: SearchMetadata get_topfive()")
    else:
        if (len(top5) == 5) and \
           (files[0] == result[0]) and \
           (files[1] == result[1]) and \
           (files[2] == result[2]) and \
           (files[3] == result[3]) and \
           (files[4] == result[4]) \
           :
            grade += 22
            print("Comment :=>> [26.2] OK: SearchMetadata get_topfive() result")
        else:
            print("Comment :=>> [26.2] FAIL: SearchMetadata get_topfive() result")

    print ("Comment :=>> Partial Grade: ", grade)
    print ("Comment :=>> --------------------------------------------")

    #########################################################################
    # End actions
    print ("Comment :=>> Partial Grade: ", grade)
    print ("             Expected Grade is ["+str(acumg+local_grade)+"] <<< ")
    return local_grade
    #########################################################################


# ###########################################################################
def main():
# START
    global acumg
    acumg = grade

    # Func1 --> class MusicFiles.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>> Test:                    MusicFiles (Func1) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func1(False)

    # Func2 --> class MusicID.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>> Test:                       MusicID (Func2) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func2(False)

    # Func3 --> class MusicData.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>> Test:                     MusicData (Func3) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func3(False)

    # Func4 --> class MusicPlayer.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>> Test:                   MusicPlayer (Func4) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func4(False)

    # Func5 --> class PlayList.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>> Test:                      PlayList (Func5) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func5(False)

    # Func6 --> class SearchMetadata.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>                Test: SearchMetadata (Func6) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func6(False)

    # Func7 --> class PlayList.py
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>                      Test: PlayList (Func7) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func7(False)

    # Func2.1 --> Iteradors i Helpers
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>         Test: Iteradors i Helpers (Func2.1) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func21(False)

    # Func2.2 --> Constructors i Extensions
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>   Test: Constructors i Extensions (Func2.2) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func22(False)

    # Func2.3 --> classes GrafHash i ElementData
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>  Test: class GrafHash/ElementData (Func2.3) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func23(False)

    # NOTE: Check Func2.5 BEFORE Func2.4 !!!
    # Func2.5 --> Xarxa de llistes de cançons
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>> Test: Xarxa de llistes de cançons (Func2.5) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func25(False)

    # NOTE: Check Func2.4 AFTER Func2.5 !!!
    # Func2.4 --> Reimplementació optimitzada classe MusicData
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>   Test: Reimplementació MusicData (Func2.4) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func24(False)

    # Func2.6 --> Generador recomanacions de cançons
    print ("Comment :=>> --------------------------------------------")
    print ("Comment :=>>       Test: Generar recomanacions (Func2.6) ")
    print ("Comment :=>> --------------------------------------------")
    acumg += func26(False)

    print ("Comment :=>> --------------------------------------------")
    if (grade == acumg):
        print ("Comment :=>> OK: Final del Test sense cap error !!!  ")
    print ("Comment :=>> --------------------------------------------")
    print ("Grade :=>> ", grade)
# END

if __name__ == "__main__":
    main()
