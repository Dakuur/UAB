import platform
import sys
import os
import os.path

#############################################################################
#
# Selecció del vostre PATH amb les cançons
#
# Nota: Cal posar el path complet (des del root), i en el cas de Windows
#       precedit per una 'r' per indicar que és un String literal (raw)
#       https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
#
# Exemples:
#ROOT_DIR = r"C:\_UAB\ED\MP3-CORPUS"             # Windows
#ROOT_DIR = "/opt/_uab/ed/mp3-corpus"            # Linux
#ROOT_DIR = "/Users/usuari/_uab/ed/mp3-corpus"   # MacOS
#
#############################################################################

ROOT_DIR = r"C:/Users/david/OneDrive - UAB/Documentos/UAB/2o/estru/projecte/CORPUS-2324-VPL-P2"

#############################################################################
#
# Arxiu per defecte per a fer proves (es pot canviar per qualsevol altre
#
#############################################################################

MP3_DEFAULT = ""
#############################################################################

#
# Mode de reproducció a utilitzar
#
#PLAY_MODE = 0  # Només imprimir per pantalla (mute on)
#PLAY_MODE = 1  # Imprimir per pantalla i reproduïr so
#PLAY_MODE = 2  # Només reproduïr so
#
#############################################################################

PLAY_MODE = 0

#############################################################################
#
# NO REALITZAR **CAP** MODIFICACIÓ A PARTIR D'AQUEST PUNT !!!
#
#############################################################################

_running_platform = platform.system()

if   _running_platform == "Windows" : _rsys = 1
elif _running_platform == "Linux"   : _rsys = 2
elif _running_platform == "Darwin"  : _rsys = 3
else                                : _rsys = 0

if _rsys > 0 :
    print("Running on: " + _running_platform + " ({})\n".format(_rsys))
else:
    print("ERROR: Platform unknown!")
    sys.exit(1)

if not os.path.isdir(ROOT_DIR):
    print("ERROR: ROOT_DIR inexistent!")
    sys.exit(1)

def get_root() -> str:
    """Retorna el local pathname complet de la col·lecció musical."""
    return os.path.realpath(ROOT_DIR)

def get_play_mode() -> int:
    
    return PLAY_MODE

def get_canonical_pathfile(filename: str) -> str: #short
    """Retorna el pathname relatiu amb un format universal."""
    """Exemple: subdir1/subdir2/file01.mp3"""
    file = os.path.normpath(filename)
    file = os.path.relpath(file, ROOT_DIR)
    file = file.replace(os.sep, '/')
    return  file

def get_one_file(mode: int = 0) -> str:
    """Retorna el local pathname complet del darrer MP3 a la col·lecció."""
    """Si el valor és 1 retorna l'arxiu per defecte envers cercar-lo."""
    """Funció d'exemple, no utilizar a la pràctica directament!"""
    file = os.path.realpath(os.path.join(ROOT_DIR, MP3_DEFAULT))
    print(ROOT_DIR , MP3_DEFAULT, file )
    if mode != 1 :
        for root, dirs, files in os.walk(ROOT_DIR):
            for filename in files:
                if filename.lower().endswith(tuple(['.mp3'])):
                    print("found:  " + os.path.join(root, filename))
                    file = os.path.join(root, filename)
    print("select: " + file + "\n")
    return file

contador_global = 0
def incrementar_contador():
    global contador_global
    contador_global += 1
    print(f"Contador: {contador_global}")