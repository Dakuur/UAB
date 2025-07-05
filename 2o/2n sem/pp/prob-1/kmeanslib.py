NOM_1 = "Adrià Muro Gómez"
NIU_1 = 1665191
NOM_2 = "David Morillo Massagué"
NIU_2 = 1666540

from ctypes import *

class Cluster(Structure):
    _fields_ = [
        ("num_puntos", c_uint32),
        ("r", c_uint8),
        ("g", c_uint8),
        ("b", c_uint8),
        ("media_r", c_uint32),
        ("media_g", c_uint32),
        ("media_b", c_uint32)
        ]

class RGB(Structure):
    _fields_ = [
        ("b", c_uint8),
        ("g", c_uint8),
        ("r", c_uint8)
        ]

class Image(Structure):
    _fields_ = [
        ("length", c_uint32),
        ("width", c_uint32),
        ("height", c_uint32),
        ("header", c_uint8 * 54),
        ("pixels", POINTER(RGB)),
        ("fp", c_void_p)
        ]



class Kmeans:

    def __init__(self, k = 10):
        self.k = k
        self.lib = CDLL('./kmeanslib.so')
        # L'arxiu kmeanslib.so l'hem obtingut executant la comanda seguent:
            # gcc -fPIC -shared -o kmeanslib.so kmeanslib.c
        # -fPIC -> position-independent code (dll compatible)
        # -shared -> shared library
        self.cluster = Cluster()
        self.imatge = Image()

    def read_file(self):
        self.lib.read_file.argtypes = [c_char_p, POINTER(Image)]
        self.lib.read_file.restype = c_int
        rf_res = self.lib.read_file(b'imagen.bmp', self.imatge)
        print(f"read_file() returned {rf_res} (0 means no errors)\n")

    def kmeans(self):
        self.lib.kmeans.argtypes = [c_uint8, POINTER(Cluster), c_uint32, POINTER(RGB)]
        self.lib.kmeans(self.k, self.cluster,self.imatge.length ,self.imatge.pixels)
        # No comprovem sortida ja que retorna void

    def getChecksum(self):
        self.lib.getChecksum.argtypes = [POINTER(Cluster), c_uint8]
        self.lib.getChecksum.restype = c_uint32
        cs_res = self.lib.getChecksum(self.cluster, self.k)
        print(f"\nChecksum result: {cs_res}\n")

    def write_file(self):
        self.lib.write_file.argtypes = [c_char_p, POINTER(Image), POINTER(Cluster), c_uint8]
        self.lib.write_file.restype = c_int
        wf_res = self.lib.write_file(b'imagen.bmp', self.imatge, self.cluster, self.k)
        print(f"write_file() returned {wf_res} (0 means no errors)\n")


if __name__ == "__main__":

    print("Init. class Kmeans:\n")

    k = 4

    objecte = Kmeans(k = k) # Canviar el valor de k al nombre de clústersdesitjat

    objecte.read_file()

    objecte.kmeans()

    objecte.getChecksum()

    objecte.write_file()

    print("End of the execution")