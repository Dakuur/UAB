from typing import List, Tuple, Dict
import re
import os

import time

start = time.time()

def llegeix_vocabulari(nom_fitxer: str) -> List[str]:
    #Llegeix d'un fitxer les paraules que formen part del vocabulari i les guarda en una llista
    fitxer = open(nom_fitxer, "r")
    llista = [paraula[:-1] for paraula in fitxer]
    fitxer.close()
    return llista

def crea_bow(nom_fitxer: str, vocabulari: List[str]) -> Dict[str, int]:
    #Obté la representació BoW en forma de diccionari d'un missatge guardat en un fitxer de text
    file = open(nom_fitxer, "r")
    data = file.read()
    clean = re.sub("[^a-zA-Z0-9]", " ", data).split()
    sortida = dict()
    for i in vocabulari:
        sortida[i] = 0
    for paraula in clean:
        if paraula.lower() in vocabulari:
            sortida[paraula.lower()] += 1
    file.close()
    return sortida

def compara_bow(bow1: Dict[str, int], bow2: Dict[str, int]) -> float:
    #Compara la representació BoW de dos missatges, retornant una distància que mesura el grau de similitud entre els dos missatges
    suma1 = sum([min(bow1[x], bow2[x]) for x in bow1.keys()])
    suma2 = sum(bow1.values())
    suma3 = sum(bow2.values())
    distancia = 1 - suma1/min(suma2, suma3)
    return distancia

def crea_conjunt_entrenament(train: str, vocabulari: List[str]) -> list[tuple[str, dict[str, int], bool]]:
    #Llegeix tots els missatges del conjunt d'entrenament i obté la seva representació BoW. Per cada missatge d'entrenament guarda la seva representació i una etiqueta que ens diu si és un missatge d'spam o no. RETURN: training_set: list[tuple[str, dict[str, int], bool]]
    llista_train = os.listdir(train)
    training_set = list()
    for nom_fitxer in llista_train:
        bow = crea_bow(train+nom_fitxer, vocabulari)
        esspam = bool(re.search("spm",nom_fitxer))
        tupla = tuple([nom_fitxer, bow, esspam])
        training_set.append(tupla)
    return training_set

def classifica_document(nom_fitxer: str, training_set: List, vocabulari: List[str], k: int) -> Tuple:
    #Classifica un missatge com a spam o no spam
    bow_analitzar = crea_bow(nom_fitxer, vocabulari)
    resultats = [[compara_bow(bow_analitzar, x[1]), x[2]] for x in training_set]
    #resultats: list[distàncies, bool]
    ordenat = sorted(resultats, key = lambda x : x[0])
    seleccio = ordenat[:k]
    spamcount = 0
    distancies = list()
    for element in seleccio:
        if element[1] == True:
            spamcount += 1
        distancies.append(element[0])
    if spamcount > k/2:
        esspam = True
    else:
        esspam = False
    sortida = tuple([esspam, distancies])
    return sortida

def deteccio_spam(train: str, test: str, fitxer_vocabulari: str, k) -> List:
    #Fa la detecció de spam per tots els missatges que estan en el directori de test
    vocabulary = llegeix_vocabulari(fitxer_vocabulari)
    training_set = crea_conjunt_entrenament(train, vocabulary)
    llista_test = os.listdir(test)
    llista_resultats = list()
    for nom_fitxer in llista_test:
        esspam = classifica_document(test + nom_fitxer, training_set, vocabulary, k)
        res = tuple([nom_fitxer, esspam[0], esspam[1]])
        llista_resultats.append(res)
    return llista_resultats

test_directory = "C:/Users/david/OneDrive - UAB/Documentos/UAB/PA/test/test/"
train_directory = "C:/Users/david/OneDrive - UAB/Documentos/UAB/PA/test/train/"
vocab_directory = "C:/Users/david/OneDrive - UAB/Documentos/UAB/PA/test/vocabulary.txt"

result = deteccio_spam(train_directory, test_directory, vocab_directory, 10)
print(result)

end = time.time()
print(f"{end-start} seconds")