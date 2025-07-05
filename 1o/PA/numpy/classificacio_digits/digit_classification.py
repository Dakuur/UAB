from PIL import Image
import numpy as np
import math
import os
import re

def distancia(llista_1, llista_2):
    suma = 0
    for i in range(0, len(llista_1)):
        suma += (llista_1[i] - llista_2[i])**2
    return math.sqrt(suma)

def black_pixels(filename):
    image = Image.open(filename)
    gray_image = image
    width, height = gray_image.size

    black_pixels_rows = np.zeros(height)
    black_pixels_columns = np.zeros(width)

    mitja = 0
    for y in range(height):
        for x in range(width):
            mitja += gray_image.getpixel((x, y))
    mitja = mitja/(height*width)

    for y in range(height):
        for x in range(width):
            pixel = gray_image.getpixel((x, y))
            if pixel < mitja:
                black_pixels_rows[y] += 1
                black_pixels_columns[x] += 1

    image.close()

    return np.append(black_pixels_rows, black_pixels_columns)

def create_training_set(train):
    train_set = []
    llista_train = os.listdir(train) #fitxers
    
    for nom_fitxer in llista_train:
        if re.search("0.jpg", nom_fitxer):
            num = 0
        elif re.search("2.jpg", nom_fitxer):
            num = 2
        else:
            raise 
        train_set.append((black_pixels(train + nom_fitxer), num))
    return train_set

def classificacio_digits(train: str, test: str, k: int):
    training_set = create_training_set(train)
    llista_test = os.listdir(test)
    llista_retornar = []

    for nom_fitxer in llista_test:
        array = black_pixels(test + nom_fitxer)
        resultats = [(distancia(x[0], array), x[1]) for x in training_set]
        ordenat = sorted(resultats, key = lambda x : x[0])
        seleccio = ordenat[:k]
        zeros = 0
        twos = 0
        for element in seleccio:
            if element[1] == 0:
                zeros += 1
            elif element[1] == 2:
                twos += 1
        if zeros > twos:
            resultat = "0"
        else:
            resultat = "2"
        tupla = (nom_fitxer, resultat, array.astype(int).tolist())
        llista_retornar.append(tupla)
    return llista_retornar