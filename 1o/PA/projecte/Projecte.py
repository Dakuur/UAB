# Nom: Adrià Muro Gómez
# NIU: 1665191
# Nom: David Morillo Massagué
# NIU: 1666540

from typing import List, Dict
from dataclasses import dataclass, field
import numpy as np
import csv
from time import time
from math import sqrt
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import pickle

@dataclass
class Vote:
    # Correspon a cada un dels vots que realitza un usuari que inclou
    # el ID de usuari i item, i el seu valor

    _user_id: int
    _item_id: int
    _rating: float

    @property
    def item_id(self):
        return self._item_id

    @property
    def rating(self):
        return self._rating

@dataclass
class User:
    # Correspon a cada un dels usuaris del sistema. Els atributs són el seu ID i
    # una llista de vots que ha realitzat de la classe Vote

    _id: str
    _votes: List[Vote]
       
    @property
    def id(self):
        return self._id

    @property
    def votes(self):
        return self._votes

@dataclass
class Item:
    # Cadascun dels ítems (movie o book) del sistema. Guardem com atributs el ID,
    # títol, la mitjana de valoracions, el número de vots que ha rebut, i la llista de vots que ha rebut

    _id: str
    _title: str
    _avg_item: float
    _n_vots: int
    _votes: List[Vote]

    @property
    def title(self):
        return self._title

    @property
    def avg_item(self):
        return self._avg_item
   
    @avg_item.setter
    def avg_item(self, value):
        self._avg_item = value
   
    @property
    def n_vots(self):
        return self._n_vots
   
    @n_vots.setter
    def n_vots(self, value):
        self._n_vots = value

    def visualitza():
        pass

@dataclass
class Movie(Item):
    # Tipus específic de Ítem corresponent a una película
    # Guardem l'atribut Genre, per a aplicar-ho en el sistema basat en contingut

    _genres: str

    @property
    def genres(self):
        return self._genres
    
    def visualitza(self):
        # Mostra per pantalla la informació de la película

        print("movieid:",self._id)
        print("title:",self._title)
        print("genres:",self._genres)

@dataclass
class Book(Item):
    # Tipus específic de Ítem corresponent a un llibre
    # Guardem els atributs: Autor i Any de publicació

    _author: str
    _pub_year: int

    @property
    def author(self):
        return self._author
    
    @property
    def pub_year(self):
        return self._pub_year
    
    def visualitza(self):
        # Mostra per pantalla la informació del llibre

        print("bookid:",self._id)
        print("title:",self._title)
        print("author:",self._author)
        print("year:",self._pub_year)

@dataclass
class Dataset:
    # Correspon a un conjunt de dades que inclouen usuaris, ítems i valoracions
    # Guardem com atributs:
    #   Un diccionari amb keys corresponents a user IDs i amb valor de la classe User
    #   Un diccionari amb keys corresponents a item IDs i amb valor de la classe Item
    #   Un valor float corresponent amb la mitjana global de tots els Items
    #   Una matriu, en la que les files corresponen als Users, i les columnes als Items
    #       (en el mateix ordre que es diccionaris)
    #   Una llista de puntuacions que proporciona el sistema de recomanaciño escollit,
    #       en forma de llista de tuples [itemID, puntuació calculada]
    #   Un número mínim de vots, i un altre d'usuaris que es fan servir en els sistemes de recomanació
    #   Un diccionari de similitud que s'utilitza en el sistema colaboratiu
    #   Un valor booleà que correspon al mode en el que s'utilitza el Dataset (True = mode avaluació)

    _users: Dict[str, User] = field(default_factory=dict) #key = user id
    _items: Dict[str, Item] = field(default_factory=dict) #key = item id
    _avg_global: float = 0

    _matriu: np.matrix = 0

    _scores: list = field(default_factory=list)
    _min_vots: int = 10

    _k_users: int = 0
    _dic_similitud: Dict[str, float] = field(default_factory=dict) #key = user id
    
    _avaluacio: bool = False

    @property
    def users(self):
        # Getter per obtenir la llista d'usuaris del dataset
        return self._users

    def inicialitza_items(self):
        # Calcula els valors com el número de vots o la mitjana de puntuació de cada ítem

        keyslist = list(self._items.keys())
        num_items = len(keyslist)

        nonzero_counts = np.sum(self._matriu != 0, axis=0)

        for i in range(num_items):
            item_id = keyslist[i]
            self._items[item_id].n_vots = nonzero_counts[i]

            column = self._matriu[:, i]
            nonzero_values = column[column != 0]
            avg = np.mean(nonzero_values)
            self._items[item_id].avg_item = avg

    def score(self, item_id: str):
        # Donat un numero de item (columna de la matriu)
        # Ens dona la seva score pel sistema de recomanació simple, si es compleixen els requisits
        # (num vots > min vots)

        num_vots = self._items[item_id].n_vots
        min_vots = self._min_vots
        if num_vots < min_vots:
            return 0
        else:
            scr = (num_vots/(num_vots+min_vots))*self._items[item_id].avg_item+(min_vots/(num_vots+min_vots))*self._avg_global
            return scr

    def index_usuari(self, user_id: str):
        # Retorna l'índex de l'ID de l'usuari introduir per a saber la fila de la matriu

        for i, value in enumerate(self._users.values()):
            if value.id == user_id:
                return i

    def recomanacio_simple(self, user_id: str):
        # Sistema de recomanació basat en la puntuació que han fet els altres usuaris
        # Fa servir la funció score() prèviament comentada per a assignar una puntuació a cada ítem
        # Assigna a self._scores als millors ítems segons el sistema (només pels ítems que l'usuari no hagi puntuat)
       
        self._scores = []

        fila = self._matriu[self.index_usuari(user_id), :]

        keylist = list(self._items.keys())
        
        posicions_visitar = []
        
        for i in range(0, len(fila)):
            
            if fila[i] == 0 and self._avaluacio == False:
                posicions_visitar.append(i)
                
            elif self._avaluacio == True:
                posicions_visitar.append(i)
                
        for i in posicions_visitar: #per totes les columnes de la matriu
            id = keylist[i]
            puntuacio = self.score(id)
            self._scores.append((id, puntuacio))

    def recomanacio_colaboratiu(self, user_id: str, k: int):
        # Sistema de recomanació que assigna un score a cada ítem no vist de l'usuari a recomanar.
        # Score és una puntuació obtinguda a partir de les valoracions de k usuaris que tenen activitat/gustos 
        # semblants a l'usuari al qual s'ha de recomanar
        # Assigna a self._scores als millors ítems segons el sistema (només pels ítems que l'usuari no hagi puntuat)
        
        # CÀLCUL DE DISTÀNCIES
        self._k_users = k
        
        user_votes = self._matriu[self.index_usuari(user_id), :]
        user_votes = list(user_votes)
        
        for i in range (0, len(self._users)):
            if i != self.index_usuari(user_id):
                comparative_user_votes = self._matriu[i, :]
                comparative_user_votes = list(comparative_user_votes)
               
                sumatori = 0
               
                valors_u = []
                valors_u2 = []
               
                for x, y in zip(user_votes, comparative_user_votes):
                   
                    if x != 0.0 and y != 0.0:
                        sumatori += x*y
                        valors_u.append(x)
                        valors_u2.append(y)
                       
                if sumatori == 0:
                    self._dic_similitud[i+1] = 0
                   
                else:
                    distance = (sumatori)/((sqrt(sum([x**2 for x in valors_u])))*(sqrt(sum([x**2 for x in valors_u2]))))
                    self._dic_similitud[i+1] = distance
             
        sorted_dic = dict(sorted(self._dic_similitud.items(), key=lambda item:item[1], reverse=True))
     
        k_elements = dict(itertools.islice(sorted_dic.items(), int(self._k_users)))
       
        #FÓRMULA ÍTEMS
       
        avg_users = dict()
        
        for i in range(0, len(self._users.keys())):

            id = list(self._users.keys())[i]

            fila = self._matriu[i]
            nonzero_values = fila[np.nonzero(fila)]
            avg = np.mean(nonzero_values)

            avg_users[id] = avg

        fila = self._matriu[self.index_usuari(user_id), :]

        keylist = list(self._items.keys())
        i_no_vist = []
        
        for i in range(0, len(fila)):
            
            if fila[i] == 0 and self._avaluacio == False:
                i_no_vist.append(i)
                
            elif self._avaluacio == True:
                i_no_vist.append(i)
                
        self._scores = []

        for item_i in i_no_vist:

            num = 0
            den = 0
            
            for key, dist in k_elements.items(): #Dic[user_ud] -> float
                score = self._matriu[key-1, item_i]
                
                if score != 0.0:
                    mitja = avg_users[key]
                    num += dist * (score - mitja)
                    den += dist
            if den != 0.0 or den != 0:
                resultat = avg_users[user_id] + (num / den)
                
                self._scores.append((keylist[item_i], resultat))
 
    def recomanacio_contingut(self, user_id: str):
        
        # Sistema de recomanació que assigna un score a cada ítem no vist de l'usuari a recomanar.
        # Score és una puntuació obtinguda a partir de les valoracions dels ítems els quals tenen característiques semblants a l'usuari a recomanar
        # Les característiques depenen del ítem, en el cas de les pel·lícules s'ha agafat com a paràmetre els gèneres que inclouen i en els llibres els autors
        # Assigna a self._scores als millors ítems segons el sistema (només pels ítems que l'usuari no hagi puntuat)
        
        #CARACTERISTIQUES
        self._scores = []
        characteristics_list = []
        
        try:
            for movie in self._items.values():
                characteristics_list.append(str(movie._genres))
                
        except:
            for llibre in self._items.values():
                characteristics_list.append(str(llibre._author))
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(characteristics_list).toarray()
        
        #PROFILE
      
        profile_list = []
        num = 0
        user_votes = self._matriu[self.index_usuari(user_id), :]
        sumatori_vots = sum(user_votes)
        
        for i in range(len(user_votes)):
            row = tfidf_matrix[i, :]
            num_value = []
            
            for value in row:
                num_value.append(value*user_votes[i])

            if type(num) == int:
                num = num_value
                
            else:
                num = np.add(num, num_value)
                
        profile_list = np.divide(num,sumatori_vots)
    
        #DISTANCIA
        
        punctuation_list = []
        
        for i in range (len(tfidf_matrix)):
            x = 0
            y = 0
            num = 0
            for j in range (len(tfidf_matrix[0])):
                
                  
                num += tfidf_matrix[i][j] * profile_list[j]
                y += tfidf_matrix[i][j]**2
                x += profile_list[j]**2
                
            
            
            distance = (num/(sqrt(x)*sqrt(y)))
            punctuation_list.append(distance*5)
        return_list = []
        for id, distance in zip(self._items, punctuation_list):
            
            return_list.append((id,distance))
            
        
        self._scores = return_list
          
    def genera_matriu(self):
        # Aquesta funció genera una matriu de mida m x n:
        #   M: Usuaris
        #   N: Ítems
        # Els valors d'aquesta venen donats per la valoració del usuari de la fila M a l'Ítem N
        # Si un usuari no ha puntuat un ítem, se li assigna valor 0
        # Primer itera per cada usuari i després per cadda Ítem puntuat (dins de la llista user.votes)

        user_keys = list(self._users.keys())
        item_keys = list(self._items.keys())
        num_users = len(user_keys)
        num_items = len(item_keys)
        
        self._matriu = np.zeros((num_users, num_items))
        
        for u in range(num_users):
            user_id = user_keys[u]
            user_votes = self._users[user_id].votes
            
            for vote in user_votes:
                item_id = vote.item_id
                if item_id in item_keys:
                    i = item_keys.index(item_id)
                    self._matriu[u, i] = vote.rating
    
    def avalua(self, sistema: str, user: str, k: int, n: int, llindar: float):
        # Donat un sistema d'avaluació i altres paràmetres pel seu funcionament, calcula els valors de:
        #   MAE (mean absolute error)
        #   Precisió
        #   Recall
        # Per tal d'avaluar l'eficiència del sistema programat
        # El sistema imprimex les millors prediccions del sistema, a part dels resultats calculats anteriorment

        self._avaluacio = True

        if sistema in ["simple", "1"]:
            self.recomanacio_simple(user)
        elif sistema in ["colab", "2"]:
            self.recomanacio_colaboratiu(user, k)
        elif sistema in ["contingut", "3"]:
            self.recomanacio_contingut(user)
       
        prediccions_sistema = []
        
        for value in self._scores:
            prediccions_sistema.append(value[1])
            
        valoracions_usuari =  self._matriu[self.index_usuari(user), :]  
            
        mae = self.calcula_MAE(prediccions_sistema, valoracions_usuari)
        precision = self.calcula_precision(prediccions_sistema, valoracions_usuari, n, llindar) #valors a canviar
        recall = self.calcula_recall(prediccions_sistema, valoracions_usuari, n, llindar) #valors a canviar
        
        self._scores.sort(key=lambda x: x[1], reverse = True)
        self._scores = [(id, round(score, 3)) for id, score in self._scores]
        
        res = self._scores[0:n]
        id_valoracio = []
        
        keylist = list(self._items.keys())
        
        for i in range (len(valoracions_usuari)):

            id = keylist[i]
        
            id_valoracio.append((id, valoracions_usuari[i]))
            
        print("Sistema a avaluar:", sistema.capitalize())

        print("")
        print("Millors N prediccions del sistema:")
        for value in res:
            print("ID:",value[0]," Score:",value[1])
            
        print("")
        print("Valoracions de l'usuari que superen el llindar: ")
        for valoracio in id_valoracio:
            if valoracio[1] >= llindar:
                print("ID:",valoracio[0]," Puntuacio:",valoracio[1])
        print("")
        print("Mae:", mae)
        print("Precision:", precision)
        print("Recall:", recall)
        print("")
        
    def calcula_MAE(self, prediccions_sistema: list, valoracions_usuari: list):
        # Calcula el Error Absolut Mitjà d'un sistema de recomanació amb els resultats ja donats
        
        suma = 0
        
        for valor_p, valor_u in zip(prediccions_sistema, valoracions_usuari):
            
            if float(valor_u) != float(0):
                suma += abs(valor_p - valor_u)
        
        n = np.count_nonzero(valoracions_usuari)
        return suma/n
    
    def calcula_precision(self, prediccions_sistema: list, valoracions_usuari: list, k: int, llindar: float):
        # Calcula la Precisió d'un sistema de recomanació amb els resultats ja donats

        conjunt = [(prediccions_sistema[i], valoracions_usuari[i]) for i in range(len(prediccions_sistema)) if valoracions_usuari[i] != 0]
        conjunt.sort(key=lambda x: x[0], reverse = True)
        
        k_millors = conjunt[:k]

        counter = 0

        for value in k_millors:
            if value[1] >= llindar:
                counter += 1

        return counter/k
        
    def calcula_recall(self, prediccions_sistema: list, valoracions_usuari: list, k: int, llindar: float):
        # Calcula el Recall d'un sistema de recomanació amb els resultats ja donats

        conjunt = [(prediccions_sistema[i], valoracions_usuari[i]) for i in range(len(prediccions_sistema)) if valoracions_usuari[i] != 0]
        conjunt.sort(key=lambda x: x[0], reverse = True)
        k_millors = conjunt[:k]

        counter = 0

        for value in k_millors:
            if value[1] >= llindar:
                counter += 1

        counter_2 = 0
        for rating in valoracions_usuari:
            if rating >= llindar:
                counter_2 += 1
                
        return counter/counter_2
        
    def inicialitza(self):
        # Funció que inicialitza la matriu i els ítems del diccionari del Dataset, a més del valor mitjà global
        # Crida a la funció genera_matriu() i inicialitza_items() comentades prèviament

        #self._min_vots = int(input("Min. vots: "))
        self.genera_matriu()
        self.inicialitza_items()
        self._avg_global = np.mean(self._matriu[np.where(self._matriu != 0)])

    def visualitza(self):
        # Funció que permet visualitzar els resultats dels càlculs d'un sistema de recomanació
        # Fa el call de la funció visualitza(), per a poder visualitzar específicament els atributs d'una classe d'ítem
        
        self._scores.sort(key=lambda x: x[1], reverse = True)
        self._scores = [(id, round(score, 3)) for id, score in self._scores]
        
        res = self._scores[0:5]
        
        print("")

        for value in res:
            item = self._items[value[0]]
            item.visualitza()
            print("score:", value[1])
            print("")

    def add_user(self, id: str, vote: float):
        # Funció que permet afegir un usuari al diccionari de Usuaris, comprovant si està prèviament
        # Si jà està al sistema, afegeix el vot que s'inclou en la crida de la funció (vote)

        if id in self._users.keys():
            self._users[id].votes.append(vote)
        else:
            self._users[id] = User(id, [vote])

    def importa_items(self, filename: str):
        # Creat per respectar els patrons GRASP
        pass

    def importa_ratings(self, filename: str):
        # Creat per respectar els patrons GRASP
        pass

    def importa(self, movies: str, ratings: str):
        # Funció principal que importa les dades d'un Dataset
        # Fent ús del polimorfisme, crida a la funció del tipus de Dataset específic (películes o llibres)

        self.importa_items(movies)
        self.importa_ratings(ratings)

@dataclass
class Dataset_Movies(Dataset):
    # Classe de Dataset específic de películes
   
    def importa_items(self, filename: str):
        # Importa des d'un arxiu donat com argument les películes en forma de Movie(), en el Dataset

        with open(filename, newline='', encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # saltem la primera línia
            for row in csvreader:
                id = int(row[0])
                title = str(row[1])
                genres = str(row[2])
                movie = Movie(id, title, 0, 0, [], genres)
                self._items[id] = movie #asumim no repetits

    def importa_ratings(self, filename: str):
        # Importa des d'un arxiu donat com argument els vots en forma de Vote() de película, en el Dataset
        # A més, afegeix l'usuari amb la funció add_user()

        with open(filename, newline='', encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # saltem la primera línia
            for row in csvreader:
                id = int(row[0])

                movieid = int(row[1])
                rating = float(row[2])

                vote = Vote(id, movieid, rating)
               
                self.add_user(id, vote)

@dataclass
class Dataset_Books(Dataset):
    # Classe de Dataset específic de llibres
       
    def importa_items(self, filename: str):
        # Importa des d'un arxiu donat com argument els llibres en forma de Book(), en el Dataset

        with open(filename, newline='', encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # saltem la primera línia
            for row in csvreader:
                id = int(row[0])
                title = str(row[9])
                author = str(row[7])
                year = str(row[8])[:-2]
                book = Book(id, title, 0, 0, 0, author, year)
                self._items[id] = book #asumim no repetits

    def importa_ratings(self, filename: str):
        # Importa des d'un arxiu donat com argument els vots en forma de Vote() de llibre, en el Dataset
        # A més, afegeix l'usuari amb la funció add_user()

        with open(filename, newline='', encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # saltem la primera línia
            for row in csvreader:
                id = int(row[1])
                bookid = int(row[0])
                rating = float(row[2])

                vote = Vote(id, bookid, rating)
               
                self.add_user(id, vote)

@dataclass
class Controller:
    # Classe que permet la comunicació entre l'usuari (terminal) i el Dataset (movies o books)

    _dataset: Dataset = 0

    def iniciar_dataset(self, dataset: str):
        # Donat el tipus de Dataset que volem crear, importa els arxius
        # i fa les càlculas prèvis requerits per els sistemes de recomanació
        # Si existeix un arxiu .dat, no farà els càlculs mencionats i importarà la variable amb pickle  
        
        #CANVIAR DIRECTORIS EN CAS D'ERROR
        if dataset in ["movies", "m", "1"]:
            pickle_file = "ds_movies.dat"
            items = "movies/movies.csv"
            ratings = "movies/ratings.csv"
            ds = Dataset_Movies()

        elif dataset in ["books", "b", "2"]:
            pickle_file = "ds_books.dat"
            items = "books/books.csv"
            ratings = "books/ratings.csv"
            ds = Dataset_Books()

        else:
            return False
        
        try:
            with open(pickle_file, 'rb') as fitxer:
                ds = pickle.load(fitxer)

        except:
            s = time()
            ds.importa(items, ratings)
            print("Dataset inicialitzat correctament")
            f = time()
            print("Importar:",round(f-s, 3),"segons")

            s = time()
            ds.inicialitza()
            f = time()
            print("Inicialitzar:",round(f-s, 3), "segons")

            with open(pickle_file, 'wb') as fitxer:
                pickle.dump(ds, fitxer)

        self._dataset = ds

    def get_request(self):
        # Funció que facilita que un usuari introdueixi per teclat un
        # tipus de activitat que vulgui realitzar en el sistema
        # Opcions: Recomanar/Avaluar/Sortir
        # La funció assegura que la opció introduida sigui correcte

        request_list = ["recomanar", "avaluar", "sortir", "r", "a", "s", "1", "2", "3", "-1", "sortir"]
        entrada = str(input("Escull una opció: (Recomanar/Avaluar/Sortir): ")).lower()
        while entrada not in request_list:
            print("Opció no vàlida")
            entrada = str(input("Escull una opció: (Recomanar/Avaluar/Sortir): ")).lower()
        return entrada

    def get_dataset_type(self):
        # Funció que facilita que un usuari introdueixi per teclat un
        # tipus de dataset que vulgui fer servir en el sistema
        # Opcions: Movies/Books
        # La funció assegura que la opció introduida sigui correcte

        dataset_list = ["movies", "books", "m", "b", "1", "2", "-1", "sortir"]
        entrada = str(input("Quin dataset vols utilitzar? (Movies/Books): " )).lower()
        while entrada not in dataset_list:
            print("Dataset no trobat")
            entrada = str(input("Quin dataset vols utilitzar? (Movies/Books): " )).lower()
        return entrada

    def get_system(self):
        # Funció que facilita que un usuari introdueixi per teclat un
        # tipus de sistema de recomanació que vulgui fer servir en el sistema
        # Opcions: Simple/Colab/Contingut
        # La funció assegura que la opció introduida sigui correcte

        system_list = ["simple","colab","contingut", "1", "2", "3", "-1", "sortir"]
        entrada = str(input("Quin sistema de recomanació vols utilitzar? (Simple/Colab/Contingut): ")).lower()
        while entrada not in system_list:
            print("Sistema no trobat")
            entrada = str(input("Quin sistema de recomanació vols utilitzar? (Simple/Colab/Contingut): ")).lower()
        return entrada

    def get_user(self, llista_valids: list):
        # Funció que facilita que un usuari introdueixi per teclat un usuari a recomanar en el sistema
        # La funció assegura que l'usuari es trobi en el Dataset

        entrada = int(input("A quin usuari vols recomanar? "))
        while entrada not in llista_valids:
            print("Usuari no trobat")
            entrada = int(input("A quin usuari vols recomanar? "))
        return entrada

    def get_user_avaluar(self, llista_valids: list):
        # Funció que facilita que un usuari introdueixi per teclat un usuari per a avaluar el sistema
        # La funció assegura que l'usuari es trobi en el Dataset

        entrada = int(input("Amb quin usuari vols avaluar? "))
        while entrada not in llista_valids:
            print("Usuari no trobat")
            entrada = int(input("A quin usuari vols recomanar? "))
        return entrada

    def main(self):
        # Funció principal del programa que permet comunicar-se via la terminal
        # Consisteix de 4 nivells de menú:
        #   Activitat (Recomanar/Avaluar/Sortir)
        #       Tipus de dataset (Movies/Books)
        #           Tipus de sistema de recomanació (Simple/Colab/Contingut)
        #               ID de usuari
        #
        # Si en un nivel s'introdueix com a entrada -1, torna al nivel anterior, per a realitzar
        # els canvis que l'usuari vulgui
        # En els sitemes que ho requereixi, tembé permet introduir peràmetres per teclat

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        request = self.get_request()
        while request not in ["3", "s", "-1", "sortir"]:
            
            dataset_type = self.get_dataset_type()
            while dataset_type not in ["-1", "sortir"]:

                self.iniciar_dataset(dataset_type)
                users = self._dataset.users
                users_ids = [users[i].id for i in users]
                users_ids.append(-1)

                if request in ["recomanar", "r", "1"]:
                    
                    system = self.get_system()
                    while system not in ["-1", "sortir"]:

                        user = self.get_user(users_ids)
                        while user != -1:
                            if system in ["simple", "1"]:
                                self._dataset.recomanacio_simple(user)
                            elif system in ["colab", "2"]:
                                k_usuaris = int(input("K usuaris? "))
                                self._dataset.recomanacio_colaboratiu(user, k_usuaris)
                            elif system in ["contingut", "3"]:
                                self._dataset.recomanacio_contingut(user)
                            self._dataset.visualitza()

                            user = self.get_user(users_ids)
                        system = self.get_system()

                elif request in ["avaluar", "a", "2"]:

                    system = self.get_system()
                    while system not in ["-1", "sortir"]:

                        try:
                            llindar = int(input("Quin llindar vols utilitzar: "))
                            n = int(input("Quina N vols utilitzar: "))
                        except:
                            print("Valor incorrecte")
                            
                        user = self.get_user_avaluar(users_ids)
                        while user != -1:

                            self._dataset.avalua(system, user, 5, n, llindar) #canviar k al valor que es vulgui

                            user = self.get_user_avaluar(users_ids)

                        system = self.get_system()

                dataset_type = self.get_dataset_type()
            request = self.get_request()
        print("Sortint del sistema...")

try:
    c = Controller()
    c.main()
except:
    print("Error en l'execució")