def file2list(file_name, long):
    llista = []
    try:
        with open(file_name, "r") as f:
            for word in f:
                word = word[:-1]
                if len(word)==long:
                    llista.append(word)
    except FileNotFoundError:
        raise FileNotFoundError("Error: Fitxer no trobat")
     
    return llista
def file2list(file_name, long):
    try:
        with open(file_name, "r") as f:
            llista = [ word[:-1] for word in f if len(word[:-1]) == long ]
    except FileNotFoundError:
        raise FileNotFoundError("Error: Fitxer no trobat")
     
    return llista

def histogram(llista):
    dic = {}
    for word in llista:
        for c in word:
            if c in dic:
                dic[c] += 1
            else:
                dic[c] = 1
    return dic

def points(word, histo):
    suma = 0
    for c in word:
        suma += histo[c]
    return suma

def points(word, histo):
    return sum(map(lambda x: histo[x], word))

def word_most_discriminative(llista, histograma):
    max_pes = 0
    paraula = ""
    for word in llista:
        suma = points(word, histograma)
        if suma > max_pes:
            max_pes = suma
            paraula = word
    return paraula
def word_most_discriminative(llista, histograma):
    puntuacions = list(map(lambda x:points(x, histograma),llista))
    pos_max = puntuacions.index(max(puntuacions))
    return llista[pos_max]

def mask2clues(word, reply, letters_ok, letters_ko, in_word, not_in_word):  
    for i in range(len(word)):
        if reply[i] == 'O':
            letters_ok[i] = word[i]
        elif reply[i] == '_':
            in_word.add(word[i])
            if word[i] not in letters_ko[i]:
                letters_ko[i].append(word[i])
        elif reply[i] == 'X':
            not_in_word.add(word[i])
def mask2clues(word, reply, letters_ok, letters_ko, in_word, not_in_word):  
    for i,r in enumerate(reply):
        if r == 'O':
            letters_ok[i] = word[i]
        elif r == '_':
            in_word.add(word[i])
            if word[i] not in letters_ko[i]:
                letters_ko[i].append(word[i])
        elif r == 'X':
            not_in_word.add(word[i])
def mask2clues(word, reply, letters_ok, letters_ko, in_word, not_in_word):  
    for i,(w,r) in enumerate(zip(word,reply)):
        if r == 'O':
            letters_ok[i] = w
        elif r == '_':
            in_word.add(w)
            if w not in letters_ko[i]:
                letters_ko[i].append(w)
        elif r == 'X':
            not_in_word.add(w)

def candidate_word(word, letters_ok, letters_ko, in_word, not_in_word):
    i = 0
    possible = True
    while i < len(word) and possible:
        if letters_ok[i] != '' and word[i] != letters_ok[i]:
            possible = False
        elif word[i] in letters_ko[i] or word[i] in not_in_word:
            possible = False
        else:
            i += 1
     
    if possible:
        possible = in_word.issubset(set(word))
         
    return possible

NOM_FITXER = "words.txt"
MAX_INTENTS = 6
letters_ok = dict()
letters_ko = dict()
in_word = set()
not_in_word = set()
longitud = int(input("Amb quina longitud de paraules vols jugar? "))
PATRO_CORRECTE = "O" * longitud
for i in range(longitud):
    letters_ok[i] = ''
    letters_ko[i] = []
try:
    ll_totes_paraules = file2list(NOM_FITXER, longitud)
except FileNotFoundError as missatge:
    print(missatge)
else:     
    histo_lletres = histogram(ll_totes_paraules)
     
    # Busquem la paraula mes discriminativa
    paraula = word_most_discriminative(ll_totes_paraules, histo_lletres)
    print(paraula)
    resposta = input("Introdueix pistes ('X' - no present, 'O' - ben col∙locada, '_' - mal col∙locada): ")
    intents = 1
     
    error = False
    while resposta != PATRO_CORRECTE and intents < MAX_INTENTS and not error:
        # Analitzem la informació d'encerts, aproximacions i lletres no presents  
        mask2clues(paraula, resposta, letters_ok, letters_ko, in_word, not_in_word)
        # Eliminem totes les paraules que no poden ser candidates
        ll_totes_paraules = [word for word in ll_totes_paraules if candidate_word(word,
letters_ok, letters_ko, in_word, not_in_word)]
         
        if ll_totes_paraules != []:
            # Busquem la paraula mes discriminativa
            paraula = word_most_discriminative(ll_totes_paraules, histo_lletres)
            print(paraula)
            resposta = input("Introdueix pistes ('X' - no present, 'O' - ben col∙locada, '_' -  mal col∙locada): ")
            intents += 1
        else:
            print("Error: no hi ha paraules candidates")
            error = True
     
    if resposta != PATRO_CORRECTE:
        print("No s'ha aconseguit trobar la paraula")
    else:
        print("S'ha trobat la paraula en",intents,"intents")