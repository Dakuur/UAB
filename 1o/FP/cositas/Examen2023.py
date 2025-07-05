def file2list(nomfitxer, longword):
    try:
        lines = open(nomfitxer, "r")
    except:
        raise FileNotFoundError("Error: File not found")
    else:
        wordlist = []
        for line in lines:
            if len(line[:-1]) == longword:
                wordlist.append(line[:-1].upper())
        lines.close()
        return wordlist

def frequencies(wordlist):
    frequencies = dict()
    for word in wordlist:
        for letter in word:
            if letter not in frequencies:
                frequencies[letter] = 1
            else:
                frequencies[letter] += 1
    return frequencies

def bestword(wordlist, dictionary):
    bestword = wordlist[0]
    bestpoints = 0
    for word in wordlist:
        points = 0
        for letter in word:
            if letter in dictionary:
                points += dictionary[letter]
        if points > bestpoints:
            bestpoints = points
            bestword = word
    return bestword

def check(guess, reply, letters_ok, letters_ko, in_word, not_in_word):
    for i in range(0, len(guess)):
        if reply[i] == "O":
            letters_ok[i] = guess[i]
        elif reply[i] == "_":
            letters_ko[i].append(guess[i])
            in_word.add(guess[i])
        else:
            not_in_word.add(guess[i])
    return letters_ok, letters_ko, in_word, not_in_word

def is_candidate(word, dictionary, letters_ok, letters_ko, not_in_word):
    for i in range(0, len(word)):
        if (letters_ok[i] in dictionary) and (letters_ok[i] != word[i]):
            return False
        elif word[i] in letters_ko[i]:
            return False
        elif word[i] in not_in_word:
            return False
    return True

def found(letters_ok, dictionary):
    for i in letters_ok:
        if i not in dictionary:
            return False
    return True

######################## MAIN ########################

'''
filename = str(input("File name: "))
length = int(input("Word length: "))
'''

filename = "words.txt"
length = 5

wordlist = file2list(filename, length)
dictionary = frequencies(wordlist)

letters_ok = dict()
letters_ko = dict()

for i in range(0, length):
    letters_ok[i] = str()
    letters_ko[i] = list()
in_word = not_in_word = set()

tries = 6
correct = False
while (correct == False) and (tries > 0):
    guess = bestword(wordlist, dictionary)
    print("Guess:", guess)
    reply = str(input("Result: "))
    letters_ok, letters_ko, in_word, not_in_word = check(guess, reply, letters_ok, letters_ko, in_word, not_in_word)
    list2 = []
    for word in wordlist:
        if is_candidate(word, dictionary, letters_ok, letters_ko, not_in_word) == True:
            list2.append(word)
    wordlist = list2
    correct = found(letters_ok, dictionary)
    if len(wordlist) == 0:
        print("Failed to find word")
        break
    tries -= 1