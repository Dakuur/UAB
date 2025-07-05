nomfitxer = str(input())

def mitjana_valors(nom_fitxer):
    try:
        with open(nom_fitxer, 'r') as f:
            valors = [int(valor) for valor in f.read().split()]
            mitjana = sum(valors) / len(valors)
            return mitjana
    except IOError:
        raise IOError("ERROR: Fitxer inexistent")
    except ValueError:
        raise ValueError("ERROR: Format de fitxer incorrecte")
    except ZeroDivisionError:
        raise ZeroDivisionError("ERROR: El fitxer no conté valors enters")
    except OverflowError:
        raise OverflowError("ERROR: Error indeterminat")

print("Mitjana:", mitjana_valors(nomfitxer))