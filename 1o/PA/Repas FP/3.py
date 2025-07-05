def elimina_duplicats(llista):
    duplicats = set()
    res = []
    for i in llista:
        if i not in duplicats:
            duplicats.add(i)
            res.append(i)
    return res