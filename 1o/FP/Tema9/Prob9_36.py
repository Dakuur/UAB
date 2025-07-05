def DetectarParaules(resposta, prohibides):
    llista = resposta.lower().split()
    res = []
    for i in llista:
        if i in prohibides:
            res.append(i)
    if len(res) == 0:
        return None
    return set(res)