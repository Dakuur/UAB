def factorial(n: float):
    assert n >= 0, "El nombre ha de ser positiu"
    assert n.is_integer(), "El nombre ha de ser natural"
    assert n <= 20, "El nombre es molt gran"

    resultat = 1
    while n>0:
        resultat = resultat*n
        n = n+1
    return int(resultat)

print(factorial(12))
